#include "indices.cuh"
#include "params.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <float.h>

// Define the warp size
const int WARP_SIZE = 32;

// Define a macro for squaring a value
#define SQR(x) ((x) * (x))

// Helper function for warp-level reduction
template <typename T>
__device__ inline T warpReduceSum(T val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Helper function for block-level reduction
template <typename T>
__inline__ __device__ float blockReduceSum(T* shared, T val, int tid, int tcount) {
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;

    val = warpReduceSum(val);  // Each warp performs partial reduction

    if (lane == 0)
        shared[wid] = val;  // Write reduced value to shared memory

    __syncthreads();  // Wait for all partial reductions

    // Read from shared memory only if that warp existed
    val = (tid < tcount / WARP_SIZE) ? shared[lane] : 0;

    if (wid == 0)
        val = warpReduceSum(val);  // Final reduce within the first warp

    return val;
}

// Helper function to calculate the absolute value of a real number raised to the power of two
__device__ __forceinline__ float abspow2(float& a) {
    return SQR(a);
}

// Integer logarithm base 2
template <typename IntType>
__device__ __inline__ uint ilog2(IntType n) {
    uint l;
    for (l = 0; n; n >>= 1, ++l);
    return l;
}

// Orthogonal transformation
template <typename T>
__device__ __inline__ void rotate(T& a, T& b) {
    T tmp = a;
    a = tmp + b;
    b = tmp - b;
}

// Fast Walsh-Hadamard transform
template <typename T>
__device__ __inline__ void fwht(T* data, uint n) {
    uint l2 = ilog2(n) - 1;
    for (uint i = 0; i < l2; ++i) {
        for (uint j = 0; j < n; j += (1 << (i + 1))) {
            for (uint k = 0; k < (1 << i); ++k) {
                rotate(data[j + k], data[j + k + (1 << i)]);
            }
        }
    }
}

// Helper function to calculate block addresses
__device__ inline void get_block_addresses(
    const uint2& start_point,
    const uint& patch_stack_size,
    const uint2& stacks_dim,
    const Params& params,
    uint2& outer_address,
    uint& start_idx) {
    // One block handles one patch_stack, data are in an array one after one
    start_idx = patch_stack_size * idx2(blockIdx.x, blockIdx.y, gridDim.x);

    outer_address.x = start_point.x + (blockIdx.x * params.p);
    outer_address.y = start_point.y + (blockIdx.y * params.p);

    // Ensure that the bottommost patches will be taken as reference patches regardless of the p parameter
    if (outer_address.y >= stacks_dim.y && outer_address.y < stacks_dim.y + params.p - 1) {
        outer_address.y = stacks_dim.y - 1;
    }

    // Ensure that the rightmost patches will be taken as reference patches regardless of the p parameter
    if (outer_address.x >= stacks_dim.x && outer_address.x < stacks_dim.x + params.p - 1) {
        outer_address.x = stacks_dim.x - 1;
    }
}

// Kernel to gather patches from the image based on matching stored in the 3D array stacks
__global__ void get_block(
    const uint2 start_point,
    const uchar* __restrict image,
    const ushort* __restrict stacks,
    const uint* __restrict g_num_patches_in_stack,
    float* patch_stack,
    const uint2 image_dim,
    const uint2 stacks_dim,
    const Params params) {
    uint start_idx;
    uint2 outer_address;
    get_block_addresses(start_point, params.k * params.k * (params.N + 1), stacks_dim, params, outer_address, start_idx);

    if (outer_address.x >= stacks_dim.x || outer_address.y >= stacks_dim.y) return;

    patch_stack += start_idx;

    const ushort* z_ptr = &stacks[idx3(0, blockIdx.x, blockIdx.y, params.N, gridDim.x)];

    uint num_patches = g_num_patches_in_stack[idx2(blockIdx.x, blockIdx.y, gridDim.x)];

    patch_stack[idx3(threadIdx.x, threadIdx.y, 0, params.k, params.k)] =
        (float)(image[idx2(outer_address.x + threadIdx.x, outer_address.y + threadIdx.y, image_dim.x)]);

    for (uint i = 0; i < num_patches; ++i) {
        int x = (int)((signed char)(z_ptr[i] & 0xFF));
        int y = (int)((signed char)((z_ptr[i] >> 8) & 0xFF));
        patch_stack[idx3(threadIdx.x, threadIdx.y, i + 1, params.k, params.k)] =
            (float)(image[idx2(outer_address.x + x + threadIdx.x, outer_address.y + y + threadIdx.y, image_dim.x)]);
    }
}

// Kernel to perform hard thresholding on transformed 3D patch stacks
__global__ void hard_treshold_block(
    const uint2 start_point,
    float* patch_stack,
    float* w_P,
    const uint* __restrict g_num_patches_in_stack,
    uint2 stacks_dim,
    const Params params,
    const uint sigma) {
    extern __shared__ float data[];

    int paramN = params.N + 1;
    uint tcount = blockDim.x * blockDim.y;
    uint tid = idx2(threadIdx.x, threadIdx.y, blockDim.x);
    uint patch_stack_size = tcount * paramN;

    uint start_idx;
    uint2 outer_address;
    get_block_addresses(start_point, patch_stack_size, stacks_dim, params, outer_address, start_idx);

    if (outer_address.x >= stacks_dim.x || outer_address.y >= stacks_dim.y) return;

    uint num_patches = g_num_patches_in_stack[idx2(blockIdx.x, blockIdx.y, gridDim.x)] + 1; // +1 for the reference patch.

    float* s_patch_stack = data + (tid * (num_patches + 1)); // +1 for avoiding bank conflicts
    patch_stack = patch_stack + start_idx + tid;

    // Load to shared memory
    for (uint i = 0; i < num_patches; ++i) {
        s_patch_stack[i] = patch_stack[i * patch_stack_size];
    }

    // Apply the hard thresholding
    if (threadIdx.x < num_patches) {
        float sum = 0.0f;
        for (uint i = 0; i < patch_stack_size; ++i) {
            sum += abspow2(s_patch_stack[i]);
        }
        float denoise_param = max(1.0f - (sigma * sigma) / (sum / patch_stack_size), 0.0f);
        w_P[threadIdx.x] = denoise_param;

        if (denoise_param == 0.0f) {
            for (uint i = 0; i < patch_stack_size; ++i) {
                patch_stack[i] = 0.0f;
            }
        }
    }
}

// Kernel to aggregate patches using weighted average
__global__ void aggregate_block(
    const uint2 start_point,
    const float* __restrict patch_stack,
    const float* __restrict w_P,
    const ushort* __restrict stacks,
    const float* __restrict kaiser_window,
    float* numerator,
    float* denominator,
    const uint* __restrict g_num_patches_in_stack,
    const uint2 image_dim,
    const uint2 stacks_dim,
    const Params params) {
    // Your existing code for the aggregate_block kernel
}

// Kernel to compute the final aggregated image
__global__ void aggregate_final(
    const float* __restrict numerator,
    const float* __restrict denominator,
    const uint2 image_dim,
    uchar* image_o) {
    // Your existing code for the aggregate_final kernel
}

// Kernel to compute Wiener filter coefficients
__global__ void wiener_filtering(
    const uint2 start_point,
    float* patch_stack,
    const float* __restrict patch_stack_basic,
    float* w_P,
    const uint* __restrict num_patches_in_stack,
    uint2 stacks_dim,
    const Params params,
    const uint sigma) {
    // Your existing code for the wiener_filtering kernel
}

// Wrapper function to run the get_block kernel
extern "C" void run_get_block(
    const uint2 start_point,
    const uchar* __restrict image,
    const ushort* __restrict stacks,
    const uint* __restrict num_patches_in_stack,
    float* patch_stack,
    const uint2 image_dim,
    const uint2 stacks_dim,
    const Params params,
    const dim3 num_threads,
    const dim3 num_blocks) {
    get_block<<<num_blocks, num_threads>>>(
        start_point,
        image,
        stacks,
        num_patches_in_stack,
        patch_stack,
        image_dim,
        stacks_dim,
        params);
    cudaDeviceSynchronize();
    cudaCheckError();
}

int main() {
    // Initialize parameters, allocate memory, and manage device/host data transfers

    // Define image dimensions and other parameters

    // Launch the get_block kernel to gather patches

    // Launch the hard_treshold_block kernel to perform hard thresholding

    // Launch the aggregate_block kernel to aggregate patches

    // Launch the aggregate_final kernel to compute the final image

    // Launch the wiener_filtering kernel to compute Wiener filter coefficients

    // Free allocated memory and clean up

    return 0;
}


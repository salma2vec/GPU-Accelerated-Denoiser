#include "params.hpp"
#include "indices.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>

// Nearest lower power of 2
__device__ __inline__ uint nearestLowerPowerOf2(uint x) {
    return (0x80000000u >> __clz(x));
}

// Computes the squared difference between two numbers
template <typename T>
__device__ __inline__ T squaredDifference(const T i1, const T i2) {
    T diff = i1 - i2;
    return diff * diff;
}

// Adds a new patch to the patch stack (only N most similar are kept)
__device__
void addToMatchedImage(uint* stack, uchar* numPatchesInStack, const uint value, const Params& params) {
    // stack[numPatchesInStack - 1] is the most similar (lowest number)
    int k;

    uchar num = (*numPatchesInStack);
    if (num < params.N) { // Add a new value
        k = num++;
        while (k > 0 && value > stack[k - 1]) {
            stack[k] = stack[k - 1];
            --k;
        }
        stack[k] = value;
        *numPatchesInStack = num;
    } else if (value >= stack[0]) {
        return;
    } else { // Delete the highest value and add a new one
        k = 1;
        while (k < params.N && value < stack[k]) {
            stack[k - 1] = stack[k];
            k++;
        }
        stack[k - 1] = value;
    }
}

// Block-matching algorithm
__global__
void blockMatching(const uchar* __restrict image, ushort* gStacks, uint* gNumPatchesInStack, const uint2 imageDim,
                   const uint2 stacksDim, const Params params, const uint2 startPoint) {
    // One block is processing warpSize patches (because each warp is computing the distance of the same warpSize patches
    // from different displaced patches)
    int tid = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    int numWarps = blockDim.x / warpSize;

    // pBlock denotes the reference rectangle on which the current CUDA block is computing
    uint pRectangleWidth = ((warpSize - 1) * params.p) + params.k;
    uint pRectangleStart = startPoint.x + blockIdx.x * warpSize * params.p;

    // Shared arrays
    extern __shared__ uint sData[];
    uint* sDiff = (uint*)&sData; // SIZE: pRectangleWidth * numWarps
    uint* sStacks = (uint*)&sData[pRectangleWidth * numWarps]; // SIZE: params.N * numWarps * warpSize
    uchar* sPatchesInStack = (uchar*)&sData[numWarps * (pRectangleWidth + params.N * warpSize)]; // SIZE: numWarps * warpSize
    uchar* sImageP = (uchar*)&sPatchesInStack[numWarps * warpSize]; // SIZE: pRectangleWidth * params.k

    sDiff += idx2(0, wid, pRectangleWidth);

    // Initialize sPatchesInStack to zero
    sPatchesInStack[idx2(tid, wid, warpSize)] = 0;

    int2 p; // Address of the reference patch
    int2 q; // Address of the patch against which the difference is computed

    p.x = pRectangleStart + (tid * params.p);
    p.y = startPoint.y + (blockIdx.y * params.p);

    // Ensure that the bottom-most patches will be taken as reference patches regardless of the p parameter
    if (p.y >= stacksDim.y && p.y < stacksDim.y + params.p - 1) {
        p.y = stacksDim.y - 1;
    } else if (p.y >= stacksDim.y) return;

    // Ensure that the right-most patches will be taken as reference patches regardless of the p parameter
    uint innerPX = tid * params.p;
    if (p.x >= stacksDim.x && p.x < stacksDim.x + params.p - 1) {
        innerPX -= (p.x - (stacksDim.x - 1));
        p.x = stacksDim.x - 1;
    }

    // Load reference patches needed by the actual block into shared memory
    for (int i = threadIdx.x; i < pRectangleWidth * params.k; i += blockDim.x) {
        int sx = i % pRectangleWidth;
        int sy = i / pRectangleWidth;
        if (pRectangleStart + sx >= imageDim.x) continue;
        sImageP[i] = image[idx2(pRectangleStart + sx, p.y + sy, imageDim.x)];
    }

    __syncthreads();

    // Scale difference so that it can fit in ushort
    uint shift = (__clz(params.Tn) < 16u) ? 16u - (uint)__clz(params.Tn) : 0;

    // Ensure that displaced patch coordinates (q) will be positive
    int2 from;
    from.y = (p.y - (int)params.n < 0) ? -p.y : -(int)params.n;
    from.x = (((int)pRectangleStart) - (int)params.n < 0) ? -((int)pRectangleStart) : -(int)params.n;
    from.x += wid;

    // For each displacement (x,y) in the n neighborhood
    for (int y = from.y; y <= (int)params.n; ++y) {
        q.y = p.y + y;
        if (q.y >= stacksDim.y) break;

        for (int x = from.x; x <= (int)params.n; x += numWarps) {
            // Reference patch is always the most similar to itself (there is no need to compute it)
            if (x == 0 && y == 0) continue;

            // Each warp is computing the same patch with slightly different displacement
            // Compute the distance of the reference patch p from the current patch q, which is displaced by (x+tid, y)

            // qBlock denotes the displaced rectangle processed by the current warp
            uint qRectangleStart = pRectangleStart + x;
            q.x = qRectangleStart + innerPX;

            // Compute the distance for each column of the reference patch
            for (uint i = tid; i < pRectangleWidth && pRectangleStart + i < imageDim.x && qRectangleStart + i < imageDim.x;
                i += warpSize) {
                uint dist = 0;
                for (uint iy = 0; iy < params.k; ++iy) {
                    dist += squaredDifference((int)sImageP[idx2(i, iy, pRectangleWidth)],
                        (int)image[idx2(qRectangleStart + i, q.y + iy, imageDim.x)]);
                }
                sDiff[i] = dist;
            }

            if (p.x >= stacksDim.x || q.x >= stacksDim.x) continue;

            // Sum column distances to obtain patch distance
            uint diff = 0;
            for (uint i = 0; i < params.k; ++i)
                diff += sDiff[innerPX + i];

            // Distance threshold
            if (diff < params.Tn) {
                uint locY = (uint)((q.y - p.y) & 0xFF); // Relative location y (-127 to 127)
                uint locX = (uint)((q.x - p.x) & 0xFF); // Relative location x (-127 to 127)
                diff >>= shift;
                diff <<= 16u; // [..DIFF(ushort)..|..LOC_Y(sbyte)..|..LOC_X(sbyte)..]
                diff |= (locY << 8u);
                diff |= locX;

                // Add the current patch to sStacks
                addToMatchedImage(&sStacks[params.N * idx2(tid, wid, warpSize)],
                    &sPatchesInStack[idx2(tid, wid, warpSize)], diff, params);
            }
        }
    }

    __syncthreads();

    uint batchSize = gridDim.x * warpSize;
    uint blockAddressX = blockIdx.x * warpSize + tid;

    if (wid > 0) return;
    // Select N most similar patches for each reference patch from stacks in shared memory and save them to global memory
    // Each thread represents one reference patch
    // Each thread will find N most similar blocks in numWarps stacks (which were computed by different warps) and save them into global memory
    // In shared memory, the most similar patch is at the end, in global memory, the order does not matter
    // DEV: performance impact approximately 8%
    if (p.x >= stacksDim.x) return;

    int j;
    for (j = 0; j < params.N; ++j) {
        uint count = 0;
        uint minIdx = 0;
        uint minVal = 0xFFFFFFFF; // INF

        // Find the patch with the minimal value of remaining
        for (int i = minIdx; i < numWarps; ++i) {
            count = (uint)sPatchesInStack[idx2(tid, i, warpSize)];
            if (count == 0) continue;

            uint newMinVal = sStacks[idx3(count - 1, tid, i, params.N, warpSize)];
            if (newMinVal < minVal) {
                minVal = newMinVal;
                minIdx = i;
            }
        }
        if (minVal == 0xFFFFFFFF) break; // All stacks are empty

        // Remove the patch from the shared stack
        sPatchesInStack[idx2(tid, minIdx, warpSize)]--;

        // Add the patch to the stack in global memory
        gStacks[idx3(j, blockAddressX, blockIdxY, params.N, batchSize)] = (ushort)(minVal & 0xFFFF);
    }
    // Save to the global memory the number of similar patches rounded to the nearest lower power of two
    gNumPatchesInStack[idx2(blockAddressX, blockIdxY, batchSize)] = nearestLowerPowerOf2(static_cast<uint>(j) + 1) - 1;
}

extern "C"
void runBlockMatching(const uchar* __restrict image, ushort* stacks, uint* numPatchesInStack, const uint2 imageDim,
                     const uint2 stacksDim, const Params params, const uint2 startPoint, const dim3 numThreads,
                     const dim3 numBlocks, const uint sharedMemorySize) {
    blockMatching << <numBlocks, numThreads, sharedMemorySize >> > (image, stacks, numPatchesInStack, imageDim, stacksDim, params, startPoint);
}


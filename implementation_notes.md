## General Pipeline
**Step 1: Obtaining the Basic Estimate**
- The image is divided into fixed-size blocks.
- Similar blocks near each reference block are searched for.
- A 3D transform is applied to obtain coefficients, hard thresholding is performed, and an inverse transform is applied to obtain an estimate for each block.
- The weight for each block within a group is calculated.
- The basic estimate for the image block is computed as the weighted average of corresponding blocks in each group.

**Step 2: Obtaining the Final Estimate**
- The basic estimate is utilized to group blocks by similarity through a similarity test. Grouping is done according to indices and original image data.
- Wiener shrinkage coefficients and their weights for each group are computed.
- The coefficients are applied to the transformed image, and an inverse transform is applied to obtain estimates for each block.
- The final estimate for the image block is calculated as the weighted average of corresponding blocks in each group.

## Implementation Details
- Steps 1 and 2 are interdependent, as Step 2 relies on the basic estimate obtained in Step 1.
- The image is divided into blocks and block transformations are precomputed.
- Either DCT or Haar transformations can be used.
- Similar blocks can be searched for either across the entire image or within restricted neighborhoods.
- Buffers are allocated for precomputed 2D transformations and basic estimates.
- Border effects are addressed, and color images are handled.
- A heap is used to maintain the N most similar patches for the current reference patch.
- Aggregation is implemented to prevent contention in per-pixel atomic add operations.
- Convolution can be considered for block matching.

## Problems
- Initialization time can be substantial, with CUFFT plan creation being time-consuming.
- The choice between DCT and Haar transformations needs to be made.

## Tricks
- .cu and .cpp files are distinct, and they do not link global variables together, so "extern" will not work.
- Device functions are written in separate files and included in the main .cu file.
- Data is rearranged for optimized 1D transforms.
- The time-consuming block matching and aggregation steps are optimized.

## To-Do
- Implement 2D transform and hard thresholding using CUFFT.
- Perform batch mode CUFFT transformations.
- Optimize the aggregation process.

## Summary
The state-of-the-art image denoising algorithm, Block Matching and 3D Filtering (BM3D), has been implemented in CUDA on NVIDIA GPUs. Performance has been compared with OpenCV and other open-source CUDA implementations. The implementation is suitable for real-time video denoising.

## Background
### Algorithm Description
BM3D is an image denoising algorithm based on collaborative filtering in the transform domain. It consists of three stages:

1. **Block Matching:** Similar patches in the 2D image are grouped into a 3D data array called a "group."
2. **Collaborative Filtering:** A 3D transform is applied to the group to create a sparse representation in the frequency domain, followed by filtering. An inverse transform is used to restore the filtered data to the image domain, resulting in a noise-free 3D array.
3. **Image Reconstruction:** The groups are redistributed to their original positions.

The algorithm runs this three-step procedure twice. In the first run, the noisy image is processed with hard thresholding to remove low-frequency noise. The inverse transform is then applied to construct the estimated image.

In the second run, the basic estimate is used to determine basic estimate groups by a similarity test. Wiener filter coefficients and weights are computed for each group, and they are applied to the transformed image. The inverse transform is then used to generate the final output image.

### Parallel Options
Block matching is a naturally parallelizable step in the algorithm. Each thread can be assigned a reference patch to search for similar patches in the search region, minimizing synchronization and communication costs. For the Fourier transform, CUDA offers optimized libraries, simplifying the implementation. However, reconstruction from groups may require synchronization due to shared data, and atomic operations such as atomicAdd may be needed.

Both block matching and aggregation are time-consuming steps in this approach. The aim is to optimize these steps to achieve better performance.

### Approach

**FFT**

CUDA's fast Fourier transform (FFT) library is utilized, including 1D, 2D, and 3D transformations. A transformation plan is created to allocate GPU memory and perform initialization, and then the plan is applied to the data for rapid computation. CUFFT supports batch mode for efficient transformations. While FFT is essential for BM3D, different transformation approaches, including separate 2D transformations and 1D transformations across patches, have been experimented with. It has been found that 3D transformation with CUFFT is highly optimized and efficient. By setting the batch size to the number of groups, only one CUFFT plan is needed, reducing initialization time and computation overhead.

To save time during the initialization of a single plan, experiments have been conducted with creating two plans (2D and 1D) or calling the API multiple times for the same plan, but it has been found that a 3D transformation with one plan is the most efficient approach.

Optimizing the 3D transformation process and applying batch transformations for multiple image patches are essential in this approach. The 3D transformation using CUFFT is highly efficient and minimizes overhead.

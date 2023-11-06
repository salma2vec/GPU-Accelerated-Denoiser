# GPU-Accelerated Denoiser

![BM3D Scheme](https://github.com/IdealisticINTJ/GPU-Accelerated-Denoiser/assets/65449934/1c900438-e871-43f5-929e-da1abcedb195)

## Introduction

Here we present a CUDA-accelerated implementation of the Block Matching and 3D Filtering (BM3D) image denoising method. This project offers a high-performance solution for image denoising, harnessing the computational power of NVIDIA GPUs.

**Author**: Salma Shaik <<salma.20bce7605@vitap.ac.in>>

## Summary

In this project, the state-of-the-art image denoising algorithm, Block Matching and 3D Filtering (BM3D), has been successfully implemented in CUDA on NVIDIA GPUs. The implementation underwent rigorous testing and comparison with other open-source alternatives, including OpenCV. It showcases a remarkable 20% speedup compared to the latter and demonstrates real-time video denoising capabilities.

## Background

The BM3D algorithm is a pioneering approach to image denoising that relies on collaborative filtering in the transform domain. Since its introduction in 2007, BM3D has remained the state-of-the-art method. The algorithm encompasses two steps, each comprising three main stages:

### Step 1: Block Matching
1. **Block Matching**: Grouping similar patches in the 2D image into a 3D data array, forming what is referred to as a "group."
2. **Collaborative Filtering**: Applying a 3D transform to the group, producing a sparse representation in the transform domain that is then filtered. An inverse transformation is used to convert the filtered data back into the image domain.
3. **Image Reconstruction**: Redistributing the patches within each group to their original positions, with each pixel potentially undergoing multiple updates.

### Step 2: Final Estimate
The algorithm repeats the three-step procedure twice. In the initial run, the noisy image undergoes processing with hard thresholding in the sparse transform, resulting in an estimated image. In the subsequent run, the same procedure is repeated with a Wiener filter replacing hard thresholding to obtain Wiener coefficients. These coefficients are then applied to the original images. The second run assumes that the energy spectrum of the first output is accurate, rendering it more efficient than hard thresholding.

## Approach

### Block Matching
The noisy image is partitioned into a set of overlapping reference patches using a sliding-window approach. Each patch has a size of 8x8 with a default stride of 3. A local window of 64x64 around the reference patch is used to search for patches that closely match the reference patch.

For example, in a 512x512 input image, a total of 28,561 reference patches are generated. Each CUDA thread is assigned a reference patch, and within each thread, a local window of 64x64 is used to find the closest matching patches. The distance metric employed for matching is the L2-distance in pixel space, simplifying computation and implementation.

![Block Matching Example](https://github.com/IdealisticINTJ/GPU-Accelerated-Denoiser/assets/65449934/a6582b80-5e6f-4150-9786-af65a4a7639a)

The result is a stack of patches containing the closest matching patches for each reference patch.

## Implementation Details

BM3D is a recent denoising method based on the fact that an image has a locally sparse representation in the transform domain. This sparsity is enhanced by grouping similar 2D image patches into 3D groups. In this paper, we propose an open-source implementation of the method. We discuss the choice of all parameter methods and confirm their actual optimality. The description of the method is rewritten with a new notation. We hope this new notation is more transparent than in the original paper. A final index gives nonetheless the correspondence between the new notation and the original notation.

## Reference Papers
1. [The BM3D Algorithm paper used](https://www.cs.tut.fi/~foi/GCF-BM3D/BM3D_TIP_2007.pdf)
2. [Video denoising by sparse 3D transform-domain collaborative filtering](https://www.researchgate.net/publication/242187593_Video_denoising_by_sparse_3D_transform-domain_collaborative_filtering)
3. [An Analysis and Implementation of the BM3D Image Denoising Method](https://www.researchgate.net/publication/289990721_An_Analysis_and_Implementation_of_the_BM3D_Image_Denoising_Method)
4. [Adaptive BM3D Algorithm for Image Denoising Using Coefficient of Variation](https://www.semanticscholar.org/paper/Adaptive-BM3D-Algorithm-for-Image-Denoising-Using-Song-Duan/83f081604196971ca3b0841a18953ec1aefaa3e8?utm_source=direct_link)

## Installation and Usage

For comprehensive instructions on the project and how to use BM3D-GPU, please refer to the [Documentation](implementation_notes.md).

## Benchmarks

To view performance benchmarks and comparisons with other implementations, please visit the [Benchmarks](benchmarks/) section.

## License

This project is licensed under the [MIT License](LICENSE) - see the [LICENSE](LICENSE) file for details.

---

*Note: If you wish to contribute or report issues, please refer to our [Contribution Guidelines](CONTRIBUTING.md).*


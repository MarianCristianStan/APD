#include <cuda_runtime.h>
#include <iostream>

__global__ void compareImagesKernel(uint8_t* img1, uint8_t* img2, int width, int height, int channels, int* matchingPixels) {
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;

   if (x < width && y < height) {
      int match = 1;
      for (int c = 0; c < channels; ++c) {
         if (abs(img1[(y * width + x) * channels + c] - img2[(y * width + x) * channels + c]) > 40) {
            match = 0;
            break;
         }
      }
      if (match) {
         atomicAdd(matchingPixels, 1);
      }
   }
}

extern "C" double compareImagesCUDA(uint8_t * img1, uint8_t * img2, int width, int height, int channels) {
   uint8_t* d_img1, * d_img2;
   int* d_matchingPixels;
   size_t size = width * height * channels * sizeof(uint8_t);
   int matchingPixels = 0;

   cudaMalloc(&d_img1, size);
   cudaMalloc(&d_img2, size);
   cudaMalloc(&d_matchingPixels, sizeof(int));

   cudaMemcpy(d_img1, img1, size, cudaMemcpyHostToDevice);
   cudaMemcpy(d_img2, img2, size, cudaMemcpyHostToDevice);
   cudaMemcpy(d_matchingPixels, &matchingPixels, sizeof(int), cudaMemcpyHostToDevice);

   dim3 threadsPerBlock(16, 16);
   dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

   compareImagesKernel << <numBlocks, threadsPerBlock >> > (d_img1, d_img2, width, height, channels, d_matchingPixels);

   cudaMemcpy(&matchingPixels, d_matchingPixels, sizeof(int), cudaMemcpyDeviceToHost);

   cudaFree(d_img1);
   cudaFree(d_img2);
   cudaFree(d_matchingPixels);

   int totalPixels = width * height;
   double similarity = (static_cast<double>(matchingPixels) / totalPixels) * 100.0;
   return similarity;
}

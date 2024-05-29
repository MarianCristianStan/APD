#include <cuda_runtime.h>

__global__ void rotate90Kernel(uint8_t* input, uint8_t* output, int width, int height, int channels) {
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;

   if (x < width && y < height) {
      for (int c = 0; c < channels; ++c) {
         int newX = y;
         int newY = width - x - 1;
         output[(newY * height + newX) * channels + c] = input[(y * width + x) * channels + c];
      }
   }
}

extern "C" void rotate90CUDA(uint8_t * input, uint8_t * output, int width, int height, int channels) {
   uint8_t* d_input, * d_output;
   size_t size = width * height * channels * sizeof(uint8_t);

   cudaMalloc(&d_input, size);
   cudaMalloc(&d_output, size);

   cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

   dim3 threadsPerBlock(16, 16);
   dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

   rotate90Kernel << <numBlocks, threadsPerBlock >> > (d_input, d_output, width, height, channels);

   cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

   cudaFree(d_input);
   cudaFree(d_output);
}

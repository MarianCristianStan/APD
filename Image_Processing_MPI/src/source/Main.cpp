
#include <iostream>
#include <cmath>
#include <mpi.h>
#include "Image.h"



void rotate90MPI(Image& img, int rank, int size) {
   if (img.getData() == nullptr) {
      if (rank == 0) {
         std::cerr << "No image data loaded. Cannot rotate." << std::endl;
      }
      return;
   }

   int width = img.getWidth();
   int height = img.getHeight();
   int channels = img.getChannels();

   int local_height = height / size;
   int start_row = rank * local_height;
   int end_row = (rank == size - 1) ? height : start_row + local_height;

   uint8_t* localRotatedData = new uint8_t[local_height * width * channels];
   uint8_t* rotatedData = nullptr;

   if (rank == 0) {
      rotatedData = new uint8_t[width * height * channels];
   }

   for (int y = start_row; y < end_row; ++y) {
      for (int x = 0; x < width; ++x) {
         for (int c = 0; c < channels; ++c) {
            int newX = y;
            int newY = width - x - 1;
            localRotatedData[((y - start_row) * width + x) * channels + c] = img.getData()[(y * width + x) * channels + c];
         }
      }
   }

   MPI_Gather(localRotatedData, local_height * width * channels, MPI_UINT8_T, rotatedData, local_height * width * channels, MPI_UINT8_T, 0, MPI_COMM_WORLD);

   if (rank == 0) {
      // Correctly place the gathered data into the final rotated image
      uint8_t* finalRotatedData = new uint8_t[width * height * channels];
      for (int y = 0; y < height; ++y) {
         for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
               int newX = y;
               int newY = width - x - 1;
               finalRotatedData[(newY * height + newX) * channels + c] = rotatedData[(y * width + x) * channels + c];
            }
         }
      }

      img.setWidth(height);
      img.setHeight(width);

      delete[] img.getData();
      img.setData(finalRotatedData);
      img.write("C:/Users/smari/OneDrive/Desktop/Apd Proiect MPI/Image_Processing_MPI/output/image1_write.jpg");

      delete[] rotatedData;
   }

   delete[] localRotatedData;
}

double compareImagesMPI(const Image& img1, const Image& img2, int rank, int size) {
   if (img1.getWidth() != img2.getWidth() || img1.getHeight() != img2.getHeight() || img1.getChannels() != img2.getChannels()) {
      if (rank == 0) {
         std::cerr << "Images have different dimensions or channels. Cannot compare.\n";
      }
      return -1.0;
   }

   int width = img1.getWidth();
   int height = img1.getHeight();
   int channels = img1.getChannels();

   uint8_t* data1 = img1.getData();
   uint8_t* data2 = img2.getData();

   int totalPixels = width * height;
   int local_totalPixels = totalPixels / size;
   int start_pixel = rank * local_totalPixels;
   int end_pixel = (rank == size - 1) ? totalPixels : start_pixel + local_totalPixels;

   int matchingPixels = 0;

   for (int i = start_pixel * channels; i < end_pixel * channels; i += channels) {
      bool match = true;
      for (int c = 0; c < channels; ++c) {
         if (std::abs(data1[i + c] - data2[i + c]) > 40) {
            match = false;
            break;
         }
      }
      if (match) {
         ++matchingPixels;
      }
   }
   std::cout << "machingPixels found by Process: "<<rank<<" "<< matchingPixels<<"\n";
   int global_matchingPixels;
   MPI_Reduce(&matchingPixels, &global_matchingPixels, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  
   if (rank == 0) {
      double similarity = (static_cast<double>(global_matchingPixels) / totalPixels) * 100.0;
      return similarity;
   }
   else {
      return 0.0;
   }
}

int main(int argc, char** argv) {
   MPI_Init(&argc, &argv);

   int rank, size;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   Image img1, img2, rotateImg;
   int dimensions[3][3] = { {0, 0, 0}, {0, 0, 0}, {0, 0, 0} };

   if (rank == 0) {
      const char* img1_path = "C:/Users/smari/OneDrive/Desktop/Apd Proiect MPI/Image_Processing_MPI/input/test_8k_1.jpg";
      const char* img2_path = "C:/Users/smari/OneDrive/Desktop/Apd Proiect MPI/Image_Processing_MPI/input/test_8k_2.jpg";
      const char* rotate_path = "C:/Users/smari/OneDrive/Desktop/Apd Proiect MPI/Image_Processing_MPI/input/test_8k_1.jpg";

      FILE* img1_file = nullptr, * img2_file = nullptr, * rotate_file = nullptr;
      errno_t err;

      // Open image files
      err = fopen_s(&img1_file, img1_path, "rb");
      if (err != 0) {
         perror("Error opening image 1");
         std::cerr << "Error code: " << err << std::endl;
         MPI_Abort(MPI_COMM_WORLD, 1);
         return 1;
      }
      err = fopen_s(&img2_file, img2_path, "rb");
      if (err != 0) {
         perror("Error opening image 2");
         std::cerr << "Error code: " << err << std::endl;
         MPI_Abort(MPI_COMM_WORLD, 1);
         return 1;
      }

      err = fopen_s(&rotate_file, rotate_path, "rb");
      if (err != 0) {
         perror("Error opening image rotate");
         std::cerr << "Error code: " << err << std::endl;
         MPI_Abort(MPI_COMM_WORLD, 1);
         return 1;
      }

      // Read images
      if (!img1.readFromFile(img1_file) || !img2.readFromFile(img2_file) || !rotateImg.readFromFile(rotate_file)) {
         std::cerr << "Failed to read image files." << std::endl;
         MPI_Abort(MPI_COMM_WORLD, 1);
         return 1;
      }

      fclose(img1_file);
      fclose(img2_file);
      fclose(rotate_file);

   
     
      dimensions[0][0] = img1.getWidth();
      dimensions[0][1] = img1.getHeight();
      dimensions[0][2] = img1.getChannels();

      dimensions[1][0] = img2.getWidth();
      dimensions[1][1] = img2.getHeight();
      dimensions[1][2] = img2.getChannels();

      dimensions[2][0] = rotateImg.getWidth();
      dimensions[2][1] = rotateImg.getHeight();
      dimensions[2][2] = rotateImg.getChannels();
   }

   MPI_Bcast(dimensions, 9, MPI_INT, 0, MPI_COMM_WORLD);

   if (rank != 0) {
      img1.allocateMemory(dimensions[0][0], dimensions[0][1], dimensions[0][2]);
      img2.allocateMemory(dimensions[1][0], dimensions[1][1], dimensions[1][2]);
      rotateImg.allocateMemory(dimensions[2][0], dimensions[2][1], dimensions[2][2]);
   }

   MPI_Bcast(img1.getData(), dimensions[0][0] * dimensions[0][1] * dimensions[0][2], MPI_UINT8_T, 0, MPI_COMM_WORLD);
   MPI_Bcast(img2.getData(), dimensions[1][0] * dimensions[1][1] * dimensions[1][2], MPI_UINT8_T, 0, MPI_COMM_WORLD);
   MPI_Bcast(rotateImg.getData(), dimensions[2][0] * dimensions[2][1] * dimensions[2][2], MPI_UINT8_T, 0, MPI_COMM_WORLD);

 

   MPI_Barrier(MPI_COMM_WORLD);

   auto rotateStart = MPI_Wtime();
   rotate90MPI(rotateImg, rank, size);
   auto rotateEnd = MPI_Wtime();
   auto rotateDuration = (rotateEnd - rotateStart) * 1000;

   MPI_Barrier(MPI_COMM_WORLD);

   auto compareStart = MPI_Wtime();
   double similarity = compareImagesMPI(img1, img2, rank, size);
   auto compareEnd = MPI_Wtime();
   auto compareDuration = (compareEnd - compareStart) * 1000;

   if (rank == 0) {
      std::cout << "Rotation duration: " << rotateDuration << " milliseconds" << std::endl;
      std::cout << "Compare duration: " << compareDuration << " milliseconds" << std::endl;
      std::cout << "Similarity between the two images: " << similarity << "%" << std::endl;
   }

   MPI_Finalize();
   return 0;
}

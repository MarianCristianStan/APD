<<<<<<< HEAD
#include "ImageUtils.h"
#include <iostream>
#include <cmath>

void rotate90(Image& img) {
   if (img.getData() == nullptr) {
      std::cerr << "No image data loaded. Cannot rotate." << std::endl;
      return;
   }
   int width = img.getWidth();
   int height = img.getHeight();
   int channels = img.getChannels();

   uint8_t* rotatedData = new uint8_t[width * height * channels];

   for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
         for (int c = 0; c < channels; ++c) {
            int newX = y;
            int newY = width - x - 1;
            rotatedData[(newY * height + newX) * channels + c] = img.getData()[(y * width + x) * channels + c];
         }
      }
   }

   img.setWidth(height);
   img.setHeight(width);

   delete[] img.getData();
   img.setData(rotatedData);
   img.write(img.getFilename());

}


double compareImages(const Image& img1, const Image& img2) {

   if (img1.getWidth() != img2.getWidth() || img1.getHeight() != img2.getHeight() || img1.getChannels() != img2.getChannels()) {
      std::cerr << "Images have different dimensions or channels. Cannot compare.\n";
      return -1.0;
   }

   int width = img1.getWidth();
   int height = img1.getHeight();

   int channels = img1.getChannels();

   uint8_t* data1 = img1.getData();
   uint8_t* data2 = img2.getData();

   int totalPixels = width * height;
   int matchingPixels = 0;


   for (int i = 0; i < totalPixels * channels; i += channels) {
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
   double similarity = (static_cast<double>(matchingPixels) / totalPixels) * 100.0;
   return similarity;
=======
#include "ImageUtils.h"
#include <iostream>
#include <cmath>

void rotate90(Image& img) {
   if (img.getData() == nullptr) {
      std::cerr << "No image data loaded. Cannot rotate." << std::endl;
      return;
   }
   int width = img.getWidth();
   int height = img.getHeight();
   int channels = img.getChannels();

   uint8_t* rotatedData = new uint8_t[width * height * channels];

   for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
         for (int c = 0; c < channels; ++c) {
            int newX = y;
            int newY = width - x - 1;
            rotatedData[(newY * height + newX) * channels + c] = img.getData()[(y * width + x) * channels + c];
         }
      }
   }

   img.setWidth(height);
   img.setHeight(width);

   delete[] img.getData();
   img.setData(rotatedData);
   img.write(img.getFilename());

}


double compareImages(const Image& img1, const Image& img2) {

   if (img1.getWidth() != img2.getWidth() || img1.getHeight() != img2.getHeight() || img1.getChannels() != img2.getChannels()) {
      std::cerr << "Images have different dimensions or channels. Cannot compare.\n";
      return -1.0;
   }

   int width = img1.getWidth();
   int height = img1.getHeight();

   int channels = img1.getChannels();

   uint8_t* data1 = img1.getData();
   uint8_t* data2 = img2.getData();

   int totalPixels = width * height;
   int matchingPixels = 0;


   for (int i = 0; i < totalPixels * channels; i += channels) {
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
   double similarity = (static_cast<double>(matchingPixels) / totalPixels) * 100.0;
   return similarity;
>>>>>>> 981f099a06b88634e0cf05a82a4a2cd9535179cd
}
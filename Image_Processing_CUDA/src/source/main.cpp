<<<<<<< HEAD
﻿#include <iostream>
#include "Image.h"
#include <chrono>

extern "C" void rotate90CUDA(uint8_t * input, uint8_t * output, int width, int height, int channels);
extern "C" double compareImagesCUDA(uint8_t * img1, uint8_t * img2, int width, int height, int channels);

using namespace std;
using namespace chrono;

void rotate90(Image& img) {
   if (img.getData() == nullptr) {
      std::cerr << "No image data loaded. Cannot rotate." << std::endl;
      return;
   }

   int width = img.getWidth();
   int height = img.getHeight();
   int channels = img.getChannels();

   uint8_t* rotatedData = new uint8_t[width * height * channels];

   rotate90CUDA(img.getData(), rotatedData, width, height, channels);

   img.setWidth(height);
   img.setHeight(width);

   delete[] img.getData();
   img.setData(rotatedData);
   img.write("../output/8k_rotate.jpg");
}

double compareImages(const Image& img1, const Image& img2) {
   if (img1.getWidth() != img2.getWidth() || img1.getHeight() != img2.getHeight() || img1.getChannels() != img2.getChannels()) {
      cerr << "Images have different dimensions or channels. Cannot compare." << endl;
      return -1.0;
   }

   return compareImagesCUDA(img1.getData(), img2.getData(), img1.getWidth(), img1.getHeight(), img1.getChannels());
}

int main(int argc, char** argv) {
   Image img1("../input/image1.png");
   Image img2("../input/image2.png");
   Image rotateImg("../input/8k.jpg");

   if (img1.getData() == nullptr || img2.getData() == nullptr || rotateImg.getData() == nullptr) {
      cerr << "Failed to load images. Exiting." << endl;
      return 1;
   }
   auto rotateStart = high_resolution_clock::now();
   rotate90(rotateImg);
   auto rotateEnd = high_resolution_clock::now();
   duration<float, milli> rotateDuration = rotateEnd - rotateStart;
   cout << "Rotation duration: " << rotateDuration.count() << " milliseconds" << endl;

   auto compareStart = high_resolution_clock::now();
   double similarity = compareImages(img1, img2);
   auto compareEnd = high_resolution_clock::now();
   duration<float, milli> compareDuration = compareEnd - compareStart;
   cout << "Compare duration: " << compareDuration.count() << " milliseconds" << endl;
   cout << "Similarity between the two images: " << similarity << "%" << endl;

   return 0;
}
=======
﻿#include <iostream>
#include "Image.h"
#include <chrono>

extern "C" void rotate90CUDA(uint8_t * input, uint8_t * output, int width, int height, int channels);
extern "C" double compareImagesCUDA(uint8_t * img1, uint8_t * img2, int width, int height, int channels);

using namespace std;
using namespace chrono;

void rotate90(Image& img) {
   if (img.getData() == nullptr) {
      std::cerr << "No image data loaded. Cannot rotate." << std::endl;
      return;
   }

   int width = img.getWidth();
   int height = img.getHeight();
   int channels = img.getChannels();

   uint8_t* rotatedData = new uint8_t[width * height * channels];

   rotate90CUDA(img.getData(), rotatedData, width, height, channels);

   img.setWidth(height);
   img.setHeight(width);

   delete[] img.getData();
   img.setData(rotatedData);
   img.write("../output/8k_rotate.jpg");
}

double compareImages(const Image& img1, const Image& img2) {
   if (img1.getWidth() != img2.getWidth() || img1.getHeight() != img2.getHeight() || img1.getChannels() != img2.getChannels()) {
      cerr << "Images have different dimensions or channels. Cannot compare." << endl;
      return -1.0;
   }

   return compareImagesCUDA(img1.getData(), img2.getData(), img1.getWidth(), img1.getHeight(), img1.getChannels());
}

int main(int argc, char** argv) {
   Image img1("../input/image1.png");
   Image img2("../input/image2.png");
   Image rotateImg("../input/8k.jpg");

   if (img1.getData() == nullptr || img2.getData() == nullptr || rotateImg.getData() == nullptr) {
      cerr << "Failed to load images. Exiting." << endl;
      return 1;
   }
   auto rotateStart = high_resolution_clock::now();
   rotate90(rotateImg);
   auto rotateEnd = high_resolution_clock::now();
   duration<float, milli> rotateDuration = rotateEnd - rotateStart;
   cout << "Rotation duration: " << rotateDuration.count() << " milliseconds" << endl;

   auto compareStart = high_resolution_clock::now();
   double similarity = compareImages(img1, img2);
   auto compareEnd = high_resolution_clock::now();
   duration<float, milli> compareDuration = compareEnd - compareStart;
   cout << "Compare duration: " << compareDuration.count() << " milliseconds" << endl;
   cout << "Similarity between the two images: " << similarity << "%" << endl;

   return 0;
}
>>>>>>> 981f099a06b88634e0cf05a82a4a2cd9535179cd

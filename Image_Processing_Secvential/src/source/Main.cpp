#include <iostream>
#include <chrono>
#include "Image.h"
#include "ImageUtils.h"


int main()
{
   Image img1("../images/image1.png");
   Image img2("../images/8k.jpg");
   Image rotateImg("../images/8k2.jpg");

   if (img1.getData() == nullptr || img2.getData() == nullptr) {
      std::cerr << "Failed to load images. Exiting.";
      return 1;
   }
   
   auto rotateStart = std::chrono::steady_clock::now();
   rotate90(rotateImg);
   auto rotateEnd = std::chrono::steady_clock::now();
   auto rotateDuration = std::chrono::duration_cast<std::chrono::milliseconds>(rotateEnd - rotateStart);

   // Measure the time taken for comparison
   auto compareStart = std::chrono::steady_clock::now();
   double similarity = compareImages(img1, img2);
   auto compareEnd = std::chrono::steady_clock::now();
   auto compareDuration = std::chrono::duration_cast<std::chrono::milliseconds>(compareEnd - compareStart);

   // Output durations to console
   std::cout << "Rotation duration: " << rotateDuration.count() << " milliseconds" << std::endl;
   std::cout << "Compare duration: " << compareDuration.count() << " milliseconds" << std::endl;
   std::cout << "Similarity between the two images: " << similarity << "%" << std::endl;

   return 0;
}


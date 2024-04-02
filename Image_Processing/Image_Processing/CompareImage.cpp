class CompareImage {
public:
   static double calculateSimilarity(const Image& img1, const Image& img2) {
      // Check if images have the same dimensions
      auto [width1, height1] = img1.getSize();
      auto [width2, height2] = img2.getSize();
      if (width1 != width2 || height1 != height2) {
         std::cerr << "Images must have the same dimensions for comparison." << std::endl;
         return -1.0;
      }

      // Calculate similarity
      double totalDifference = 0.0;
      int numPixels = width1 * height1;

      for (int i = 0; i < height1; ++i) {
         for (int j = 0; j < width1; ++j) {
            RGB pixel1 = img1.getPixel(i, j);
            RGB pixel2 = img2.getPixel(i, j);

            totalDifference += std::abs(pixel1.red - pixel2.red) +
               std::abs(pixel1.green - pixel2.green) +
               std::abs(pixel1.blue - pixel2.blue);
         }
      }

      // Normalize the difference
      double similarity = 1.0 - (totalDifference / (255 * 3 * numPixels));
      return similarity;
   }
};

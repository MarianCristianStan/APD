<<<<<<< HEAD
#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <cstring>

enum ImageType {
   PNG,JPG
};
class Image {


public:
   Image(const char* filename);
   Image(int w, int h, int channels);
   Image(const Image& img);
   ~Image();

   bool read(const char* filename);
   bool write(const char* filename);

   ImageType getFileType(const char* filename);
 

   void setWidth(int w) { width = w; }
   void setHeight(int h) { height = h; }
   void setData(uint8_t* d) { data = d; }
   void setSize(size_t s) { size = s; }
   void setChannels(int c) { channels = c; }
   void setFilename(const char* filename) { strcpy_s(this->filename, sizeof(this->filename), filename); } 

   int getWidth() const { return width; }
   int getHeight() const { return height; }
   uint8_t* getData() const { return data; }
   size_t getSize() const { return size; }
   int getChannels() const { return channels; }
   const char* getFilename() const { return filename; } 
   

private:
   char filename[256];
   int width;
   int height;
   uint8_t* data = NULL;
   size_t size = 0;
   int channels;

};
=======
#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <cstring>

enum ImageType {
   PNG,JPG
};
class Image {


public:
   Image(const char* filename);
   Image(int w, int h, int channels);
   Image(const Image& img);
   ~Image();

   bool read(const char* filename);
   bool write(const char* filename);

   ImageType getFileType(const char* filename);
 

   void setWidth(int w) { width = w; }
   void setHeight(int h) { height = h; }
   void setData(uint8_t* d) { data = d; }
   void setSize(size_t s) { size = s; }
   void setChannels(int c) { channels = c; }
   void setFilename(const char* filename) { strcpy_s(this->filename, sizeof(this->filename), filename); } 

   int getWidth() const { return width; }
   int getHeight() const { return height; }
   uint8_t* getData() const { return data; }
   size_t getSize() const { return size; }
   int getChannels() const { return channels; }
   const char* getFilename() const { return filename; } 
   

private:
   char filename[256];
   int width;
   int height;
   uint8_t* data = NULL;
   size_t size = 0;
   int channels;

};
>>>>>>> 981f099a06b88634e0cf05a82a4a2cd9535179cd

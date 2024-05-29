
#define STB_IMAGE_IMPLEMENTATION
#include "../stb_libs/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_libs/stb_image_write.h"

#include "Image.h"
#include <iostream>

Image::Image(const char* filename) {
	if (read(filename))
	{
		size = width * height * channels;
		setFilename(filename);
		std::cout << "Read " << filename << std::endl;

	}
	else {
		std::cout << "Failed to read " << filename<<std::endl;;
	}
}

Image::Image(int w, int h, int ch ):width(w), height(h), channels(ch) {

	size = width * height * channels;
	data = new uint8_t[size];
}

Image::Image(const Image& img): Image(img.width,img.height,img.channels) {
	memcpy(data, img.data, size);
}

Image::~Image() {
	stbi_image_free(data);
}



bool Image::read(const char* filename) {
	data = stbi_load(filename, &width, &height, &channels, 0);
	return data != NULL;
}

bool Image::write(const char* filename) {
	ImageType imageType = getFileType(filename);
	int succes = 0 ;
	switch (imageType) {
		case PNG:
			succes = stbi_write_png(filename, width, height, channels,data,width*channels);
		case JPG:
			succes = stbi_write_jpg(filename, width, height, channels, data, width * channels);
	}
	return succes != 0;
}

ImageType Image::getFileType(const char* filename) {
	const char* ext = strchr(filename, '.');
	if (ext != nullptr)
	{
		if (strcmp(ext, ".png") == 0)
		{
			return PNG;
		}
		else if (strcmp(ext, ".jpg") == 0)
		{
			return JPG;
		}
	}
	return PNG;
}



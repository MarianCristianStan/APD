
#define STB_IMAGE_IMPLEMENTATION
#include "../stb_libs/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_libs/stb_image_write.h"

#include "Image.h"
#include <iostream>

Image::Image(const char* filename) : data(nullptr), width(0), height(0), channels(0), size(0) {
	if (read(filename)) {
		size = width * height * channels;
		setFilename(filename);
		std::cout << "Read " << filename << std::endl;
	}
	else {
		std::cout << "Failed to read " << filename << std::endl;
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

bool Image::readFromFile(FILE* file) {
	data = stbi_load_from_file(file, &width, &height, &channels, 0);
	if (data == nullptr) {
		std::cerr << "Reason: " << stbi_failure_reason() << std::endl;
		perror("Error details");
	}
	return data != nullptr;
}

bool Image::read(const char* filename) {
	data = stbi_load(filename, &width, &height, &channels, 0);
	if (data == nullptr) {
		std::cerr << "Error: " << stbi_failure_reason() << " while reading file: " << filename << std::endl;
	}
	return data != nullptr;
}

bool Image::write(const char* filename) {
	ImageType imageType = getFileType(filename);
	int succes = 0 ;
	switch (imageType) {
		case PNG:
			succes = stbi_write_png(filename, width, height, channels,data,width*channels);
			break;
		case JPG:
			succes = stbi_write_jpg(filename, width, height, channels, data, width * channels);
			break;
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

void Image::allocateMemory(int width, int height, int channels) {
	this->width = width;
	this->height = height;
	this->channels = channels;
	this->size = width * height * channels;
	this->data = new uint8_t[this->size];
}



#ifndef ETA_IMAGE_CU
#define ETA_IMAGE_CU

#include "Image.cuh"

namespace eta {

int Image::height() const { return this->height_; }
int Image::width() const { return this->width_; }

int Image::size() const { return this->height_ * this->width_; }
int Image::capacity() const { return this->capacity_; }

const unsigned char* Image::data() const { return this->data_; }

unsigned char& Image::get(int h, int w, int c) {
    return this->data_[4 * this->width_ * h + 4 * w + c];
}

const unsigned char& Image::get(int h, int w, int c) const {
    return this->data_[4 * this->width_ * h + 4 * w + c];
}

Image& Image::operator=(const Image& img) {
    this->clear();

    this->height_ = img.height_;
    this->width_ = img.width_;

    this->capacity_ = img.capacity_;

    this->data_ = img.data_;

    return *this;
}

Image::~Image() { ::free(this->data_); }

void Image::create(int height, int width) {
    this->height_ = height;
    this->width_ = width;

    int size{ height * width };

    if (size <= this->capacity_) { return; }

    this->capacity_ = size;

    ::free(this->data_);
    this->data_ =
        static_cast<unsigned char*>(::malloc(sizeof(unsigned char) * 4 * size));
}

void Image::load(const std::string& path) {
    int height;
    int width;
    int channel;

    unsigned char* data{ stbi_load(path.c_str(), &width, &height, &channel,
                                   0) };

    if (data == nullptr) { ETA_throw("load error\n"); }

    if (channel != 1 && channel != 3 && channel != 4) {
        ETA_throw("undefined channel\n");
    }

    if (channel == 4) {
        this->height_ = height;
        this->width_ = width;

        this->capacity_ = height * width;

        ::free(this->data_);
        this->data_ = data;

        return;
    }

    this->create(height, width);

    int a{ channel * width };
    int b{ channel };

    if (channel == 1) {
        for (int h{ 0 }; h < height; ++h) {
            for (int w{ 0 }; w < width; ++w) {
                this->get(h, w, 0) = data[a * h + b * w];
                this->get(h, w, 1) = data[a * h + b * w];
                this->get(h, w, 2) = data[a * h + b * w];
                this->get(h, w, 3) = 255;
            }
        }
    } else {
        for (int h{ 0 }; h < height; ++h) {
            for (int w{ 0 }; w < width; ++w) {
                this->get(h, w, 0) = data[a * h + b * w + 0];
                this->get(h, w, 1) = data[a * h + b * w + 1];
                this->get(h, w, 2) = data[a * h + b * w + 2];
                this->get(h, w, 3) = 255;
            }
        }
    }

    ::free(data);
}

void Image::save(const std::string& path) const {
    stbi_write_png(path.c_str(), this->width_, this->height_, 4, this->data_,
                   0);
}

void Image::clear() {
    this->height_ = 0;
    this->width_ = 0;

    this->capacity_ = 0;

    ::free(this->data_);
    this->data_ = nullptr;
}

} // namespace eta

#endif

#ifndef ETA_CUBEMAP_CU
#define ETA_CUBEMAP_CU

#include "Cubemap.cuh"
#include "Image.cu"

namespace eta {

int Cubemap::size() const { return this->size_; }

Image& Cubemap::get(int index) { return this->data_[index]; }
const Image& Cubemap::get(int index) const { return this->data_[index]; }

unsigned char& Cubemap::get(int d, int h, int w, int c) {
    return this->data_[d].get(h, w, c);
}

const unsigned char& Cubemap::get(int d, int h, int w, int c) const {
    return this->data_[d].get(h, w, c);
}

void Cubemap::create(int size) {
    this->size_ = size;

    for (int d{ 0 }; d < 6; ++d) { this->data_[d].create(size, size); }
}

void Cubemap::load(const std::string& path) {
    Image img;
    img.load(path);

    if ((img.height() % 3 != 0) || (img.width() % 4 != 0) ||
        (img.height() / 3 != img.width() / 4)) {
        ETA_throw("size error\n");
    }

    this->size_ = img.height() / 3;

    constexpr int offset[6][2]{
        { 1, 2 }, // X_POX
        { 1, 0 }, // X_NEG
        { 1, 1 }, // Y_POS
        { 1, 3 }, // Y_NEG
        { 0, 1 }, // Z_POS
        { 2, 1 }, // Z_NEG
    };

    for (int d{ 0 }; d < 6; ++d) {
        this->data_[d].create(this->size_, this->size_);

        for (int h{ 0 }; h < this->size_; ++h) {
            for (int w{ 0 }; w < this->size_; ++w) {
                for (int c{ 0 }; c < 4; ++c) {
                    this->data_[d].get(h, w, c) =
                        img.get(offset[d][0] * this->size_ + h,
                                offset[d][1] * this->size_ + w, c);
                }
            }
        }
    }
}

void Cubemap::save(const std::string& path) const {
    Image img;
    img.create(this->size_ * 3, this->size_ * 4);

    constexpr int offset[6][2]{
        { 1, 2 }, // X_POX
        { 1, 0 }, // X_NEG
        { 1, 1 }, // Y_POS
        { 1, 3 }, // Y_NEG
        { 0, 1 }, // Z_POS
        { 2, 1 }, // Z_NEG
    };

    for (int d{ 0 }; d < 6; ++d) {
        for (int h{ 0 }; h < this->size_; ++h) {
            for (int w{ 0 }; w < this->size_; ++w) {
                for (int c{ 0 }; c < 4; ++c) {
                    img.get(offset[d][0] * this->size_ + h,
                            offset[d][1] * this->size_ + w, c) =
                        this->data_[d].get(h, w, c);
                }
            }
        }
    }

    constexpr int void_offset[6][2]{
        { 0, 0 }, { 0, 2 }, { 0, 3 }, { 2, 0 }, { 2, 2 }, { 2, 3 },
    };

    for (int d{ 0 }; d < 6; ++d) {
        for (int h{ 0 }; h < this->size_; ++h) {
            for (int w{ 0 }; w < this->size_; ++w) {
                for (int c{ 0 }; c < 4; ++c) {
                    img.get(void_offset[d][0] * this->size_ + h,
                            void_offset[d][1] * this->size_ + w, c) = 255;
                }
            }
        }
    }

    img.save(path);
}

void Cubemap::clear() {
    for (int d{ 0 }; d < 6; ++d) { this->data_[d].clear(); }
}

} // namespace eta

#endif

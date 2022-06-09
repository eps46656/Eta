#ifndef ETA_IMAGE_CUH
#define ETA_IMAGE_CUH

#include "define.cuh"

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

namespace eta {

class Image {
public:
    __host__ int height() const;
    __host__ int width() const;

    __host__ int size() const;
    __host__ int capacity() const;

    __host__ const unsigned char* data() const;

    __host__ Image& operator=(const Image& img);

    __host__ ~Image();

    __host__ unsigned char& get(int h, int w, int c);
    __host__ const unsigned char& get(int h, int w, int c) const;

    __host__ void create(int height, int width);

    __host__ void load(const std::string& path);
    __host__ void save(const std::string& path) const;

    __host__ void clear();

private:
    int height_{ 0 };
    int width_{ 0 };

    int capacity_{ 0 };

    unsigned char* data_{ nullptr };
};

} // namespace eta

#endif
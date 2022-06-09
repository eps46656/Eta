#ifndef ETA_CUBEMAP_CUH
#define ETA_CUBEMAP_CUH

#include "Image.cuh"

namespace eta {

class Cubemap {
public:
    __host__ int size() const;

    __host__ Image& operator=(const Image& img);

    __host__ Image& get(int d);
    __host__ const Image& get(int d) const;

    __host__ unsigned char& get(int d, int h, int w, int c);
    __host__ const unsigned char& get(int d, int h, int w, int c) const;

    __host__ void create(int size);

    __host__ void load(const std::string& path);
    __host__ void save(const std::string& path) const;

    __host__ void change_channel(int channel);

    __host__ void clear();

private:
    int size_{ 0 };

    Image data_[6];
};

} // namespace eta

#endif
#ifndef ETA_TEXTURECUBE_CUH
#define ETA_TEXTURECUBE_CUH

#include "Texture2D.cuh"

namespace eta {

#define ETA_X_POS (0)
#define ETA_X_NEG (1)
#define ETA_Y_POS (2)
#define ETA_Y_NEG (3)
#define ETA_Z_POS (4)
#define ETA_Z_NEG (5)

struct TextureCubeVector {
    static constexpr Vector<float, 3> r[6]{
        { 0, 1, 0 }, // x_pos
        { 0, -1, 0 }, // x_neg
        { -1, 0, 0 }, // y_pos
        { 1, 0, 0 }, // y_neg
        { -1, 0, 0 }, // z_pos
        { -1, 0, 0 }, // z_neg
    };

    static constexpr Vector<float, 3> u[6]{
        { 0, 0, 1 }, // x_pos
        { 0, 0, 1 }, // x_neg
        { 0, 0, 1 }, // y_pos
        { 0, 0, 1 }, // y_neg
        { 0, -1, 0 }, // z_pos
        { 0, 1, 0 }, // z_neg
    };

    static constexpr Vector<float, 3> outside_f[6]{
        { 1, 0, 0 }, // x_pos
        { -1, 0, 0 }, // x_neg
        { 0, 1, 0 }, // y_pos
        { 0, -1, 0 }, // y_neg
        { 0, 0, 1 }, // z_pos
        { 0, 0, -1 }, // z_neg
    };

    static constexpr Vector<float, 3> inside_f[6]{
        { -1, 0, 0 }, // x_pos
        { 1, 0, 0 }, // x_neg
        { 0, -1, 0 }, // y_pos
        { 0, 1, 0 }, // y_neg
        { 0, 0, -1 }, // z_pos
        { 0, 0, 1 }, // z_neg
    };
};

constexpr Vector<float, 3> TextureCubeVector::r[];
constexpr Vector<float, 3> TextureCubeVector::u[];
constexpr Vector<float, 3> TextureCubeVector::outside_f[];
constexpr Vector<float, 3> TextureCubeVector::inside_f[];

template<typename T> struct TextureCube {
    Texture2D<T> tex[6];

    __host__ __device__ void get_dst(int* d, float* s, float* t,
                                     Vector<float, 3> coord) const {
        float x{ coord.data[0] };
        float y{ coord.data[1] };
        float z{ coord.data[2] };

        float abs_x{ ::abs(x) };
        float abs_y{ ::abs(y) };
        float abs_z{ ::abs(z) };

        if (abs_y <= abs_x && abs_z <= abs_x) {
            if (0 < x) {
                *d = ETA_X_POS;
                *s = y / abs_x;
                *t = z / abs_x;
            } else {
                *d = ETA_X_NEG;
                *s = -y / abs_x;
                *t = z / abs_x;
            }
        } else if (abs_x <= abs_y && abs_z <= abs_y) {
            if (0 < y) {
                *d = ETA_Y_POS;
                *s = -x / abs_y;
                *t = z / abs_y;
            } else {
                *d = ETA_Y_NEG;
                *s = x / abs_y;
                *t = z / abs_y;
            }
        } else {
            if (0 < z) {
                *d = ETA_Z_POS;
                *s = -x / abs_z;
                *t = -y / abs_z;
            } else {
                *d = ETA_Z_NEG;
                *s = -x / abs_z;
                *t = y / abs_z;
            }
        }
    }

    __host__ __device__ T get(Vector<float, 3> coord) const {
        int d;
        float s;
        float t;

        get_dst(&d, &s, &t, coord);

        return this->tex[d].get({ s * 0.5f + 0.5f, t * 0.5f + 0.5f });
    }

    __host__ __device__ int count_16(Vector<float, 3> coord, T value) {
        int d;
        float s;
        float t;

        get_dst(&d, &s, &t, coord);

        return this->tex[d].count_16({ s * 0.5f + 0.5f, t * 0.5f + 0.5f },
                                     value);
    }
};

}

#endif

#ifndef ETA_TEXTURE2D_CUH
#define ETA_TEXTURE2D_CUH

#include "utils.cuh"
#include "Vector.cuh"
#include "View.cuh"

namespace eta {

enum Texture2DWrappingMode {
    REPEAT,
    MIRRORED_REPEAT,
    CLAMP_TO_EDGE,
    CLAMP_TO_BORDER,
};

enum Texture2DFilteringMode {
    NEAREST,
    LINEAR,
};

template<typename T> struct Texture2D {
    int height;
    int width;
    View data;
    T border_value;

    Texture2DWrappingMode s_wrapping_mode{ REPEAT };
    Texture2DFilteringMode s_filtering_mode{ LINEAR };

    Texture2DWrappingMode t_wrapping_mode{ REPEAT };
    Texture2DFilteringMode t_filtering_mode{ LINEAR };

    __host__ __device__ static float
    wrapping_arrange(float value, Texture2DWrappingMode wrapping_mode) {
        switch (wrapping_mode) {
            case REPEAT: {
                return value - floor(value);
            }

            case MIRRORED_REPEAT: {
                return 1.0f -
                       2.0f * abs(value / 2.0f - floor(value / 2.0f) - 0.5f);
            }

            case CLAMP_TO_EDGE: {
                return clamp<float>(value, 0.0f, 1.0f);
            }

            case CLAMP_TO_BORDER: {
                return value - floor(value);
            }
        }

        return 0;
    }

    __host__ __device__ T get(Vector<float, 2> coord) const {
        float s{ wrapping_arrange(coord.data[0], this->s_wrapping_mode) };
        float t{ wrapping_arrange(coord.data[1], this->t_wrapping_mode) };

        float h{ (1.0f - t) * (this->height - 1.0f) };
        float w{ s * (this->width - 1.0f) };

        int h_0, h_1, w_0, w_1;

        switch (this->t_filtering_mode) {
            case NEAREST: {
                h_0 = h_1 = clamp<int>(::round(h), 0, height - 1);
                break;
            }

            case LINEAR: {
                h_0 = clamp<int>(::floor(h), 0, height - 1);
                h_1 = clamp<int>(::ceil(h), 0, height - 1);
                break;
            }
        }

        switch (this->s_filtering_mode) {
            case NEAREST: {
                w_0 = w_1 = clamp<int>(::round(w), 0, width - 1);
                break;
            }

            case LINEAR: {
                w_0 = clamp<int>(::floor(w), 0, width - 1);
                w_1 = clamp<int>(::ceil(w), 0, width - 1);
                break;
            }
        }

        T v00{ this->data.get<T>(h_0, w_0) };
        T v01{ this->data.get<T>(h_0, w_1) };
        T v10{ this->data.get<T>(h_1, w_0) };
        T v11{ this->data.get<T>(h_1, w_1) };

        return bilinear(v00, v01, v10, v11, h - h_0, w - w_0);
    }
};

} // namespace eta

#endif

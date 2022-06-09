#ifndef ETA_RASTERIZE_CU
#define ETA_RASTERIZE_CU

#include "Rasterize.cuh"

namespace eta {

template<typename MaterialGetter>
__global__ void Rasterize(int height, int width, //

                          float* dst_w, //
                          Material* dst_material, //

                          int* locker, //

                          int num_of_faces, //
                          View face_coord, // Vector<float, 3> [num_of_faces, 3]
                          Matrix<float, 3, 4> transform, //

                          MaterialGetter material_getter //
) {
    __shared__ float tmp[3][1024];

    __shared__ float x[1024];
    __shared__ float y[1024];
    __shared__ float w[3][1024];

    __shared__ float n1_v[2][1024];
    __shared__ float n2_v[2][1024];

    ////////////////////////////////////////////////////////////////////////////

    float x_value{ (width - 1.0f) / 2.0f };
    float y_value{ (height - 1.0f) / 2.0f };

    short min_x;
    short max_x;
    short min_y;
    short max_y;

    for (int face_i{ static_cast<int>(GLOBAL_TID) }; face_i < num_of_faces;
         face_i += gridDim.x * blockDim.x) {
        { // v == 1
            Vector<float, 3> vertex{ face_coord.get<Vector<float, 3>>(face_i,
                                                                      1) };

            w[1][TID] = 1.0f / (transform.data[2][0] * vertex.data[0] + //
                                transform.data[2][1] * vertex.data[1] + //
                                transform.data[2][2] * vertex.data[2] + //
                                transform.data[2][3]);

            x[TID] = (transform.data[0][0] * vertex.data[0] + //
                      transform.data[0][1] * vertex.data[1] + //
                      transform.data[0][2] * vertex.data[2] + //
                      transform.data[0][3]) *
                         w[1][TID] * x_value +
                     x_value;
            y[TID] = (transform.data[1][0] * vertex.data[0] + //
                      transform.data[1][1] * vertex.data[1] + //
                      transform.data[1][2] * vertex.data[2] + //
                      transform.data[1][3]) *
                         w[1][TID] * -y_value +
                     y_value;

            n2_v[0][TID] = -y[TID];
            n2_v[1][TID] = x[TID];

            min_x = __float2int_rd(x[TID]);
            max_x = __float2int_ru(x[TID]);

            min_y = __float2int_rd(y[TID]);
            max_y = __float2int_ru(y[TID]);
        }

        /*
         * n1_v[0][TID] = y[2][TID] - y[0][TID];
         * n1_v[1][TID] = x[0][TID] - x[2][TID];
         * n2_v[0][TID] = y[0][TID] - y[1][TID];
         * n2_v[1][TID] = x[1][TID] - x[0][TID];
         */

        { // v == 2
            /*
            x[TID] = transform.data[0][3];
            y[TID] = transform.data[1][3];
            w[2][TID] = transform.data[2][3];

            tmp[0][TID] = face_coord.get<Vector<float, 3>>(face_i, 2).data[0];

            x[TID] += transform.data[0][0] * tmp[0][TID];
            y[TID] += transform.data[1][0] * tmp[0][TID];
            w[2][TID] += transform.data[2][0] * tmp[0][TID];

            tmp[0][TID] = face_coord.get<Vector<float, 3>>(face_i, 2).data[1];

            x[TID] += transform.data[0][1] * tmp[0][TID];
            y[TID] += transform.data[1][1] * tmp[0][TID];
            w[2][TID] += transform.data[2][1] * tmp[0][TID];

            tmp[0][TID] = face_coord.get<Vector<float, 3>>(face_i, 2).data[2];

            x[TID] += transform.data[0][2] * tmp[0][TID];
            y[TID] += transform.data[1][2] * tmp[0][TID];
            w[2][TID] += transform.data[2][2] * tmp[0][TID];

            w[2][TID] = 1.0f / w[2][TID];
            x[TID] = x[TID] * w[2][TID] * x_value + x_value;
            y[TID] = y[TID] * w[2][TID] * -y_value + y_value;
            */

            Vector<float, 3> vertex{ face_coord.get<Vector<float, 3>>(face_i,
                                                                      2) };

            w[2][TID] = 1.0f / (transform.data[2][0] * vertex.data[0] + //
                                transform.data[2][1] * vertex.data[1] + //
                                transform.data[2][2] * vertex.data[2] + //
                                transform.data[2][3]);

            x[TID] = (transform.data[0][0] * vertex.data[0] + //
                      transform.data[0][1] * vertex.data[1] + //
                      transform.data[0][2] * vertex.data[2] + //
                      transform.data[0][3]) *
                         w[2][TID] * x_value +
                     x_value;
            y[TID] = (transform.data[1][0] * vertex.data[0] + //
                      transform.data[1][1] * vertex.data[1] + //
                      transform.data[1][2] * vertex.data[2] + //
                      transform.data[1][3]) *
                         w[2][TID] * -y_value +
                     y_value;

            n1_v[0][TID] = y[TID];
            n1_v[1][TID] = -x[TID];

            min_x = min(min_x, __float2int_rd(x[TID]));
            max_x = max(max_x, __float2int_ru(x[TID]));

            min_y = min(min_y, __float2int_rd(y[TID]));
            max_y = max(max_y, __float2int_ru(y[TID]));
        }

        /*
         * n1_v[0][TID] = y[2][TID] - y[0][TID];
         * n1_v[1][TID] = x[0][TID] - x[2][TID];
         * n2_v[0][TID] = y[0][TID] - y[1][TID];
         * n2_v[1][TID] = x[1][TID] - x[0][TID];
         */

        { // v == 0
            Vector<float, 3> vertex{ face_coord.get<Vector<float, 3>>(face_i,
                                                                      0) };

            w[0][TID] = 1.0f / (transform.data[2][0] * vertex.data[0] + //
                                transform.data[2][1] * vertex.data[1] + //
                                transform.data[2][2] * vertex.data[2] + //
                                transform.data[2][3]);

            x[TID] = (transform.data[0][0] * vertex.data[0] + //
                      transform.data[0][1] * vertex.data[1] + //
                      transform.data[0][2] * vertex.data[2] + //
                      transform.data[0][3]) *
                         w[0][TID] * x_value +
                     x_value;
            y[TID] = (transform.data[1][0] * vertex.data[0] + //
                      transform.data[1][1] * vertex.data[1] + //
                      transform.data[1][2] * vertex.data[2] + //
                      transform.data[1][3]) *
                         w[0][TID] * -y_value +
                     y_value;

            n1_v[0][TID] -= y[TID];
            n1_v[1][TID] += x[TID];
            n2_v[0][TID] += y[TID];
            n2_v[1][TID] -= x[TID];

            min_x = min(min_x, __float2int_rd(x[TID]));
            max_x = max(max_x, __float2int_ru(x[TID]));

            min_y = min(min_y, __float2int_rd(y[TID]));
            max_y = max(max_y, __float2int_ru(y[TID]));
        }

        /*
         * n1_v[0][TID] = y[2][TID] - y[0][TID];
         * n1_v[1][TID] = x[0][TID] - x[2][TID];
         * n2_v[0][TID] = y[0][TID] - y[1][TID];
         * n2_v[1][TID] = x[1][TID] - x[0][TID];
         */

        min_x = max(min_x, 0);
        max_x = min(max_x, width - 1);
        min_y = max(min_y, 0);
        max_y = min(max_y, height - 1);

        w[1][TID] -= w[0][TID];
        w[2][TID] -= w[0][TID];

        tmp[0][TID] =
            1.0f / (n1_v[0][TID] * n2_v[1][TID] - n1_v[1][TID] * n2_v[0][TID]);

        n1_v[0][TID] *= tmp[0][TID];
        n1_v[1][TID] *= tmp[0][TID];
        n2_v[0][TID] *= tmp[0][TID];
        n2_v[1][TID] *= tmp[0][TID];

        if (dst_material == nullptr) {
            for (short x_i{ min_x }; x_i <= max_x; ++x_i) {
                for (short y_i{ min_y }; y_i <= max_y; ++y_i) {
                    tmp[0][TID] = x_i - x[TID]; // dx
                    tmp[1][TID] = y_i - y[TID]; // dy

                    float n1{ n1_v[0][TID] * tmp[0][TID] +
                              n1_v[1][TID] * tmp[1][TID] }; // n1
                    float n2{ n2_v[0][TID] * tmp[0][TID] +
                              n2_v[1][TID] * tmp[1][TID] }; // n2

                    if (n1 < 0.0f || n2 < 0.0f || 1.0f < n1 + n2) { continue; }

                    tmp[0][TID] =
                        w[0][TID] + w[1][TID] * n1 + w[2][TID] * n2; // w

                    int pixel_i{ y_i * width + x_i };

                    atomicMax(reinterpret_cast<int*>(&dst_w[pixel_i]),
                              reinterpret_cast<int&>(tmp[0][TID]));
                }
            }
        } else {
            for (short x_i{ min_x }; x_i <= max_x; ++x_i) {
                for (short y_i{ min_y }; y_i <= max_y; ++y_i) {
                    tmp[0][TID] = x_i - x[TID]; // dx
                    tmp[1][TID] = y_i - y[TID]; // dy

                    float n1{ n1_v[0][TID] * tmp[0][TID] +
                              n1_v[1][TID] * tmp[1][TID] }; // n1
                    float n2{ n2_v[0][TID] * tmp[0][TID] +
                              n2_v[1][TID] * tmp[1][TID] }; // n2

                    if (n1 < 0.0f || n2 < 0.0f || 1.0f < n1 + n2) { continue; }

                    tmp[0][TID] =
                        w[0][TID] + w[1][TID] * n1 + w[2][TID] * n2; // w

                    int pixel_i{ y_i * width + x_i };

                    if (tmp[0][TID] <= dst_w[pixel_i]) { continue; }

                    Material material{ material_getter(
                        face_i, { 1.0f - n1 - n2, n1, n2 }) };

                    do {
                        if (locker[pixel_i] == 1) { continue; }

                        if (atomicOr(&locker[pixel_i], 1) == 0) {
                            if (dst_w[pixel_i] < tmp[0][TID]) {
                                dst_w[pixel_i] = tmp[0][TID];
                                dst_material[pixel_i] = material;
                            }

                            locker[pixel_i] = 0;

                            break;
                        }
                    } while (dst_w[pixel_i] < tmp[0][TID]);
                }
            }
        }
    }
}

__global__ void RasterizationNormalize(int height, int width, //

                                       float* w,

                                       Vector<float, 3> r, // right
                                       Vector<float, 3> u, // up
                                       Vector<float, 3> f // front
) {
    __shared__ float tmp[2][1024];

    __shared__ Vector<float, 3> ray_direct[1024];

    for (int idx{ static_cast<int>(GLOBAL_TID) }; idx < height * width;
         idx += gridDim.x * blockDim.x) {
        tmp[0][TID] = (idx % width) * 2.0f / (width - 1.0f) - 1.0f;
        tmp[1][TID] = (idx / width) * 2.0f / (1.0f - height) + 1.0f;

#pragma unroll
        for (int i{ 0 }; i < 3; ++i) {
            ray_direct[TID].data[i] =
                r.data[i] * tmp[0][TID] + u.data[i] * tmp[1][TID] - f.data[i];
        }

        w[idx] *= rnorm3df(ray_direct[TID].data[0], //
                           ray_direct[TID].data[1], //
                           ray_direct[TID].data[2] //
        );
    }
}

} // namespace eta

#endif

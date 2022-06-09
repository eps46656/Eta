#ifndef ETA_RAYCAST_CU
#define ETA_RAYCAST_CU

#include "RayCast.cuh"

namespace eta {

template<typename MaterialGetter>
__global__ void RayCast(int num_of_rays, //

                        float* dst_w, //
                        Material* dst_material, //

                        float* eff, //
                        Vector<float, 3>* ray_origin_, //
                        Vector<float, 3>* ray_direct_, //
                        int* locker, //

                        int num_of_faces, //
                        View face_coord, // Vector<float, 3> [num_of_faces, 3]

                        Matrix<float, 4, 4> transform, //

                        MaterialGetter material_getter //
) {
    __shared__ Vector<float, 3> tmp[1024];
    __shared__ Matrix<float, 3, 3> m[1024];

    for (int ray_i{ static_cast<int>(GLOBAL_TID) }; ray_i < num_of_rays;
         ray_i += gridDim.x * blockDim.x) {
        if (eff[ray_i] < 0.0f) { continue; }

        int nearest_face_i{ 0 };
        float nearest_w{ dst_w[ray_i] };
        Vector<float, 3> nearest_n;

        Vector<float, 3> ray_origin{ ray_origin_[ray_i] };
        Vector<float, 3> ray_direct{ ray_direct_[ray_i] };

        for (int face_i{ 0 }; face_i < num_of_faces; ++face_i) {
#pragma unroll
            for (int i{ 0 }; i < 3; ++i) {
                tmp[TID] = face_coord.get<Vector<float, 3>>(face_i, i);

#pragma unroll
                for (int j{ 0 }; j < 3; ++j) {
                    m[TID].data[j][i] =
                        transform.data[j][0] * tmp[TID].data[0] +
                        transform.data[j][1] * tmp[TID].data[1] +
                        transform.data[j][2] * tmp[TID].data[2] +
                        transform.data[j][3];
                }
            }

            m[TID] = m[TID].inv();

            Vector<float, 3> ray_origin_n{ m[TID] * ray_origin };
            Vector<float, 3> ray_direct_n{ m[TID] * ray_direct };

            float w{ (ray_direct_n.data[0] + //
                      ray_direct_n.data[1] + //
                      ray_direct_n.data[2] //
                      ) /
                     (1.0f - //
                      ray_origin_n.data[0] - //
                      ray_origin_n.data[1] - //
                      ray_origin_n.data[2] //
                      ) };

            if (1e2 < w || w <= nearest_w) { continue; }

            Vector<float, 3> n{
                ray_origin_n.data[0] + ray_direct_n.data[0] / w,
                ray_origin_n.data[1] + ray_direct_n.data[1] / w,
                ray_origin_n.data[2] + ray_direct_n.data[2] / w,
            };

            if (-1e-3f <= n.data[0] && -1e-3f <= n.data[1] &&
                -1e-3f <= n.data[2]) {
                nearest_face_i = face_i;
                nearest_w = w;
                nearest_n = n;
            }
        }

        if (dst_w[ray_i] < nearest_w) {
            dst_w[ray_i] = nearest_w;

            if (dst_material != nullptr) {
                dst_material[ray_i] =
                    material_getter(nearest_face_i, nearest_n);
            }
        }
    }
}

template<typename MaterialGetter>
__global__ void __launch_bounds__(512)
    RayCast2(int num_of_rays, //

             float* dst_w, //
             Material* dst_material, //

             float* eff, //
             Vector<float, 3>* ray_origin_, //
             Vector<float, 3>* ray_direct_, //
             int* locker, //

             int num_of_faces, //
             View face_coord, // Vector<float, 3> [num_of_faces, 3]

             Matrix<float, 4, 4> transform, //

             MaterialGetter material_getter //
    ) {
    __shared__ Vector<float, 3> tmp[512];
    __shared__ Matrix<float, 3, 3> m[512];

    for (int face_i{ static_cast<int>(GLOBAL_TID) }; face_i < num_of_faces;
         face_i += gridDim.x * blockDim.x) {
#pragma unroll
        for (int i{ 0 }; i < 3; ++i) {
            tmp[TID] = face_coord.get<Vector<float, 3>>(face_i, i);

#pragma unroll
            for (int j{ 0 }; j < 3; ++j) {
                m[TID].data[j][i] = transform.data[j][0] * tmp[TID].data[0] +
                                    transform.data[j][1] * tmp[TID].data[1] +
                                    transform.data[j][2] * tmp[TID].data[2] +
                                    transform.data[j][3];
            }
        }

        m[TID] = m[TID].inv();

        /* printf("%f %f %f %f %f %f %f %f %f ", m[TID].data[0][0],
               m[TID].data[0][1], m[TID].data[0][2], m[TID].data[1][0],
               m[TID].data[1][1], m[TID].data[1][2], m[TID].data[2][0],
               m[TID].data[2][1], m[TID].data[2][2]); */

        for (int ray_i_{ 0 }; ray_i_ < num_of_rays; ++ray_i_) {
            int ray_i{ (ray_i_ + static_cast<int>(GLOBAL_TID)) % num_of_rays };

            if (eff[ray_i] < 0.0f) { continue; }

            Vector<float, 3> ray_origin_n{ m[TID] * ray_origin_[ray_i] };
            Vector<float, 3> ray_direct_n{ m[TID] * ray_direct_[ray_i] };

            float w{ (ray_direct_n.data[0] + //
                      ray_direct_n.data[1] + //
                      ray_direct_n.data[2] //
                      ) /
                     (1.0f - //
                      ray_origin_n.data[0] - //
                      ray_origin_n.data[1] - //
                      ray_origin_n.data[2] //
                      ) };

            if (1e3 < w && w <= dst_w[ray_i]) { continue; }

            Vector<float, 3> n{
                ray_origin_n.data[0] + ray_direct_n.data[0] / w,
                ray_origin_n.data[1] + ray_direct_n.data[1] / w,
                ray_origin_n.data[2] + ray_direct_n.data[2] / w,
            };

            if (n.data[0] < 0.0f || n.data[1] < 0.0f || n.data[2] < 0.0f) {
                continue;
            }

            Material material{ material_getter(face_i, n) };

            do {
                if (locker[ray_i] == 1) { continue; }

                if (atomicOr(&locker[ray_i], 1) == 0) {
                    if (dst_w[ray_i] < w) {
                        dst_w[ray_i] = w;
                        dst_material[ray_i] = material;
                    }

                    locker[ray_i] = 0;

                    break;
                }
            } while (dst_w[ray_i] < w);
        }
    }
}

}

#endif
#ifndef ETA_POINTLIGHT_CU
#define ETA_POINTLIGHT_CU

#include "PointLight.cuh"
#include "TextureCube.cuh"
#include "Buffer.cuh"
#include "Rasterize.cu"
#include "Shader.cu"

namespace eta {

int PointLight::cage_resolution() const { return this->cage_resolution_; }

void PointLight::InitCage(int resolution, cudaStream_t stream) {
    this->cage_resolution_ = resolution;

    int resolution_sq{ resolution * resolution };

    for (int d{ 0 }; d < 6; ++d) {
        this->cage_w_[d].Resize(sizeof(float) * resolution_sq);

        this->cage_.tex[d].height = this->cage_.tex[d].width = resolution;
        this->cage_.tex[d].data.base = this->cage_w_[d].data();
        this->cage_.tex[d].data.coeff[0] = sizeof(float) * resolution;
        this->cage_.tex[d].data.coeff[1] = sizeof(float);

        setvalue<float><<<64, 1024, 0, stream>>>(this->cage_w_[d].data(),
                                                 resolution_sq, 0.0f);
    }
}

void PointLight::GenerateCage(const Model* model, cudaStream_t stream) {
    if (this->cage_resolution_ == 0) { ETA_throw("uninitialized\n"); }

    for (int d{ 0 }; d < 6; ++d) {
        model->Rasterize(this->cage_resolution_,
                         this->cage_resolution_, // height, width

                         this->cage_w_[d].data<float>(), // dst_w
                         nullptr, // dst_material

                         nullptr, // locker

                         perspective_mat(this->origin, // origin
                                         TextureCubeVector::r[d], // right
                                         TextureCubeVector::u[d], // up
                                         TextureCubeVector::inside_f[d] // front
                                         ), // transform

                         stream // stream
        );
    }
}

void PointLight::LoadOn(cudaStream_t stream) {
    for (int d{ 0 }; d < 6; ++d) {
        RasterizationNormalize<<<64, 1024, 0, stream>>>(
            this->cage_resolution_,
            this->cage_resolution_, // height, width

            this->cage_w_[d].data<float>(), // w

            TextureCubeVector::r[d], // right
            TextureCubeVector::u[d], // up
            TextureCubeVector::inside_f[d] // front
        );
    }
}

__global__ void PointLight__Shade_(int size,

                                   Vector<int, 3>* pixel, //

                                   TracingBatch tracing_batch, //

                                   Vector<float, 3> light_point, //
                                   Vector<float, 3> light_intensity, //
                                   TextureCube<float> cage //
) {
    __shared__ float tmp[2][1024];
    __shared__ Vector<float, 3> eye[1024];
    __shared__ Vector<float, 3> light[1024];
    __shared__ Vector<float, 3> intensity[1024];

    for (int idx{ static_cast<int>(GLOBAL_TID) }; idx < size;
         idx += gridDim.x * blockDim.x) {
        if (tracing_batch.w[idx] <= 0.0f) { continue; }
        intensity[TID] = tracing_batch.previous_acc_decay[idx];

#pragma unroll
        for (int i{ 0 }; i < 3; ++i) {
            intensity[TID].data[i] *= light_intensity.data[i];

            eye[TID].data[i] =
                tracing_batch.ray_direct[idx].data[i] / tracing_batch.w[idx];

            light[TID].data[i] = tracing_batch.ray_origin[idx].data[i] +
                                 eye[TID].data[i] - light_point.data[i];
        }

        tmp[0][TID] =
            norm3df(eye[TID].data[0], eye[TID].data[1], eye[TID].data[2]);
        eye[TID].data[0] /= tmp[0][TID];
        eye[TID].data[1] /= tmp[0][TID];
        eye[TID].data[2] /= tmp[0][TID];

        tmp[1][TID] =
            norm3df(light[TID].data[0], light[TID].data[1], light[TID].data[2]);
        light[TID].data[0] /= tmp[1][TID];
        light[TID].data[1] /= tmp[1][TID];
        light[TID].data[2] /= tmp[1][TID];

        /* if ((1.0f / (cage.get(light[TID]) + 1e-3f) - tmp[1][TID]) < -8e-2) {
            continue;
        } */

        tmp[0][TID] +=
            tmp[1][TID] + tracing_batch.previous_acc_path_length[idx];

        tmp[0][TID] = 1.0f / (tmp[0][TID] * tmp[0][TID]);

        Vector<float, 3> bsdf{ Shader::BSDF(eye[TID], light[TID],
                                            tracing_batch.material[idx]) };

        intensity[TID].data[0] *= bsdf.data[0] * tmp[0][TID];
        intensity[TID].data[1] *= bsdf.data[1] * tmp[0][TID];
        intensity[TID].data[2] *= bsdf.data[2] * tmp[0][TID];

        /*
         * intensity =
         *         previous_acc_decay * bsdf * light_intensity
         * ------------------------------------------------------------
         *     (previous_acc_path_length + eye.norm + light.norm)^2
        */

#pragma unroll
        for (int i{ 0 }; i < 3; ++i) {
            atomicAdd(
                &pixel[idx].data[i],
                __float2int_rn(ETA_MUL * ::min(intensity[TID].data[i], 1.0f)));
        }
    }
}

__host__ void PointLight::Shade(int size, Vector<int, 3>* pixel,
                                TracingBatch tracing_batch,
                                cudaStream_t stream) const {
    PointLight__Shade_<<<64, 1024, 0, stream>>>(size, //

                                                pixel, //

                                                tracing_batch, //

                                                this->origin, //
                                                this->intensity, //
                                                this->cage_ //
    );
}

} // namespace eta

#endif

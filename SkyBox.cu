#ifndef ETA_SKYBOX_CU
#define ETA_SKYBOX_CU

#include "Cubemap.cu"
#include "SkyBox.cuh"
#include "TracingBatch.cu"

namespace eta {
TextureCube<Vector<float, 3>>& SkyBox::tex_cube_cpu() {
    return this->tex_cube_cpu_;
}
const TextureCube<Vector<float, 3>>& SkyBox::tex_cube_cpu() const {
    return this->tex_cube_cpu_;
}

TextureCube<Vector<float, 3>>& SkyBox::tex_cube_gpu() {
    return this->tex_cube_gpu_;
}
const TextureCube<Vector<float, 3>>& SkyBox::tex_cube_gpu() const {
    return this->tex_cube_gpu_;
}

SkyBox::SkyBox(): size_{ 0 } {}

void SkyBox::Load(const Cubemap& cubemap) {
    this->size_ = cubemap.size();

    int size{ static_cast<int>(sizeof(Vector<float, 3>)) * this->size_ *
              this->size_ };

    for (int d{ 0 }; d < 6; ++d) {
        this->tex_cube_cpu_.tex[d].height = this->size_;
        this->tex_cube_cpu_.tex[d].width = this->size_;

        this->tex_cube_cpu_.tex[d].data.base =
            this->tex_cube_buffer_cpu_[d].Resize(size);

        this->tex_cube_cpu_.tex[d].data.coeff[0] =
            sizeof(Vector<float, 3>) * this->size_;
        this->tex_cube_cpu_.tex[d].data.coeff[1] = sizeof(Vector<float, 3>);

        for (int h{ 0 }; h < this->size_; ++h) {
            for (int w{ 0 }; w < this->size_; ++w) {
                this->tex_cube_cpu_.tex[d].data.get<Vector<float, 3>>(h, w) = {
                    cubemap.get(d, h, this->size_ - 1 - w, 0) / 255.0f,
                    cubemap.get(d, h, this->size_ - 1 - w, 1) / 255.0f,
                    cubemap.get(d, h, this->size_ - 1 - w, 2) / 255.0f,
                };
            }
        }
    }
}

void SkyBox::LoadOn(cudaStream_t stream) {
    int size{ static_cast<int>(sizeof(Vector<float, 3>)) * this->size_ *
              this->size_ };

    for (int d{ 0 }; d < 6; ++d) {
        this->tex_cube_gpu_.tex[d].height = this->size_;
        this->tex_cube_gpu_.tex[d].width = this->size_;

        this->tex_cube_gpu_.tex[d].data.base =
            this->tex_cube_buffer_gpu_[d].Resize(size);

        this->tex_cube_gpu_.tex[d].data.coeff[0] =
            sizeof(Vector<float, 3>) * this->size_;
        this->tex_cube_gpu_.tex[d].data.coeff[1] = sizeof(Vector<float, 3>);

        this->tex_cube_buffer_cpu_[d].PassTo(this->tex_cube_buffer_gpu_[d],
                                             size, stream);
    }
}

__global__ void SkyBox__Shade_(int size, Vector<int, 3>* pixel,
                               TracingBatch tracing_batch,
                               TextureCube<Vector<float, 3>> tex_cube) {
    __shared__ Vector<float, 3> previous_acc_decay[1024];
    __shared__ Vector<float, 3> color[1024];

    for (int idx{ static_cast<int>(GLOBAL_TID) }; idx < size;
         idx += gridDim.x * blockDim.x) {
        if (tracing_batch.previous_acc_path_length[idx] < 0.0f ||
            0.0f < tracing_batch.w[idx]) {
            continue;
        }

        previous_acc_decay[TID] = tracing_batch.previous_acc_decay[idx];
        color[TID] = tex_cube.get(tracing_batch.ray_direct[idx]);

        atomicAdd(&pixel[idx].data[0],
                  __float2int_rn(ETA_MUL * previous_acc_decay[TID].data[0] *
                                 color[TID].data[0]));
        atomicAdd(&pixel[idx].data[1],
                  __float2int_rn(ETA_MUL * previous_acc_decay[TID].data[1] *
                                 color[TID].data[1]));
        atomicAdd(&pixel[idx].data[2],
                  __float2int_rn(ETA_MUL * previous_acc_decay[TID].data[2] *
                                 color[TID].data[2]));
    }
}

void SkyBox::Shade(int size, Vector<int, 3>* pixel, TracingBatch tracing_batch,
                   cudaStream_t stream) const {
    SkyBox__Shade_<<<4, 1024, 0, stream>>>(size, pixel, tracing_batch,
                                           this->tex_cube_gpu_);
}

} // namespace eta

#endif

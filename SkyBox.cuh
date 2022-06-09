#ifndef ETA_SKYBOX_CUH
#define ETA_SKYBOX_CUH

#include "Buffer.cuh"
#include "Cubemap.cuh"
#include "Light.cuh"
#include "TextureCube.cuh"

namespace eta {

class SkyBox: public Light {
public:
    __host__ TextureCube<Vector<float, 3>>& tex_cube_cpu();
    __host__ const TextureCube<Vector<float, 3>>& tex_cube_cpu() const;

    __host__ TextureCube<Vector<float, 3>>& tex_cube_gpu();
    __host__ const TextureCube<Vector<float, 3>>& tex_cube_gpu() const;

    __host__ SkyBox();

    __host__ void Load(const Cubemap& cube_map);

    __host__ void LoadOn(cudaStream_t stream);

    __host__ void Shade(int size, Vector<int, 3>* pixel,
                        TracingBatch tracing_batch,
                        cudaStream_t stream) const override;

private:
    int size_;

    Buffer<CPU> tex_cube_buffer_cpu_[6];
    Buffer<GPU> tex_cube_buffer_gpu_[6];

    TextureCube<Vector<float, 3>> tex_cube_cpu_;
    TextureCube<Vector<float, 3>> tex_cube_gpu_;
};

} // namespace eta

#endif

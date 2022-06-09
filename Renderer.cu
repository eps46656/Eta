#ifndef ETA_RENDERER_CU
#define ETA_RENDERER_CU

#include "Renderer.cuh"

namespace eta {

__global__ void GenerateRay(int height, int width, //

                            TracingBatch dst_tracing_batch, //

                            Vector<float, 3> view_point, //
                            Vector<float, 3> r, //
                            Vector<float, 3> u, //
                            Vector<float, 3> f //
) {
    for (int idx{ static_cast<int>(GLOBAL_TID) }; idx < height * width;
         idx += gridDim.x * blockDim.x) {
        float a{ (idx % width) * 2.0f / (width - 1.0f) - 1.0f };
        float b{ (idx / width) * 2.0f / (1.0f - height) + 1.0f };

        dst_tracing_batch.previous_acc_path_length[idx] = 0.0f;
        dst_tracing_batch.previous_acc_decay[idx] = { 1.0f, 1.0f, 1.0f };

        dst_tracing_batch.ray_origin[idx] = view_point;

        Vector<float, 3> ray_direct;

#pragma unroll
        for (int i{ 0 }; i < 3; ++i) {
            ray_direct.data[i] = r.data[i] * a + u.data[i] * b - f.data[i];
        }

        dst_tracing_batch.ray_direct[idx] =
            ray_direct * rnorm3df(ray_direct.data[0], ray_direct.data[1],
                                  ray_direct.data[2]);

        dst_tracing_batch.w[idx] = 0.0f;
    }
}

Renderer::Data Renderer::AcquireData_(int size) {
    while (!this->spare_data_.empty()) {
        Data data{ this->spare_data_.back() };
        this->spare_data_.pop_back();

        if (size <= data.capacity) { return data; }

        DestroyTracingBatch(data.tracing_batch);
    }

    Data data;
    data.capacity = size;
    data.tracing_batch = CreateTracingBatch(size);

    return data;
}

void Renderer::ReleaseData_(Data& data) { this->spare_data_.push_back(data); }

void Renderer::Reserve() {
    int size{ this->height * this->width };

    for (size_t i{ 0 }; i < this->spare_data_.size(); ++i) { //
        Data data{ this->spare_data_[i] };

        if (data.capacity < size) {
            DestroyTracingBatch(data.tracing_batch);
            this->spare_data_[i] = this->spare_data_.back();
        }
    }

    int need_num_of_data{ 1 + ETA_BRANCH_NUMBER * this->max_depth };

    while (static_cast<int>(this->spare_data_.size()) < need_num_of_data) {
        this->spare_data_.emplace_back();
        this->spare_data_.back().capacity = size;
        this->spare_data_.back().tracing_batch = CreateTracingBatch(size);
    }
}

std::vector<Vector<int, 3>> Renderer::Render(cudaStream_t stream) {
    int size{ this->height * this->width };

    Matrix<float, 3, 4> camera_mat{ perspective_mat(view_point, // view_point
                                                    r, // right
                                                    u, // up
                                                    f // front
                                                    ) };

    std::vector<Vector<int, 3>> result_cpu;
    result_cpu.resize(size);

    Vector<int, 3>* result_gpu{ Malloc<GPU, Vector<int, 3>>(
        sizeof(Vector<int, 3>) * size) };
    setvalue<Vector<int, 3>>
        <<<64, 1024, 0, stream>>>(result_gpu, size, { 0, 0, 0 });

    std::vector<Data> stack;

    if (false) {
        Data ras{ this->AcquireData_(size) };
        ras.depth = 0;

        InitializeTracingBatch(size, ras.tracing_batch, stream);

        for (size_t model_i{ 0 }; model_i < this->models.size(); ++model_i) {
            this->models[model_i]->Rasterize(
                height, width, //

                ras.tracing_batch.w, // dst_w
                ras.tracing_batch.material, // dst_material

                ras.tracing_batch.locker, // locker

                camera_mat, // camera_mat

                stream // stream
            );
        }

        RasterizationToTracingBatch<<<64, 1024, 0, stream>>>(
            this->height, this->width, //

            ras.tracing_batch, // dst

            this->view_point, // view_point
            this->r, // right
            this->u, // up
            this->f // front
        );

        stack.push_back(ras);
    } else {
        Data ras{ this->AcquireData_(size) };
        ras.depth = 0;

        InitializeTracingBatch(size, ras.tracing_batch, stream);

        GenerateRay<<<64, 1024, 0, stream>>>(this->height, this->width, //

                                             ras.tracing_batch, //

                                             this->view_point, // view_point
                                             this->r, // right
                                             this->u, // up
                                             this->f // front
        );

        for (size_t model_i{ 0 }; model_i < this->models.size(); ++model_i) {
            this->models[model_i]->RayCast(
                size, //

                ras.tracing_batch.w, //
                ras.tracing_batch.material, //

                ras.tracing_batch.previous_acc_path_length, //
                ras.tracing_batch.ray_origin, //
                ras.tracing_batch.ray_direct, //
                ras.tracing_batch.locker, //

                stream //
            );
        }

        stack.push_back(ras);
    }

    while (!stack.empty()) {
        Data data{ stack.back() };
        stack.pop_back();

        if (0 < data.depth) {
            for (size_t model_i{ 0 }; model_i < this->models.size();
                 ++model_i) {
                this->models[model_i]->RayCast(
                    size, //

                    data.tracing_batch.w, //
                    data.tracing_batch.material, //

                    data.tracing_batch.previous_acc_path_length, //
                    data.tracing_batch.ray_origin, //
                    data.tracing_batch.ray_direct, //
                    data.tracing_batch.locker, //

                    stream //
                );
            }
        }

        for (size_t light_i{ 0 }; light_i < this->lights.size(); ++light_i) {
            this->lights[light_i]->Shade(size, result_gpu, data.tracing_batch,
                                         stream);
        }

        if (data.depth < max_depth) {
            Array<TracingBatch, ETA_BRANCH_NUMBER> next_tracing_batch;

            for (int b{ 0 }; b < ETA_BRANCH_NUMBER; ++b) {
                Data next_data{ this->AcquireData_(size) };
                next_data.depth = data.depth + 1;
                next_tracing_batch.data[b] = next_data.tracing_batch;
                stack.push_back(next_data);
            }

            TracingBatchBraching<<<64, 1024, 0, stream>>>(
                size, next_tracing_batch, data.tracing_batch);
        }

        this->ReleaseData_(data);
    }

    MemcpyAsync<CPU, GPU>(result_cpu.data(), result_gpu,
                          sizeof(Vector<int, 3>) * size, stream);

    return result_cpu;
}

} // namespace eta

#endif

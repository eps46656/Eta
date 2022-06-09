#ifndef ETA_TRACINGBATCH_CU
#define ETA_TRACINGBATCH_CU

#include "TracingBatch.cuh"
#include "Shader.cu"

namespace eta {

__host__ TracingBatch CreateTracingBatch(int capacity) {
    TracingBatch r;

    r.previous_acc_path_length = Malloc<GPU, float>(sizeof(float) * capacity);
    r.previous_acc_decay =
        Malloc<GPU, Vector<float, 3>>(sizeof(Vector<float, 3>) * capacity);

    r.ray_origin =
        Malloc<GPU, Vector<float, 3>>(sizeof(Vector<float, 3>) * capacity);
    r.ray_direct =
        Malloc<GPU, Vector<float, 3>>(sizeof(Vector<float, 3>) * capacity);

    r.w = Malloc<GPU, float>(sizeof(float) * capacity);
    r.material = Malloc<GPU, Material>(sizeof(Material) * capacity);

    r.locker = Malloc<GPU, int>(sizeof(int) * capacity);

    return r;
}

__host__ void InitializeTracingBatch(int size, TracingBatch& tracing_batch,
                                     cudaStream_t stream) {
    setvalue<int><<<64, 1024, 0, stream>>>(tracing_batch.locker, size, 0);
    setvalue<int><<<64, 1024, 0, stream>>>(tracing_batch.w, size, 0.0f);
}

__host__ void DestroyTracingBatch(TracingBatch& tracing_batch) {
    Free<GPU>(tracing_batch.previous_acc_path_length);
    Free<GPU>(tracing_batch.previous_acc_decay);

    Free<GPU>(tracing_batch.ray_origin);
    Free<GPU>(tracing_batch.ray_direct);

    Free<GPU>(tracing_batch.w);
    Free<GPU>(tracing_batch.material);

    Free<GPU>(tracing_batch.locker);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

__global__ void
RasterizationToTracingBatch(int height, int width, //

                            TracingBatch tracing_batch,

                            Vector<float, 3> view_point, // view point
                            Vector<float, 3> r, // right
                            Vector<float, 3> u, // up
                            Vector<float, 3> f // front
) {
    __shared__ float tmp[3][1024];
    __shared__ Vector<float, 3> ray_direct[1024];

    for (int idx{ static_cast<int>(GLOBAL_TID) }; idx < height * width;
         idx += gridDim.x * blockDim.x) {
        tracing_batch.previous_acc_path_length[idx] = 0.0f;
        tracing_batch.previous_acc_decay[idx] = { 1.0f, 1.0f, 1.0f };

        tmp[0][TID] = (idx % width) * 2.0f / (width - 1.0f) - 1.0f;
        tmp[1][TID] = (idx / width) * 2.0f / (1.0f - height) + 1.0f;

        tracing_batch.ray_origin[idx] = view_point;

#pragma unroll
        for (int i{ 0 }; i < 3; ++i) {
            ray_direct[TID].data[i] =
                r.data[i] * tmp[0][TID] + u.data[i] * tmp[1][TID] - f.data[i];
        }

        tmp[2][TID] = rnorm3df(ray_direct[TID].data[0], ray_direct[TID].data[1],
                               ray_direct[TID].data[2]);

        tracing_batch.ray_direct[idx] = {
            ray_direct[TID].data[0] * tmp[2][TID],
            ray_direct[TID].data[1] * tmp[2][TID],
            ray_direct[TID].data[2] * tmp[2][TID],
        };

        tracing_batch.w[idx] *= tmp[2][TID];
    }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

__global__ void TracingBatchBraching(int size,

                                     Array<TracingBatch, ETA_BRANCH_NUMBER> dst,

                                     TracingBatch tracing_batch) {
    __shared__ Vector<float, 3> ray_origin[1024];
    __shared__ Vector<float, 3> ray_direct[1024];
    __shared__ float w[1024];

    for (int idx{ static_cast<int>(GLOBAL_TID) }; idx < size;
         idx += gridDim.x * blockDim.x) {
#pragma unroll
        for (int b{ 0 }; b < ETA_BRANCH_NUMBER; ++b) {
            dst.data[b].locker[idx] = 0;
        }

        if (tracing_batch.w[idx] <= 0.0f) {
#pragma unroll
            for (int b{ 0 }; b < ETA_BRANCH_NUMBER; ++b) {
                dst.data[b].previous_acc_path_length[idx] = -1.0f;
                dst.data[b].w[idx] = 0.0f;
            }

            continue;
        }

        ray_origin[TID] = tracing_batch.ray_origin[idx];
        ray_direct[TID] = tracing_batch.ray_direct[idx];
        w[TID] = tracing_batch.w[idx];

        Vector<float, 3> x_point;

#pragma unroll
        for (int i{ 0 }; i < 3; ++i) {
            x_point.data[i] =
                ray_origin[TID].data[i] + ray_direct[TID].data[i] / w[TID];
        }

        Vector<float, 3> eye{ tracing_batch.ray_direct[idx] };

        BranchResult branch_result{ Shader::Branch(
            eye, tracing_batch.material[idx]) };

        for (int b{ 0 }; b < ETA_BRANCH_NUMBER; ++b) {
            dst.data[b].previous_acc_path_length[idx] =
                tracing_batch.previous_acc_path_length[idx] + (1.0f / w[TID]);

#pragma unroll
            for (int i{ 0 }; i < 3; ++i) {
                dst.data[b].previous_acc_decay[idx].data[i] =
                    tracing_batch.previous_acc_decay[idx].data[i] *
                    branch_result.data[b].decay.data[i];
            }

            dst.data[b].ray_origin[idx] = x_point;
            dst.data[b].ray_direct[idx] = branch_result.data[b].ray_direct;

            dst.data[b].w[idx] = 0;
        }
    }
}

} // namespace eta

#endif

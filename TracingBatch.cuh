#ifndef ETA_TRACINGBATCH_CUH
#define ETA_TRACINGBATCH_CUH

#include "Material.cuh"
#include "Shader.cuh"

namespace eta {

struct TracingBatch {
    float* previous_acc_path_length;
    Vector<float, 3>* previous_acc_decay;

    Vector<float, 3>* ray_origin;
    Vector<float, 3>* ray_direct;

    float* w;
    Material* material;

    int* locker;
};

__host__ TracingBatch CreateTracingBatch(int capacity_of_rays);
__host__ void InitializeTracingBatch(int size, TracingBatch& tracing_batch,
                                     cudaStream_t stream);
__host__ void DestroyTracingBatch(TracingBatch& tracing_batch);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

__global__ void
RasterizationToTracingBatch(int height, int width, //

                            TracingBatch tracing_batch_,

                            Vector<float, 3> view_point_, // view point
                            Vector<float, 3> r_, // right
                            Vector<float, 3> u_, // up
                            Vector<float, 3> f_ // front
);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

__global__ void TracingBatchBraching(int size,

                                     Array<TracingBatch, ETA_BRANCH_NUMBER> dst,

                                     TracingBatch tracing_batch);

} // namespace eta

#endif

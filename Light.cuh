#ifndef ETA_LIGHT_CUH
#define ETA_LIGHT_CUH

#include "TracingBatch.cuh"

namespace eta {

class Light {
public:
    virtual ~Light() = default;

    __host__ virtual void Shade(int size, Vector<int, 3>* pixel,
                                TracingBatch tracing_batch,
                                cudaStream_t stream) const = 0;
};

} // namespace eta

#endif

#ifndef ETA_RENDERER_CUH
#define ETA_RENDERER_CUH

#include <vector>

#include "Light.cuh"
#include "Model.cuh"

namespace eta {

class Renderer {
public:
    struct Data {
        int capacity;
        int depth;
        int phase;

        TracingBatch tracing_batch;
    };

    int height;
    int width;

    Vector<float, 3> view_point;
    Vector<float, 3> r;
    Vector<float, 3> u;
    Vector<float, 3> f;

    int max_depth;

    std::vector<Model*> models;
    std::vector<Light*> lights;

    __host__ void Reserve();
    __host__ std::vector<Vector<int, 3>> Render(cudaStream_t stream);

private:
    std::vector<Data> spare_data_;

    __host__ Data AcquireData_(int size);
    __host__ void ReleaseData_(Data& data);
};

}

#endif

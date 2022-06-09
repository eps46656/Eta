#ifndef ETA_CAGE_CUH
#define ETA_CAGE_CUH

#include "TextureCube.cuh"
#include "Mem.cuh"
#include "VectorView.cuh"

namespace eta {

class Cage {
public:
    Vector<float, 3>& origin();
    const Vector<float, 3>& origin() const;

    int resolution() const;

    int Init(int resolution);

    void Generate(const Model* model, cudaStream_t stream);

private:
    Vector<float, 3> origin_;
    int resolution_;

    Mem<GPU> tex_mem_[6];
    TextureCube tex_;

    Mem<GPu> w_mem_[6];
    VectorView w_;
};

} // namespace eta

#endif
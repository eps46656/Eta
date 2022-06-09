#ifndef ETA_SHADER_CUH
#define ETA_SHADER_CUH

#include "utils.cuh"

namespace eta {

struct BranchResult_ {
    Vector<float, 3> decay;
    Vector<float, 3> ray_direct;
};

using BranchResult = Array<BranchResult_, ETA_BRANCH_NUMBER>;

struct Shader {
    __device__ static Vector<float, 3> BSDF(Vector<float, 3> eye,
                                            Vector<float, 3> light,
                                            const Material& material);

    __device__ static Vector<float, 3> Reflectance(Vector<float, 3> eye,
                                                   const Material& material);

    __device__ static Vector<float, 3> Transmittance(Vector<float, 3> eye,
                                                     const Material& material);

    /*
     * [decay_r, decay_g, decay_b, ray_direct_x, ray_direct_y, ray_direct_z]
     * 
    */
    __device__ static BranchResult Branch(Vector<float, 3>& eye,
                                          const Material& material);
};

struct ASimpleExampleShader {
    __device__ static Vector<float, 3> BSDF(Vector<float, 3> eye,
                                            Vector<float, 3> light,
                                            const Material& material);

    __device__ static Vector<float, 3> Reflectance(Vector<float, 3> eye,
                                                   const Material& material);

    __device__ static Vector<float, 3> Transmittance(Vector<float, 3> eye,
                                                     const Material& material);

    __device__ static BranchResult Branch(Vector<float, 3>& eye,
                                          const Material& material);
};

} // namespace eta

#endif

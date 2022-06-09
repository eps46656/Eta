#ifndef ETA_FRAGMENT_CUH
#define ETA_FRAGMENT_CUH

#include "Vector.cuh"

namespace eta {

/*
 * This part is user modified.
 * Users can design whatever material they want,
 * and write corresponding Refectance function and Transmittance function.

 * The following is a simple implementation. 
*/

struct Material {
    Vector<float, 3> normal;
    Vector<float, 3> diffuse_color;
    Vector<float, 3> specular_color;
    Vector<float, 3> shininess;
    Vector<float, 3> transparent;
};

__device__ Vector<float, 3> BSDF(Vector<float, 3> eye, Vector<float, 3> light,
                                 const Material& material);

__device__ Vector<float, 3> Reflectance(Vector<float, 3> eye,
                                        const Material& material);

__device__ Vector<float, 3> Transmittance(Vector<float, 3> eye,
                                          const Material& material);

} // namespace eta

#endif

#ifndef ETA_SHADER_CU
#define ETA_SHADER_CU

#include "Shader.cuh"

namespace eta {

__device__ Vector<float, 3> Shader::BSDF(Vector<float, 3> eye,
                                         Vector<float, 3> light,
                                         const Material& material) {
    return ASimpleExampleShader::BSDF(eye, light, material);
}

__device__ BranchResult Shader::Branch(Vector<float, 3>& eye,
                                       const Material& material) {
    return ASimpleExampleShader::Branch(eye, material);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

__device__ Vector<float, 3>
ASimpleExampleShader::BSDF(Vector<float, 3> eye, Vector<float, 3> light,
                           const Material& material) {
    float eye_rnorm{ rnorm3df(eye.data[0], eye.data[1], eye.data[2]) };
    eye.data[0] *= eye_rnorm;
    eye.data[1] *= eye_rnorm;
    eye.data[2] *= eye_rnorm;

    float light_rnorm{ rnorm3df(light.data[0], light.data[1], light.data[2]) };
    light.data[0] *= light_rnorm;
    light.data[1] *= light_rnorm;
    light.data[2] *= light_rnorm;

    float en{ eye.data[0] * material.normal.data[0] + //
              eye.data[1] * material.normal.data[1] + //
              eye.data[2] * material.normal.data[2] };

    float ln{ light.data[0] * material.normal.data[0] + //
              light.data[1] * material.normal.data[1] + //
              light.data[2] * material.normal.data[2] };

    float phase{ ((en < 0) == (ln < 0)) ? 1.0f : -1.0f };

    Vector<float, 3> half{
        eye.data[0] + light.data[0] * phase,
        eye.data[1] + light.data[1] * phase,
        eye.data[2] + light.data[2] * phase,
    };

    Vector<float, 3> transparent{
        (0.5f - material.transparent.data[0]) * phase + 0.5f,
        (0.5f - material.transparent.data[1]) * phase + 0.5f,
        (0.5f - material.transparent.data[2]) * phase + 0.5f,
    };

    float half_rnorm{ rnorm3df(half.data[0], half.data[1], half.data[2]) };
    half.data[0] *= half_rnorm;
    half.data[1] *= half_rnorm;
    half.data[2] *= half_rnorm;

    float hn{ abs(half.data[0] * material.normal.data[0] + //
                  half.data[1] * material.normal.data[1] + //
                  half.data[2] * material.normal.data[2]) };

    Vector<float, 3> s{
        pow(hn, material.shininess.data[0]), //
        pow(hn, material.shininess.data[1]), //
        pow(hn, material.shininess.data[2]), //
    };

    return {
        (material.diffuse_color.data[0] +
         material.specular_color.data[0] * s.data[0]) *
            transparent.data[0] * abs(ln),
        (material.diffuse_color.data[1] +
         material.specular_color.data[1] * s.data[1]) *
            transparent.data[1] * abs(ln),
        (material.diffuse_color.data[2] +
         material.specular_color.data[2] * s.data[2]) *
            transparent.data[2] * abs(ln),
    };
}

__device__ BranchResult ASimpleExampleShader::Branch(Vector<float, 3>& eye,
                                                     const Material& material) {
    BranchResult r;

    { // reflect
        constexpr int b{ 0 };

        float reflectance_value{ pow(1.0f - abs(dot(eye, material.normal)),
                                     0.5f) };

        r.data[b].decay = { reflectance_value, reflectance_value,
                            reflectance_value };
        // r.data[b].decay = { 0.5f, 0.5f, 0.5f };

        r.data[b].ray_direct = eye.reflect(material.normal);
    }

    return r;
}

} // namespace eta

#endif

#ifndef ETA_MODEL_CU
#define ETA_MODEL_CU

#include "Model.cuh"

namespace eta {

Matrix<float, 4, 4>
normalize_vertex_coord(const std::vector<float>& vertex_coord) {
    int num_of_faces{ static_cast<int>(vertex_coord.size()) / 3 };

    float min_x{ ETA_inf };
    float max_x{ -ETA_inf };
    float min_y{ ETA_inf };
    float max_y{ -ETA_inf };
    float min_z{ ETA_inf };
    float max_z{ -ETA_inf };

    for (int i{ 0 }; i < num_of_faces; ++i) {
        float x{ vertex_coord[3 * i + 0] };
        float y{ vertex_coord[3 * i + 1] };
        float z{ vertex_coord[3 * i + 2] };

        min_x = min(min_x, x);
        max_x = max(max_x, x);
        min_y = min(min_y, y);
        max_y = max(max_y, y);
        min_z = min(min_z, z);
        max_z = max(max_z, z);
    }

    float center_x{ (min_x + max_x) / 2.0f };
    float center_y{ (min_y + max_y) / 2.0f };
    float center_z{ (min_z + max_z) / 2.0f };

    float scale_x{ 2.0f / (max_x - min_x) };
    float scale_y{ 2.0f / (max_y - min_y) };
    float scale_z{ 2.0f / (max_z - min_z) };

    float scale{ min(scale_x, min(scale_y, scale_z)) };

    return Matrix<float, 4, 4>{
        scale, 0,     0,     -center_x * scale, //
        0,     scale, 0,     -center_y * scale, //
        0,     0,     scale, -center_z * scale, //
        0,     0,     0,     1, //
    };
}

} // namespace eta

#endif

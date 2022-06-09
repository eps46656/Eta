#ifndef ETA_SIMPLEMODEL_CUH
#define ETA_SIMPLEMODEL_CUH

#include "Model.cuh"
#include "SafeAttrView.cuh"
#include "Image.cuh"

namespace eta {

class SimpleModel: public Model {
public:
    struct MaterialGetter {
        Matrix<float, 4, 4> transform;
        AttrView<Vector<float, 3>> face_normal;
        AttrView<Vector<float, 3>> face_color;

        __device__ Material operator()(int face_i,
                                       const Vector<float, 3>& n) const;
    };

    __host__ int num_of_faces() const;

    __host__ Matrix<float, 4, 4>& transform();
    __host__ const Matrix<float, 4, 4>& transform() const;

    __host__ SafeAttrView<CPU, Vector<float, 3>>& face_coord_cpu();
    __host__ const SafeAttrView<CPU, Vector<float, 3>>& face_coord_cpu() const;

    __host__ SafeAttrView<GPU, Vector<float, 3>>& face_coord_gpu();
    __host__ const SafeAttrView<GPU, Vector<float, 3>>& face_coord_gpu() const;

    __host__ SafeAttrView<CPU, Vector<float, 3>>& face_normal_cpu();
    __host__ const SafeAttrView<CPU, Vector<float, 3>>& face_normal_cpu() const;

    __host__ SafeAttrView<GPU, Vector<float, 3>>& face_normal_gpu();
    __host__ const SafeAttrView<GPU, Vector<float, 3>>& face_normal_gpu() const;

    __host__ SafeAttrView<CPU, Vector<float, 3>>& face_color_cpu();
    __host__ const SafeAttrView<CPU, Vector<float, 3>>& face_color_cpu() const;

    __host__ SafeAttrView<GPU, Vector<float, 3>>& face_color_gpu();
    __host__ const SafeAttrView<GPU, Vector<float, 3>>& face_color_gpu() const;

    __host__ void set_num_of_faces(int num_of_faces);

    __host__ void set_use_tex(bool use_tex);

    __host__ void set_tex(View img_view);
    __host__ void set_tex(const Image& img);

    __host__ void clear();

    __host__ SimpleModel();
    __host__ ~SimpleModel();

    __host__ void LoadFromDir(const std::string& dir, bool normalize_face_coord,
                              AttrViewMode face_normal_mode,
                              AttrViewMode face_color_mode);

    __host__ void LoadOn(cudaStream_t stream) override;

    __host__ void Rasterize(int height, int width, //

                            float* dst_w, //
                            Material* dst_material, //

                            int* locker, //

                            Matrix<float, 3, 4> camera_mat, // camera mat

                            cudaStream_t stream // stream
    ) const override;

    __host__ void RayCast(int num_of_rays, // num_of_rays

                          float* dst_w, //
                          Material* dst_material, //

                          float* eff, //
                          Vector<float, 3>* ray_origin, //
                          Vector<float, 3>* ray_direct, //
                          int* locker, //

                          cudaStream_t stream // stream
    ) const override;

private:
    int num_of_faces_;

    Matrix<float, 4, 4> transform_{ Matrix<float, 4, 4>::eye() };

    SafeAttrView<CPU, Vector<float, 3>> face_coord_cpu_;
    SafeAttrView<GPU, Vector<float, 3>> face_coord_gpu_;

    SafeAttrView<CPU, Vector<float, 3>> face_normal_cpu_;
    SafeAttrView<GPU, Vector<float, 3>> face_normal_gpu_;

    SafeAttrView<CPU, Vector<float, 3>> face_color_cpu_;
    SafeAttrView<GPU, Vector<float, 3>> face_color_gpu_;

    MaterialGetter material_getter_;
};

} // namespace eta

#endif

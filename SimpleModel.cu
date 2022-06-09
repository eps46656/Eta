#ifndef ETA_SIMPLEMODEL_CU
#define ETA_SIMPLEMODEL_CU

#include "utils.cuh"
#include "SimpleModel.cuh"
#include "Model.cu"
#include "io.cuh"
#include "Image.cu"
#include "Rasterize.cu"
#include "RayCast.cu"

namespace eta {

int SimpleModel::num_of_faces() const { return this->num_of_faces_; }

Matrix<float, 4, 4>& SimpleModel::transform() { return this->transform_; }
const Matrix<float, 4, 4>& SimpleModel::transform() const {
    return this->transform_;
}

SafeAttrView<CPU, Vector<float, 3>>& SimpleModel::face_coord_cpu() {
    return this->face_coord_cpu_;
}
const SafeAttrView<CPU, Vector<float, 3>>& SimpleModel::face_coord_cpu() const {
    return this->face_coord_cpu_;
}

SafeAttrView<GPU, Vector<float, 3>>& SimpleModel::face_coord_gpu() {
    return this->face_coord_gpu_;
}
const SafeAttrView<GPU, Vector<float, 3>>& SimpleModel::face_coord_gpu() const {
    return this->face_coord_gpu_;
}

SafeAttrView<CPU, Vector<float, 3>>& SimpleModel::face_normal_cpu() {
    return this->face_normal_cpu_;
}
const SafeAttrView<CPU, Vector<float, 3>>&
SimpleModel::face_normal_cpu() const {
    return this->face_normal_cpu_;
}

SafeAttrView<GPU, Vector<float, 3>>& SimpleModel::face_normal_gpu() {
    return this->face_normal_gpu_;
}
const SafeAttrView<GPU, Vector<float, 3>>&
SimpleModel::face_normal_gpu() const {
    return this->face_normal_gpu_;
}

SafeAttrView<CPU, Vector<float, 3>>& SimpleModel::face_color_cpu() {
    return this->face_color_cpu_;
}
const SafeAttrView<CPU, Vector<float, 3>>& SimpleModel::face_color_cpu() const {
    return this->face_color_cpu_;
}

SafeAttrView<GPU, Vector<float, 3>>& SimpleModel::face_color_gpu() {
    return this->face_color_gpu_;
}
const SafeAttrView<GPU, Vector<float, 3>>& SimpleModel::face_color_gpu() const {
    return this->face_color_gpu_;
}

void SimpleModel::set_num_of_faces(int num_of_faces) {
    this->num_of_faces_ = num_of_faces;

    this->face_coord_cpu_.set_num_of_faces(this->num_of_faces_);
    this->face_coord_gpu_.set_num_of_faces(this->num_of_faces_);

    this->face_normal_cpu_.set_num_of_faces(this->num_of_faces_);
    this->face_normal_gpu_.set_num_of_faces(this->num_of_faces_);

    this->face_color_cpu_.set_num_of_faces(this->num_of_faces_);
    this->face_color_gpu_.set_num_of_faces(this->num_of_faces_);
}

void SimpleModel::clear() {
    this->num_of_faces_ = 0;

    this->face_coord_cpu_.clear();
    this->face_coord_gpu_.clear();

    this->face_normal_cpu_.clear();
    this->face_normal_gpu_.clear();

    this->face_color_cpu_.clear();
    this->face_color_gpu_.clear();
}

SimpleModel::SimpleModel(): num_of_faces_{ 0 } {}

SimpleModel::~SimpleModel() { this->clear(); }

void SimpleModel::LoadFromDir(const std::string& dir, bool normalize_face_coord,
                              AttrViewMode face_normal_mode,
                              AttrViewMode face_color_mode) {
    std::vector<float> face_coord{ read_vector<float>(
        join_path(dir, "face_coord.txt")) };

    int num_of_faces{ static_cast<int>(face_coord.size()) / 9 };
    this->set_num_of_faces(num_of_faces);

    Print(num_of_faces, "\n");

    ////////////////////////////////////////////////////////////////////////////

    if (normalize_face_coord) {
        this->transform_ = normalize_vertex_coord(face_coord);
    } else {
        this->transform_ = Matrix<float, 4, 4>::eye();
    }

    for (int face_i{ 0 }; face_i < num_of_faces; ++face_i) {
        for (int vertex_i{ 0 }; vertex_i < 3; ++vertex_i) {
            this->face_coord_cpu_.attr().get<Vector<float, 3>>(face_i,
                                                               vertex_i) = {
                face_coord[9 * face_i + 3 * vertex_i + 0],
                face_coord[9 * face_i + 3 * vertex_i + 1],
                face_coord[9 * face_i + 3 * vertex_i + 2],
            };
        }
    }

    ////////////////////////////////////////////////////////////////////////////

    this->face_normal_cpu_.set_mode(face_normal_mode);

    switch (face_normal_mode) {
        case VERTEX: {
            std::vector<float> face_normal{ read_vector<float>(
                join_path(dir, "face_normal.txt")) };

            for (int face_i{ 0 }; face_i < num_of_faces; ++face_i) {
                for (int vertex_i{ 0 }; vertex_i < 3; ++vertex_i) {
                    this->face_normal_cpu_.attr().get<Vector<float, 3>>(
                        face_i, vertex_i) = {
                        face_normal[9 * face_i + 3 * vertex_i + 0],
                        face_normal[9 * face_i + 3 * vertex_i + 1],
                        face_normal[9 * face_i + 3 * vertex_i + 2],
                    };
                }
            }

            break;
        }

        case TEXTURE: {
            std::vector<float> face_normal_tex_coord{ read_vector<float>(
                join_path(dir, "face_normal_tex_coord.txt")) };

            for (int face_i{ 0 }; face_i < num_of_faces; ++face_i) {
                for (int vertex_i{ 0 }; vertex_i < 3; ++vertex_i) {
                    this->face_normal_cpu_.tex_coord().get<Vector<float, 2>>(
                        face_i, vertex_i) = {
                        face_normal_tex_coord[6 * face_i + 2 * vertex_i + 0],
                        face_normal_tex_coord[6 * face_i + 2 * vertex_i + 1]
                    };
                }
            }

            Image face_normal_tex;
            face_normal_tex.load(join_path(dir, "face_normal.png"));

            this->face_normal_cpu_.set_tex_shape(face_normal_tex.height(),
                                                 face_normal_tex.width());

            for (int i{ 0 }; i < face_normal_tex.height(); ++i) {
                for (int j{ 0 }; j < face_normal_tex.width(); ++j) {
                    this->face_normal_cpu_.tex().data.get<Vector<float, 3>>(
                        i, j) = {
                        face_normal_tex.get(i, j, 0) * 2.0f / 255.0f - 1,
                        face_normal_tex.get(i, j, 1) * 2.0f / 255.0f - 1,
                        face_normal_tex.get(i, j, 2) * 2.0f / 255.0f - 1,
                    };
                }
            }

            break;
        }
    }

    ////////////////////////////////////////////////////////////////////////////

    this->face_color_cpu_.set_mode(face_color_mode);

    switch (face_color_mode) {
        case VERTEX: {
            std::vector<float> face_color{ read_vector<float>(
                join_path(dir, "face_color.txt")) };

            for (int face_i{ 0 }; face_i < num_of_faces; ++face_i) {
                for (int vertex_i{ 0 }; vertex_i < 3; ++vertex_i) {
                    this->face_color_cpu_.attr().get<Vector<float, 3>>(
                        face_i, vertex_i) = {
                        face_color[9 * face_i + 3 * vertex_i + 0],
                        face_color[9 * face_i + 3 * vertex_i + 1],
                        face_color[9 * face_i + 3 * vertex_i + 2],
                    };
                }
            }

            break;
        }

        case TEXTURE: {
            std::vector<float> face_color_tex_coord{ read_vector<float>(
                join_path(dir, "face_color_tex_coord.txt")) };

            for (int face_i{ 0 }; face_i < num_of_faces; ++face_i) {
                for (int vertex_i{ 0 }; vertex_i < 3; ++vertex_i) {
                    this->face_color_cpu_.tex_coord().get<Vector<float, 2>>(
                        face_i, vertex_i) = {
                        face_color_tex_coord[6 * face_i + 2 * vertex_i + 0],
                        face_color_tex_coord[6 * face_i + 2 * vertex_i + 1]
                    };
                }
            }

            Image face_color_tex;
            face_color_tex.load(join_path(dir, "face_color_tex.png"));

            this->face_color_cpu_.set_tex_shape(face_color_tex.height(),
                                                face_color_tex.width());

            for (int i{ 0 }; i < face_color_tex.height(); ++i) {
                for (int j{ 0 }; j < face_color_tex.width(); ++j) {
                    Vector<float, 3> v{
                        face_color_tex.get(i, j, 0) / 255.0f,
                        face_color_tex.get(i, j, 1) / 255.0f,
                        face_color_tex.get(i, j, 2) / 255.0f,
                    };

                    this->face_color_cpu_.tex().data.get<Vector<float, 3>>(
                        i, j) = v;
                }
            }

            break;
        }
    }
}

void SimpleModel::LoadOn(cudaStream_t stream) {
    this->face_coord_cpu_.PassTo(this->face_coord_gpu_, stream);
    this->face_normal_cpu_.PassTo(this->face_normal_gpu_, stream);
    this->face_color_cpu_.PassTo(this->face_color_gpu_, stream);

    this->material_getter_.transform = this->transform_;
    this->material_getter_.face_normal = this->face_normal_gpu_.attr_view();
    this->material_getter_.face_color = this->face_color_gpu_.attr_view();
}

void SimpleModel::Rasterize(int height, int width, //

                            float* dst_w, //
                            Material* dst_material, //

                            int* locker, //

                            Matrix<float, 3, 4> camera_mat, // camera mat

                            cudaStream_t stream // stream
) const {
    eta::Rasterize<MaterialGetter>
        <<<16, 1024, 0, stream>>>(height, width, //

                                  dst_w, //
                                  dst_material, //

                                  locker, //

                                  this->num_of_faces_, // num_of_faces
                                  this->face_coord_gpu_.attr(), // face_coord

                                  camera_mat * this->transform_, // transform

                                  this->material_getter_ // material_getter
        );
}

void SimpleModel::RayCast(int num_of_rays, //

                          float* dst_w, //
                          Material* dst_material, //

                          float* eff, //
                          Vector<float, 3>* ray_origin, //
                          Vector<float, 3>* ray_direct, //
                          int* locker, //

                          cudaStream_t stream //
) const {
    eta::RayCast<MaterialGetter>
        <<<64, 1024, 0, stream>>>(num_of_rays, //

                                  dst_w, //
                                  dst_material, //

                                  eff, //
                                  ray_origin, //
                                  ray_direct, //
                                  locker, //

                                  this->num_of_faces_, //
                                  this->face_coord_gpu_.attr(), //

                                  this->transform_, //

                                  this->material_getter_ //
        );
}

__device__ Material SimpleModel::MaterialGetter::operator()(
    int face_i, const Vector<float, 3>& n) const {
    Material r;

    Vector<float, 3> normal{ this->face_normal.get(face_i, n) };

#pragma unroll
    for (int i{ 0 }; i < 3; ++i) {
        r.normal.data[i] = this->transform.data[i][0] * normal.data[0] +
                           this->transform.data[i][1] * normal.data[1] +
                           this->transform.data[i][2] * normal.data[2];
    }

    r.normal = r.normal.normalize();

    Vector<float, 3> color{ face_color.get(face_i, n) };

    r.diffuse_color = color;
    r.specular_color = color;
    r.shininess = { 2.0f, 2.0f, 2.0f };
    r.transparent = { 0.0f, 0.0f, 0.0f };

    return r;
}

} // namespace eta

#endif

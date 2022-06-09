#ifndef ETA_CAGE_CU
#define ETA_CAGE_CU

#include "Cage.cuh"

namespace eta {

Vector<float, 3>& Cage::resolution() { return this->origin_; }
const Vector<float, 3>& Cage::resolution() const { return this->origin_; }

int Cage::resolution() const { return this->resolution_; }

int Cage::Init(int resolution) {
    this->resolution_ = resolution;

    int resolution_sq{ resolution * resolution };

    for (int d{ 0 }; d < 6; ++d) {
        this->tex[d].data.base =
            this->tex_mem_[d].Resize(sizeof(int) * resolution_sq);

        this->tex[d].data.coeff[0] = sizeof(int) * resolution;
        this->tex[d].data.coeff[1] = sizeof(int);
        this->tex[d].height = this->cage_.tex[d].width = resolution;
        setvalue<int>
            <<<1, 1024, 0, stream>>>(this->tex[d].data.base, resolution_sq, -1);

        this->w_[d].base =
            this->w_mem_[d].Resize(sizeof(float) * resolution_sq);
        this->w_[d].coeff[0] = sizeof(float) * resolution;
        this->w_[d].coeff[1] = sizeof(float);
        setvalue<int>
            <<<1, 1024, 0, stream>>>(this->w_[d].base, resolution_sq, 0);
    }
}

void Cage::Generate(const Model* model, cudaStream_t stream) {
    if (this->resolution_ == 0) { ETA_throw("uninitialized\n"); }

    for (int d{ 0 }; d < 6; ++d) {
        model->Rasterize(this->resolution_,
                         this->resolution_, // height, width

                         this->tex[d].data, // dst_id
                         this->w_[d], // dst_w

                         false, // record_n
                         VectorView{}, // dst_n1

                         perspective_mat(this->origin_, // origin
                                         TextureCubeVector::r[d], // right
                                         TextureCubeVector::u[d], // up
                                         TextureCubeVector::inside_f[d] // front
                                         ), // transform

                         stream // stream
        );
    }
}

} // namespace eta

#endif
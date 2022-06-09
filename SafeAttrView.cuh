#ifndef ETA_SAFEATTRVIEW_CUH
#define ETA_SAFEATTRVIEW_CUH

#include "AttrView.cuh"
#include "Buffer.cuh"

namespace eta {

template<typename D, typename T> class SafeAttrView {
public:
    template<typename D_, typename T_> friend class SafeAttrView;

    using device = D;

    __host__ __device__ AttrViewMode mode() const {
        return this->attr_view_.mode;
    }
    __host__ void set_mode(AttrViewMode mode) {
        this->attr_view_.mode = mode;
        this->set_num_of_faces(this->attr_view_.num_of_faces);
    }

    __host__ __device__ int num_of_faces() const {
        return this->attr_view_.num_of_faces;
    }
    __host__ void set_num_of_faces(int num_of_faces) {
        this->attr_view_.num_of_faces = num_of_faces;

        switch (this->attr_view_.mode) {
            case VERTEX:
                this->attr_view_.attr.base =
                    this->attr_mem_.Resize(sizeof(T) * 3 * num_of_faces);
                break;
            case TEXTURE:
                this->attr_view_.tex_coord.base = this->tex_coord_mem_.Resize(
                    sizeof(Vector<float, 2>) * 3 * num_of_faces);
        }
    }

    __host__ __device__ AttrView<T> attr_view() const {
        return this->attr_view_;
    }

    __host__ __device__ View attr() const { return this->attr_view_.attr; }

    __host__ __device__ View tex_coord() const {
        return this->attr_view_.tex_coord;
    }

    __host__ __device__ Texture2D<T>& tex() { return this->attr_view_.tex; }
    __host__ __device__ const Texture2D<T>& tex() const {
        return this->attr_view_.tex;
    }

    __host__ void set_tex_shape(int height, int width) {
        this->attr_view_.tex.height = height;
        this->attr_view_.tex.width = width;

        this->attr_view_.tex.data.coeff[0] = sizeof(T) * width;
        this->attr_view_.tex.data.coeff[1] = sizeof(T);

        int size{ height * width };

        this->attr_view_.tex.data.base =
            this->tex_mem_.Resize(sizeof(T) * size);
    }

    __host__ void clear() {
        this->attr_view_.num_of_faces = 0;

        this->attr_view_.attr.base = nullptr;
        this->attr_view_.tex_coord.base = nullptr;
        this->attr_view_.tex.data.base = nullptr;

        this->attr_mem_.clear();
        this->tex_coord_mem_.clear();
        this->tex_mem_.clear();
    }

    __host__ SafeAttrView() {
        this->attr_view_.attr.coeff[0] = sizeof(T) * 3;
        this->attr_view_.attr.coeff[1] = sizeof(T);

        this->attr_view_.tex_coord.coeff[0] = sizeof(Vector<float, 2>) * 3;
        this->attr_view_.tex_coord.coeff[1] = sizeof(Vector<float, 2>);
    }

    __host__ __device__ T get(int face_idx, Vector<float, 3> n) const {
        return this->attr_view_.get(face_idx, n);
    }

    template<typename D_>
    __host__ void PassTo(SafeAttrView<D_, T>& safe_attr_view,
                         cudaStream_t stream) {
        int num_of_faces{ this->attr_view_.num_of_faces };

        safe_attr_view.set_mode(this->attr_view_.mode);
        safe_attr_view.set_num_of_faces(num_of_faces);

        cudaMemcpyKind kind{ cuda_memcpy_kind[D::value][D_::value] };

        switch (this->attr_view_.mode) {
            case VERTEX: {
                this->attr_mem_.PassTo(safe_attr_view.attr_mem_,
                                       sizeof(T) * 3 * num_of_faces, stream);

                break;
            }

            case TEXTURE: {
                int height{ this->attr_view_.tex.height };
                int width{ this->attr_view_.tex.width };

                safe_attr_view.set_tex_shape(height, width);

                safe_attr_view.attr_view_.tex.border_value =
                    this->attr_view_.tex.border_value;

                safe_attr_view.attr_view_.tex.s_wrapping_mode =
                    this->attr_view_.tex.s_wrapping_mode;
                safe_attr_view.attr_view_.tex.s_filtering_mode =
                    this->attr_view_.tex.s_filtering_mode;

                safe_attr_view.attr_view_.tex.t_wrapping_mode =
                    this->attr_view_.tex.t_wrapping_mode;
                safe_attr_view.attr_view_.tex.t_filtering_mode =
                    this->attr_view_.tex.t_filtering_mode;

                this->tex_coord_mem_.PassTo(
                    safe_attr_view.tex_coord_mem_,
                    sizeof(Vector<float, 2>) * 3 * num_of_faces, stream);

                /*cudaMemcpyAsync(
                    safe_attr_view.attr_view_.tex_coord.base, // dst
                    this->attr_view_.tex_coord.base, // src
                    sizeof(Vector<float, 2>) * 3 * num_of_faces, // size
                    kind, // kind
                    stream // stream
                );*/

                this->tex_mem_.PassTo(safe_attr_view.tex_mem_,
                                      sizeof(T) * height * width, stream);

                /* cudaMemcpyAsync(safe_attr_view.attr_view_.tex.data.base, // dst
                                this->attr_view_.tex.data.base, // src
                                sizeof(T) * height * width, // size
                                kind, // kind
                                stream // stream
                ); */

                break;
            }
        }
    }

private:
    AttrView<T> attr_view_;

    Buffer<D> attr_mem_;
    Buffer<D> tex_coord_mem_;
    Buffer<D> tex_mem_;
};

} // namespace eta

#endif

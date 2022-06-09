#ifndef ETA_ATTRVIEW_CUH
#define ETA_ATTRVIEW_CUH

#include "Texture2D.cuh"

namespace eta {

enum AttrViewMode {
    VERTEX,
    TEXTURE,
};

template<typename T> struct AttrView {
    AttrViewMode mode{ VERTEX };
    int num_of_faces;

    // vertex mode
    View attr; // T [num_of_faces, 3]

    // texture mode
    View tex_coord; // Vector<float, 2> [num_of_faces, 3]
    Texture2D<T> tex; // T [height, width]

    __host__ __device__ T get(int face_idx, Vector<float, 3> n) const {
        switch (this->mode) {
            case VERTEX: {
                T a{ this->attr.get<T>(face_idx, 0) };
                T b{ this->attr.get<T>(face_idx, 1) };
                T c{ this->attr.get<T>(face_idx, 2) };

                return a * n.data[0] + b * n.data[1] + c * n.data[2];
            }

            case TEXTURE: {
                Vector<float, 2> a{ this->tex_coord.get<Vector<float, 2>>(
                    face_idx, 0) };
                Vector<float, 2> b{ this->tex_coord.get<Vector<float, 2>>(
                    face_idx, 1) };
                Vector<float, 2> c{ this->tex_coord.get<Vector<float, 2>>(
                    face_idx, 2) };

                return this->tex.get(a * n.data[0] + b * n.data[1] +
                                     c * n.data[2]);
            }
        }
    }

    __host__ void PassValue(AttrView& attr_view) {
        attr_view.num_of_faces = this->num_of_faces;
        attr_view.mode = this->mode_;
    }
};

} // namespace eta

#endif

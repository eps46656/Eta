#ifndef ETA_UTILS_CUH
#define ETA_UTILS_CUH

#include "Matrix.cuh"

namespace eta {

template<typename T, int N> struct Array { T data[N]; };

struct CPU {
    static constexpr int value{ 0 };
};

struct GPU {
    static constexpr int value{ 1 };
};

////////////////////////////////////////////////////////////////////////////////

template<typename T> __host__ __device__ T clamp(T v, T lo, T hi) {
    return ::min(hi, ::max(lo, v));
}

////////////////////////////////////////////////////////////////////////////////

template<typename D, typename T> struct MallocHelper;

template<typename T> struct MallocHelper<CPU, T> {
    __host__ static T* Malloc(size_t size) {
        return static_cast<T*>(std::malloc(size));
    }
};

template<typename T> struct MallocHelper<GPU, T> {
    __host__ static T* Malloc(size_t size) {
        T* r;
        ETA_CheckCudaError(cudaMalloc(&r, size));
        return r;
    }
};

template<typename D, typename T = void> __host__ T* Malloc(size_t size) {
    return MallocHelper<D, T>::Malloc(size);
}

////////////////////////////////////////////////////////////////////////////////

template<typename D> struct FreeHelper;

template<> struct FreeHelper<CPU> {
    __host__ static void Free(void* ptr) { std::free(ptr); }
};

template<> struct FreeHelper<GPU> {
    __host__ static void Free(void* ptr) { ETA_CheckCudaError(cudaFree(ptr)); }
};

template<typename D> __host__ void Free(void* ptr) { FreeHelper<D>::Free(ptr); }

////////////////////////////////////////////////////////////////////////////////

template<typename Dst, typename Src> struct MemcpyHelper;

template<> struct MemcpyHelper<CPU, CPU> {
    __host__ static void Memcpy(void* dst, void* src, size_t size) {
        cudaMemcpy(dst, src, size, cudaMemcpyHostToHost);
    }
};

template<> struct MemcpyHelper<CPU, GPU> {
    __host__ static void Memcpy(void* dst, void* src, size_t size) {
        cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    }
};

template<> struct MemcpyHelper<GPU, CPU> {
    __host__ static void Memcpy(void* dst, void* src, size_t size) {
        cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    }
};

template<> struct MemcpyHelper<GPU, GPU> {
    __host__ static void Memcpy(void* dst, void* src, size_t size) {
        cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    }
};

template<typename Dst, typename Src>
void Memcpy(void* dst, void* src, size_t size) {
    MemcpyHelper<Dst, Src>::Memcpy(dst, src, size);
}

////////////////////////////////////////////////////////////////////////////////

template<typename Dst, typename Src> struct MemcpyAsyncHelper;

template<> struct MemcpyAsyncHelper<CPU, CPU> {
    __host__ static void MemcpyAsync(void* dst, void* src, size_t size,
                                     cudaStream_t stream) {
        cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToHost, stream);
    }
};

template<> struct MemcpyAsyncHelper<CPU, GPU> {
    __host__ static void MemcpyAsync(void* dst, void* src, size_t size,
                                     cudaStream_t stream) {
        cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream);
    }
};

template<> struct MemcpyAsyncHelper<GPU, CPU> {
    __host__ static void MemcpyAsync(void* dst, void* src, size_t size,
                                     cudaStream_t stream) {
        cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
    }
};

template<> struct MemcpyAsyncHelper<GPU, GPU> {
    __host__ static void MemcpyAsync(void* dst, void* src, size_t size,
                                     cudaStream_t stream) {
        cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);
    }
};

template<typename Dst, typename Src>
void MemcpyAsync(void* dst, void* src, size_t size, cudaStream_t stream) {
    MemcpyAsyncHelper<Dst, Src>::MemcpyAsync(dst, src, size, stream);
}

////////////////////////////////////////////////////////////////////////////////

static_assert(CPU::value == 0);
static_assert(GPU::value == 1);

constexpr cudaMemcpyKind cuda_memcpy_kind[2][2]{
    { cudaMemcpyHostToHost, cudaMemcpyHostToDevice },
    { cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice },
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template<typename T>
__host__ __device__ T linear(const T& a, const T& b, float ratio) {
    return a * (1 - ratio) + b * ratio;
}

struct DefaultLinear {
    template<typename T>
    __host__ __device__ T operator()(const T& a, const T& b,
                                     float ratio) const {
        return linear(a, b, ratio);
    }
};

template<typename T>
__host__ __device__ T bilinear(const T& x00, const T& x01, const T& x10,
                               const T& x11, float ratio_0, float ratio_1) {
    return linear(linear(x00, x10, ratio_0), linear(x01, x11, ratio_0),
                  ratio_1);
}

struct DefaultBiliear {
    template<typename T>
    __host__ __device__ T operator()(const T& a, const T& b,
                                     float ratio) const {
        return bilinear(a, b, ratio);
    }
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

Matrix<float, 4, 4> translation_mat(const Vector<float, 3>& translation) {
    return {
        1, 0, 0, translation.data[0], //
        0, 1, 0, translation.data[1], //
        0, 0, 1, translation.data[2], //
        0, 0, 0, 1 //
    };
}

Matrix<float, 4, 4> scale_mat(const Vector<float, 3>& scale) {
    return {
        scale.data[0],
        0,
        0,
        0, //
        0,
        scale.data[1],
        0,
        0, //
        0,
        0,
        scale.data[2],
        0, //
        0,
        0,
        0,
        1 //
    };
}

Matrix<float, 4, 4> rotation_mat(const Vector<float, 3>& axis, float theta) {
    float rn{ 1 / axis.norm() };
    float x{ axis.data[0] * rn };
    float y{ axis.data[1] * rn };
    float z{ axis.data[2] * rn };

    float s{ std::sin(theta) };
    float c{ std::cos(theta) };

    return { c + x * x * (1 - c),
             x * y * (1 - c) - z * s,
             x * z * (1 - c) + y * s,
             0,

             y * x * (1 - c) + z * s,
             c + y * y * (1 - c),
             y * z * (1 - c) - x * s,
             0,

             z * x * (1 - c) - y * s,
             z * y * (1 - c) + x * s,
             c + z * z * (1 - c),
             0,

             0,
             0,
             0,
             1 };
}

void look_at_origin(Vector<float, 3>& dst_r, // dst right
                    Vector<float, 3>& dst_u, // dst up
                    Vector<float, 3>& dst_f, // dst front
                    Vector<float, 3> point, float fov, float aspect) {
    dst_f = point.normalize();
    dst_r = cross(Vector<float, 3>{ 0, 0, 1 }, point).renorm(tan(fov / 2));
    dst_u = cross(dst_f, dst_r) / aspect;
}

Matrix<float, 3, 4> perspective_mat(const Vector<float, 3>& p, // view_point
                                    const Vector<float, 3>& r, // right
                                    const Vector<float, 3>& u, // up
                                    const Vector<float, 3>& f // front
) {
    Matrix<float, 3, 3> M;

    M.data[0][0] = r.data[0];
    M.data[0][1] = u.data[0];
    M.data[0][2] = -f.data[0];

    M.data[1][0] = r.data[1];
    M.data[1][1] = u.data[1];
    M.data[1][2] = -f.data[1];

    M.data[2][0] = r.data[2];
    M.data[2][1] = u.data[2];
    M.data[2][2] = -f.data[2];

    Matrix<float, 3, 3> A{ M.inv() };
    Vector<float, 3> B{ A * p };

    return {
        A.data[0][0], A.data[0][1], A.data[0][2], -B.data[0], //
        A.data[1][0], A.data[1][1], A.data[1][2], -B.data[1], //
        A.data[2][0], A.data[2][1], A.data[2][2], -B.data[2], //
    };
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template<typename T>
__global__ void setvalue(void* data, size_t size, T value) {
    for (int i{ static_cast<int>(GLOBAL_TID) }; i < size;
         i += gridDim.x * blockDim.x) {
        static_cast<T*>(data)[i] = value;
    }
}

} // namespace eta

#endif

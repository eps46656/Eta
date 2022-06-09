#ifndef ETA_VECTOR_CUH
#define ETA_VECTOR_CUH

#include "Print.cuh"

namespace eta {

template<typename T, int N> struct Vector {
    static_assert(1 <= N && N <= 4, "1 <= N && N <= 4");

    static constexpr int n{ N };

    T data[N];

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////

    __host__ __device__ Vector<T, N>& operator+() { return *this; }

    __host__ __device__ const Vector<T, N>& operator+() const { return *this; }

    __host__ __device__ Vector<T, N> operator-() const {
        Vector<T, N> r;
        for (int i{ 0 }; i < N; ++i) { r.data[i] = -this->data[i]; }
        return r;
    }

    ////////////////////////////////////////////////////////////////////////////

    __host__ __device__ T sq() const { return dot(*this, *this); }
    __host__ __device__ T norm() const { return sqrt(this->sq()); }

    __host__ __device__ Vector<T, N> normalize() const {
        return (*this) / this->norm();
    }
    __host__ __device__ bool is_normalized() const {
        T n{ this->norm() };
        return ETA__eq_const(n, 1);
    }

    __host__ __device__ Vector<T, N> renorm(T norm) const {
        return (*this) * (norm / this->norm());
    }

    __host__ __device__ Vector<T, N> project(const Vector<T, N>& normal) const {
        return normal * dot(*this, normal);
    }
    __host__ __device__ Vector<T, N> reject(const Vector<T, N>& normal) const {
        T k{ dot(*this, normal) };
        Vector<T, N> r;

        for (int i{ 0 }; i < N; ++i) {
            r.data[i] = this->data[i] - normal.data[i] * k;
        }

        return r;
    }
    __host__ __device__ Vector<T, N> reflect(const Vector<T, N>& normal) const {
        T k{ 2 * dot(*this, normal) };
        Vector<T, N> r;

        for (int i{ 0 }; i < N; ++i) {
            r.data[i] = this->data[i] - normal.data[i] * k;
        }

        return r;
    }

    ////////////////////////////////////////////////////////////////////////////

    __host__ __device__ static Vector<T, N> zero() {
        Vector<T, N> r;
        for (int i{ 0 }; i < N; ++i) { r.data[i] = 0; }
        return r;
    }

    __host__ __device__ static Vector<T, N> one() {
        Vector<T, N> r;
        for (int i{ 0 }; i < N; ++i) { r.data[i] = 1; }
        return r;
    }
};

template<typename T, int N>
__host__ __device__ bool operator==(const Vector<T, N>& vec_a,
                                    const Vector<T, N>& vec_b) {
    if (&vec_a == &vec_b) { return true; }

    for (int i{ 0 }; i < N; ++i) {
        if (ETA_le_const(abs(vec_a.data[i] - vec_b.data[i]), 0)) {
            return false;
        }
    }

    return true;
}
template<typename T, int N>
__host__ __device__ bool operator!=(const Vector<T, N>& vec_a,
                                    const Vector<T, N>& vec_b) {
    return !(vec_a == vec_b);
}

template<typename T, int N>
__host__ __device__ Vector<T, N> operator+(const Vector<T, N>& vec_a,
                                           const Vector<T, N>& vec_b) {
    Vector<T, N> r;
    for (int i{ 0 }; i < N; ++i) { r.data[i] = vec_a.data[i] + vec_b.data[i]; }
    return r;
}
template<typename T, int N>
__host__ __device__ Vector<T, N> operator-(const Vector<T, N>& vec_a,
                                           const Vector<T, N>& vec_b) {
    Vector<T, N> r;
    for (int i{ 0 }; i < N; ++i) { r.data[i] = vec_a.data[i] - vec_b.data[i]; }
    return r;
}
template<typename T, int N>
__host__ __device__ Vector<T, N> operator*(const Vector<T, N>& vec_a, T value) {
    Vector<T, N> r;
    for (int i{ 0 }; i < N; ++i) { r.data[i] = vec_a.data[i] * value; }
    return r;
}
template<typename T, int N>
__host__ __device__ Vector<T, N> operator/(const Vector<T, N>& vec_a, T value) {
    Vector<T, N> r;
    for (int i{ 0 }; i < N; ++i) { r.data[i] = vec_a.data[i] / value; }
    return r;
}

template<typename T, int N>
__host__ __device__ Vector<T, N>& operator+=(Vector<T, N>& vec_a,
                                             const Vector<T, N>& vec_b) {
    for (int i{ 0 }; i < N; ++i) { vec_a.data[i] += vec_b.data[i]; }
    return vec_a;
}
template<typename T, int N>
__host__ __device__ Vector<T, N>& operator-=(Vector<T, N>& vec_a,
                                             const Vector<T, N>& vec_b) {
    for (int i{ 0 }; i < N; ++i) { vec_a.data[i] -= vec_b.data[i]; }
    return vec_a;
}
template<typename T, int N>
__host__ __device__ Vector<T, N>& operator*=(Vector<T, N>& vec_a, T n) {
    for (int i{ 0 }; i < N; ++i) { vec_a.data[i] *= n; }
    return vec_a;
}
template<typename T, int N>
__host__ __device__ Vector<T, N>& operator/=(Vector<T, N>& vec_a, T n) {
    for (int i{ 0 }; i < N; ++i) { vec_a.data[i] /= n; }
    return vec_a;
}

template<typename T, int N>
__host__ __device__ T dot(const Vector<T, N>& vec_a,
                          const Vector<T, N>& vec_b) {
    T r{ 0 };
    for (int i{ 0 }; i < N; ++i) { r += vec_a.data[i] * vec_b.data[i]; }
    return r;
}

template<typename T, int N>
__host__ __device__ Vector<T, N> cross(const Vector<T, N>& vec_a,
                                       const Vector<T, N>& vec_b) {
    static_assert(N == 3, "N == 3");
    return { vec_a.data[1] * vec_b.data[2] - vec_a.data[2] * vec_b.data[1],
             vec_a.data[2] * vec_b.data[0] - vec_a.data[0] * vec_b.data[2],
             vec_a.data[0] * vec_b.data[1] - vec_a.data[1] * vec_b.data[0] };
}

template<typename T, int N>
__host__ __device__ T angle(const Vector<T, N>& vec_a,
                            const Vector<T, N>& vec_b) {
    return acos(cos_angle(vec_a, vec_b));
}
template<typename T, int N>
__host__ __device__ T cos_angle(const Vector<T, N>& vec_a,
                                const Vector<T, N>& vec_b) {
    return dot(vec_a, vec_b) / sqrt(vec_a.sq() * vec_b.sq());
}
template<typename T, int N>
__host__ __device__ T sin_angle(const Vector<T, N>& vec_a,
                                const Vector<T, N>& vec_b) {
    T d{ dot(vec_a, vec_b) };
    return sqrt(1 - d * d / vec_a.sq() / vec_b.sq());
}

template<typename T, int N>
__host__ __device__ T distance(const Vector<T, N>& vec_a,
                               const Vector<T, N>& vec_b);
template<typename T, int N>
__host__ __device__ T sq_distance(const Vector<T, N>& vec_a,
                                  const Vector<T, N>& vec_b);

template<typename T, int N>
__host__ __device__ void Print(const Vector<T, N>& vec) {
    Print("<");
    for (int i{ 0 }; i < N; ++i) { Print(" ", vec.data[i]); }
    Print(" >");
}

} // namespace eta

#endif

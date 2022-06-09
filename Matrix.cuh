#ifndef ETA_MATRIX_CUH
#define ETA_MATRIX_CUH

#include "Print.cuh"
#include "Vector.cuh"

namespace eta {

template<typename T, int M> struct DetHelper {};
template<typename T, int M> struct AdjHelper {};
template<typename T, int M> struct InvHelper {};

template<typename T, int M, int N> struct Matrix {
    static_assert(1 <= M && M <= 4 && 1 <= N && N <= 4,
                  "1 <= M && M <= 4 && 1 <= N && N <= 4");

    static constexpr int m{ M };
    static constexpr int n{ N };

    T data[M][N];

    ////////////////////////////////////////////////////////////////////////////

    __host__ __device__ static Matrix eye() {
        static_assert(M == N, "M == N");

        Matrix r;

        for (int i{ 0 }; i < M; ++i) {
            for (int j{ 0 }; j < N; ++j) { r.data[i][j] = (i == j); }
        }

        return r;
    }

    ////////////////////////////////////////////////////////////////////////////

    __host__ __device__ Matrix& operator+() { return *this; }
    __host__ __device__ const Matrix& operator+() const { return *this; }
    __host__ __device__ Matrix operator-() const {
        Matrix r;

        for (int i{ 0 }; i < M; ++i) {
            for (int j{ 0 }; j < N; ++j) { r.data[i][j] = -this->data[i][j]; }
        }

        return r;
    }

    ////////////////////////////////////////////////////////////////////////////

    __host__ __device__ T det() const { return DetHelper<T, M>::det(*this); }

    __host__ __device__ Matrix adj() const {
        return AdjHelper<T, M>::adj(*this);
    }

    __host__ __device__ Matrix inv() const {
        return InvHelper<T, M>::inv(*this);
    }
};

template<typename T, int M, int N>
__host__ __device__ bool operator==(const Matrix<T, M, N>& mat_a,
                                    const Matrix<T, M, N>& mat_b) {
    if (&mat_a == &mat_b) { return true; }

    for (int i{ 0 }; i < M; ++i) {
        for (int j{ 0 }; j < N; ++j) {
            if (ETA_gt_const(abs(mat_a.data[i][j] - mat_b.data[i][j]), 0)) {
                return false;
            }
        }
    }

    return true;
}
template<typename T, int M, int N>
__host__ __device__ bool operator!=(const Matrix<T, M, N>& mat_a,
                                    const Matrix<T, M, N>& mat_b) {
    return !(mat_a == mat_b);
}

template<typename T, int M, int N>
__host__ __device__ Matrix<T, M, N> operator+(const Matrix<T, M, N>& mat_a,
                                              const Matrix<T, M, N>& mat_b) {
    Matrix<T, M, N> r;

    for (int i{ 0 }; i < M; ++i) {
        for (int j{ 0 }; j < N; ++j) {
            r.data[i][j] = mat_a.data[i][j] + mat_b.data[i][j];
        }
    }

    return r;
}
template<typename T, int M, int N>
__host__ __device__ Matrix<T, M, N> operator-(const Matrix<T, M, N>& mat_a,
                                              const Matrix<T, M, N>& mat_b) {
    Matrix<T, M, N> r;

    for (int i{ 0 }; i < M; ++i) {
        for (int j{ 0 }; j < N; ++j) {
            r.data[i][j] = mat_a.data[i][j] - mat_b.data[i][j];
        }
    }

    return r;
}

template<typename T, int M, int N>
__host__ __device__ Matrix<T, M, N> operator*(const Matrix<T, M, N>& mat, T n) {
    Matrix<T, M, N> r;

    for (int i{ 0 }; i < M; ++i) {
        for (int j{ 0 }; j < N; ++j) { r.data[i][j] = mat.data[i][j] * n; }
    }

    return r;
}
template<typename T, int M, int N>
__host__ __device__ Matrix<T, M, N> operator/(const Matrix<T, M, N>& mat, T n) {
    Matrix<T, M, N> r;

    for (int i{ 0 }; i < M; ++i) {
        for (int j{ 0 }; j < N; ++j) { r.data[i][j] = mat.data[i][j] / n; }
    }

    return r;
}

template<typename T, int M, int N>
__host__ __device__ Matrix<T, M, N>& operator+=(Matrix<T, M, N>& mat_a,
                                                const Matrix<T, M, N>& mat_b) {
    for (int i{ 0 }; i < M; ++i) {
        for (int j{ 0 }; j < N; ++j) { mat_a.data[i][j] += mat_b.data[i][j]; }
    }

    return mat_a;
}
template<typename T, int M, int N>
__host__ __device__ Matrix<T, M, N>& operator-=(Matrix<T, M, N>& mat_a,
                                                const Matrix<T, M, N>& mat_b) {
    for (int i{ 0 }; i < M; ++i) {
        for (int j{ 0 }; j < N; ++j) { mat_a.data[i][j] -= mat_b.data[i][j]; }
    }

    return mat_a;
}

template<typename T, int M, int N>
__host__ __device__ Matrix<T, M, N>& operator*=(Matrix<T, M, N>& mat, T n) {
    for (int i{ 0 }; i < M; ++i) {
        for (int j{ 0 }; j < N; ++j) { mat.data[i][j] *= n; }
    }

    return mat;
}
template<typename T, int M, int N>
__host__ __device__ Matrix<T, M, N>& operator/=(Matrix<T, M, N>& mat, T n) {
    for (int i{ 0 }; i < M; ++i) {
        for (int j{ 0 }; j < N; ++j) { mat.data[i][j] /= n; }
    }

    return mat;
}

template<typename T, int M, int K, int N>
__host__ __device__ Matrix<T, M, N> operator*(const Matrix<T, M, K>& mat_a,
                                              const Matrix<T, K, N>& mat_b) {
    Matrix<T, M, N> r;

    for (int i{ 0 }; i < M; ++i) {
        for (int j{ 0 }; j < N; ++j) {
            r.data[i][j] = 0;

            for (int k{ 0 }; k < K; ++k) {
                r.data[i][j] += mat_a.data[i][k] * mat_b.data[k][j];
            }
        }
    }

    return r;
}

template<typename T, int M, int N>
__host__ __device__ Vector<T, M> operator*(const Matrix<T, M, N>& mat,
                                           const Vector<T, N>& vec) {
    Vector<T, M> r;

    for (int i{ 0 }; i < M; ++i) {
        r.data[i] = 0;

        for (int k{ 0 }; k < N; ++k) {
            r.data[i] += mat.data[i][k] * vec.data[k];
        }
    }

    return r;
}

////////////////////////////////////////////////////////////////////////////////

template<typename T> struct DetHelper<T, 1> {
    __host__ __device__ static T det(const Matrix<T, 1, 1>& mat) {
        return mat.data[0][0];
    }
};

template<typename T> struct DetHelper<T, 2> {
    __host__ __device__ static T det(const Matrix<T, 2, 2>& mat) {
        return mat.data[0][0] * mat.data[1][1] -
               mat.data[0][1] * mat.data[1][0];
    }
};

template<typename T> struct DetHelper<T, 3> {
    __host__ __device__ static T det(const Matrix<T, 3, 3>& mat) {
        return mat.data[0][0] * mat.data[1][1] * mat.data[2][2] -
               mat.data[0][0] * mat.data[1][2] * mat.data[2][1] -
               mat.data[0][1] * mat.data[1][0] * mat.data[2][2] +
               mat.data[0][1] * mat.data[1][2] * mat.data[2][0] +
               mat.data[0][2] * mat.data[1][0] * mat.data[2][1] -
               mat.data[0][2] * mat.data[1][1] * mat.data[2][0];
    }
};

template<typename T> struct DetHelper<T, 4> {
    __host__ __device__ static T det(const Matrix<T, 4, 4>& mat) {
        return mat.data[0][0] * mat.data[1][1] * mat.data[2][2] *
                   mat.data[3][3] -
               mat.data[0][0] * mat.data[1][1] * mat.data[2][3] *
                   mat.data[3][2] -
               mat.data[0][0] * mat.data[1][2] * mat.data[2][1] *
                   mat.data[3][3] +
               mat.data[0][0] * mat.data[1][2] * mat.data[2][3] *
                   mat.data[3][1] +
               mat.data[0][0] * mat.data[1][3] * mat.data[2][1] *
                   mat.data[3][2] -
               mat.data[0][0] * mat.data[1][3] * mat.data[2][2] *
                   mat.data[3][1] -
               mat.data[0][1] * mat.data[1][0] * mat.data[2][2] *
                   mat.data[3][3] +
               mat.data[0][1] * mat.data[1][0] * mat.data[2][3] *
                   mat.data[3][2] +
               mat.data[0][1] * mat.data[1][2] * mat.data[2][0] *
                   mat.data[3][3] -
               mat.data[0][1] * mat.data[1][2] * mat.data[2][3] *
                   mat.data[3][0] -
               mat.data[0][1] * mat.data[1][3] * mat.data[2][0] *
                   mat.data[3][2] +
               mat.data[0][1] * mat.data[1][3] * mat.data[2][2] *
                   mat.data[3][0] +
               mat.data[0][2] * mat.data[1][0] * mat.data[2][1] *
                   mat.data[3][3] -
               mat.data[0][2] * mat.data[1][0] * mat.data[2][3] *
                   mat.data[3][1] -
               mat.data[0][2] * mat.data[1][1] * mat.data[2][0] *
                   mat.data[3][3] +
               mat.data[0][2] * mat.data[1][1] * mat.data[2][3] *
                   mat.data[3][0] +
               mat.data[0][2] * mat.data[1][3] * mat.data[2][0] *
                   mat.data[3][1] -
               mat.data[0][2] * mat.data[1][3] * mat.data[2][1] *
                   mat.data[3][0] -
               mat.data[0][3] * mat.data[1][0] * mat.data[2][1] *
                   mat.data[3][2] +
               mat.data[0][3] * mat.data[1][0] * mat.data[2][2] *
                   mat.data[3][1] +
               mat.data[0][3] * mat.data[1][1] * mat.data[2][0] *
                   mat.data[3][2] -
               mat.data[0][3] * mat.data[1][1] * mat.data[2][2] *
                   mat.data[3][0] -
               mat.data[0][3] * mat.data[1][2] * mat.data[2][0] *
                   mat.data[3][1] +
               mat.data[0][3] * mat.data[1][2] * mat.data[2][1] *
                   mat.data[3][0];
    }
};

////////////////////////////////////////////////////////////////////////////////

template<typename T> struct AdjHelper<T, 1> {
    __host__ __device__ static Matrix<T, 1, 1> adj(const Matrix<T, 1, 1>& mat) {
        return { 1 };
    }
};

template<typename T> struct AdjHelper<T, 2> {
    __host__ __device__ static Matrix<T, 2, 2> adj(const Matrix<T, 2, 2>& mat) {
        return { mat.data[1][1], -mat.data[0][1], -mat.data[1][0],
                 mat.data[0][0] };
    }
};

template<typename T> struct AdjHelper<T, 3> {
    __host__ __device__ static Matrix<T, 3, 3> adj(const Matrix<T, 3, 3>& mat) {
        return {
            mat.data[1][1] * mat.data[2][2] - mat.data[1][2] * mat.data[2][1],
            -mat.data[0][1] * mat.data[2][2] + mat.data[0][2] * mat.data[2][1],
            mat.data[0][1] * mat.data[1][2] - mat.data[0][2] * mat.data[1][1],
            -mat.data[1][0] * mat.data[2][2] + mat.data[1][2] * mat.data[2][0],
            mat.data[0][0] * mat.data[2][2] - mat.data[0][2] * mat.data[2][0],
            -mat.data[0][0] * mat.data[1][2] + mat.data[0][2] * mat.data[1][0],
            mat.data[1][0] * mat.data[2][1] - mat.data[1][1] * mat.data[2][0],
            -mat.data[0][0] * mat.data[2][1] + mat.data[0][1] * mat.data[2][0],
            mat.data[0][0] * mat.data[1][1] - mat.data[0][1] * mat.data[1][0]
        };
    }
};

template<typename T> struct AdjHelper<T, 4> {
    __host__ __device__ static Matrix<T, 4, 4> adj(const Matrix<T, 4, 4>& mat) {
        return { mat.data[1][1] * mat.data[2][2] * mat.data[3][3] -
                     mat.data[1][1] * mat.data[2][3] * mat.data[3][2] -
                     mat.data[1][2] * mat.data[2][1] * mat.data[3][3] +
                     mat.data[1][2] * mat.data[2][3] * mat.data[3][1] +
                     mat.data[1][3] * mat.data[2][1] * mat.data[3][2] -
                     mat.data[1][3] * mat.data[2][2] * mat.data[3][1],
                 -mat.data[0][1] * mat.data[2][2] * mat.data[3][3] +
                     mat.data[0][1] * mat.data[2][3] * mat.data[3][2] +
                     mat.data[0][2] * mat.data[2][1] * mat.data[3][3] -
                     mat.data[0][2] * mat.data[2][3] * mat.data[3][1] -
                     mat.data[0][3] * mat.data[2][1] * mat.data[3][2] +
                     mat.data[0][3] * mat.data[2][2] * mat.data[3][1],
                 mat.data[0][1] * mat.data[1][2] * mat.data[3][3] -
                     mat.data[0][1] * mat.data[1][3] * mat.data[3][2] -
                     mat.data[0][2] * mat.data[1][1] * mat.data[3][3] +
                     mat.data[0][2] * mat.data[1][3] * mat.data[3][1] +
                     mat.data[0][3] * mat.data[1][1] * mat.data[3][2] -
                     mat.data[0][3] * mat.data[1][2] * mat.data[3][1],
                 -mat.data[0][1] * mat.data[1][2] * mat.data[2][3] +
                     mat.data[0][1] * mat.data[1][3] * mat.data[2][2] +
                     mat.data[0][2] * mat.data[1][1] * mat.data[2][3] -
                     mat.data[0][2] * mat.data[1][3] * mat.data[2][1] -
                     mat.data[0][3] * mat.data[1][1] * mat.data[2][2] +
                     mat.data[0][3] * mat.data[1][2] * mat.data[2][1],
                 -mat.data[1][0] * mat.data[2][2] * mat.data[3][3] +
                     mat.data[1][0] * mat.data[2][3] * mat.data[3][2] +
                     mat.data[1][2] * mat.data[2][0] * mat.data[3][3] -
                     mat.data[1][2] * mat.data[2][3] * mat.data[3][0] -
                     mat.data[1][3] * mat.data[2][0] * mat.data[3][2] +
                     mat.data[1][3] * mat.data[2][2] * mat.data[3][0],
                 mat.data[0][0] * mat.data[2][2] * mat.data[3][3] -
                     mat.data[0][0] * mat.data[2][3] * mat.data[3][2] -
                     mat.data[0][2] * mat.data[2][0] * mat.data[3][3] +
                     mat.data[0][2] * mat.data[2][3] * mat.data[3][0] +
                     mat.data[0][3] * mat.data[2][0] * mat.data[3][2] -
                     mat.data[0][3] * mat.data[2][2] * mat.data[3][0],
                 -mat.data[0][0] * mat.data[1][2] * mat.data[3][3] +
                     mat.data[0][0] * mat.data[1][3] * mat.data[3][2] +
                     mat.data[0][2] * mat.data[1][0] * mat.data[3][3] -
                     mat.data[0][2] * mat.data[1][3] * mat.data[3][0] -
                     mat.data[0][3] * mat.data[1][0] * mat.data[3][2] +
                     mat.data[0][3] * mat.data[1][2] * mat.data[3][0],
                 mat.data[0][0] * mat.data[1][2] * mat.data[2][3] -
                     mat.data[0][0] * mat.data[1][3] * mat.data[2][2] -
                     mat.data[0][2] * mat.data[1][0] * mat.data[2][3] +
                     mat.data[0][2] * mat.data[1][3] * mat.data[2][0] +
                     mat.data[0][3] * mat.data[1][0] * mat.data[2][2] -
                     mat.data[0][3] * mat.data[1][2] * mat.data[2][0],
                 mat.data[1][0] * mat.data[2][1] * mat.data[3][3] -
                     mat.data[1][0] * mat.data[2][3] * mat.data[3][1] -
                     mat.data[1][1] * mat.data[2][0] * mat.data[3][3] +
                     mat.data[1][1] * mat.data[2][3] * mat.data[3][0] +
                     mat.data[1][3] * mat.data[2][0] * mat.data[3][1] -
                     mat.data[1][3] * mat.data[2][1] * mat.data[3][0],
                 -mat.data[0][0] * mat.data[2][1] * mat.data[3][3] +
                     mat.data[0][0] * mat.data[2][3] * mat.data[3][1] +
                     mat.data[0][1] * mat.data[2][0] * mat.data[3][3] -
                     mat.data[0][1] * mat.data[2][3] * mat.data[3][0] -
                     mat.data[0][3] * mat.data[2][0] * mat.data[3][1] +
                     mat.data[0][3] * mat.data[2][1] * mat.data[3][0],
                 mat.data[0][0] * mat.data[1][1] * mat.data[3][3] -
                     mat.data[0][0] * mat.data[1][3] * mat.data[3][1] -
                     mat.data[0][1] * mat.data[1][0] * mat.data[3][3] +
                     mat.data[0][1] * mat.data[1][3] * mat.data[3][0] +
                     mat.data[0][3] * mat.data[1][0] * mat.data[3][1] -
                     mat.data[0][3] * mat.data[1][1] * mat.data[3][0],
                 -mat.data[0][0] * mat.data[1][1] * mat.data[2][3] +
                     mat.data[0][0] * mat.data[1][3] * mat.data[2][1] +
                     mat.data[0][1] * mat.data[1][0] * mat.data[2][3] -
                     mat.data[0][1] * mat.data[1][3] * mat.data[2][0] -
                     mat.data[0][3] * mat.data[1][0] * mat.data[2][1] +
                     mat.data[0][3] * mat.data[1][1] * mat.data[2][0],
                 -mat.data[1][0] * mat.data[2][1] * mat.data[3][2] +
                     mat.data[1][0] * mat.data[2][2] * mat.data[3][1] +
                     mat.data[1][1] * mat.data[2][0] * mat.data[3][2] -
                     mat.data[1][1] * mat.data[2][2] * mat.data[3][0] -
                     mat.data[1][2] * mat.data[2][0] * mat.data[3][1] +
                     mat.data[1][2] * mat.data[2][1] * mat.data[3][0],
                 mat.data[0][0] * mat.data[2][1] * mat.data[3][2] -
                     mat.data[0][0] * mat.data[2][2] * mat.data[3][1] -
                     mat.data[0][1] * mat.data[2][0] * mat.data[3][2] +
                     mat.data[0][1] * mat.data[2][2] * mat.data[3][0] +
                     mat.data[0][2] * mat.data[2][0] * mat.data[3][1] -
                     mat.data[0][2] * mat.data[2][1] * mat.data[3][0],
                 -mat.data[0][0] * mat.data[1][1] * mat.data[3][2] +
                     mat.data[0][0] * mat.data[1][2] * mat.data[3][1] +
                     mat.data[0][1] * mat.data[1][0] * mat.data[3][2] -
                     mat.data[0][1] * mat.data[1][2] * mat.data[3][0] -
                     mat.data[0][2] * mat.data[1][0] * mat.data[3][1] +
                     mat.data[0][2] * mat.data[1][1] * mat.data[3][0],
                 mat.data[0][0] * mat.data[1][1] * mat.data[2][2] -
                     mat.data[0][0] * mat.data[1][2] * mat.data[2][1] -
                     mat.data[0][1] * mat.data[1][0] * mat.data[2][2] +
                     mat.data[0][1] * mat.data[1][2] * mat.data[2][0] +
                     mat.data[0][2] * mat.data[1][0] * mat.data[2][1] -
                     mat.data[0][2] * mat.data[1][1] * mat.data[2][0] };
    }
};

////////////////////////////////////////////////////////////////////////////////

template<typename T> struct InvHelper<T, 1> {
    __host__ __device__ static Matrix<T, 1, 1> inv(const Matrix<T, 1, 1>& mat) {
        return { 1 / mat.data[0][0] };
    }
};

template<typename T> struct InvHelper<T, 2> {
    __host__ __device__ static Matrix<T, 2, 2> inv(const Matrix<T, 2, 2>& mat) {
        float rdet{ 1 / mat.det() };
        return { mat.data[1][1] * rdet, -mat.data[0][1] * rdet,
                 -mat.data[1][0] * rdet, mat.data[0][0] * rdet };
    }
};

template<typename T> struct InvHelper<T, 3> {
    __host__ __device__ static Matrix<T, 3, 3> inv(const Matrix<T, 3, 3>& mat) {
        Matrix<T, 3, 3> r{ mat.adj() };
        return r /
               (mat.data[0][0] * r.data[0][0] + mat.data[1][0] * r.data[0][1] +
                mat.data[2][0] * r.data[0][2]);
    }
};

template<typename T> struct InvHelper<T, 4> {
    __host__ __device__ static Matrix<T, 4, 4> inv(const Matrix<T, 4, 4>& mat) {
        Matrix<T, 4, 4> r{ mat.adj() };
        return r /
               (mat.data[0][0] * r.data[0][0] + mat.data[1][0] * r.data[0][1] +
                mat.data[2][0] * r.data[0][2] + mat.data[3][0] * r.data[0][3]);
    }
};

////////////////////////////////////////////////////////////////////////////////

template<typename T, int M, int N>
__host__ __device__ void Print(const Matrix<T, M, N>& mat) {
    Print("<");

    for (int j{ 0 }; j < N; ++j) { Print(" ", mat.data[0][j]); }

    for (int i{ 1 }; i < M; ++i) {
        Print("\n ");

        for (int j{ 0 }; j < N; ++j) { Print(" ", mat.data[i][j]); }
    }

    Print(" >\n");
}

} // namespace eta

#endif
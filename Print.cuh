#ifndef ETA_PRINT_CUH
#define ETA_PRINT_CUH

#include "define.cuh"

#include <vector>

namespace eta {

////////////////////////////////////////////////////////////////////////////////

__host__ __device__ void Print(const char* str) { printf("%s", str); }
__host__ __device__ void Print(bool x) { printf(x ? " true" : "false"); }

__host__ __device__ void Print(char x) { printf("%c", x); }
__host__ __device__ void Print(unsigned char x) { printf("%c", x); }

__host__ __device__ void Print(int x) { printf("%d", x); }
__host__ __device__ void Print(unsigned int x) { printf("%u", x); }

__host__ __device__ void Print(short x) { printf("%hd", x); }
__host__ __device__ void Print(unsigned short x) { printf("%hu", x); }

__host__ __device__ void Print(long x) { printf("%ld", x); }
__host__ __device__ void Print(unsigned long x) { printf("%lu", x); }

__host__ __device__ void Print(long long x) { printf("%lld", x); }
__host__ __device__ void Print(unsigned long long x) { printf("%llu", x); }

__host__ __device__ void Print(float x) { printf("%12.5e", x); }

__host__ __device__ void Print(double x) { printf("%12.5e", x); }

template<typename T> __host__ void Print(const std::vector<T>& x) {
    if (x.empty()) {
        Print("[]");
    } else {
        Print("[ ");
        Print(x.front());

        for (size_t i{ 1 }; i < x.size(); ++i) {
            Print(", ");
            Print(x[i]);
        }

        Print(" ]");
    }
}

__host__ __device__ void Print() {}

template<typename X, typename Y, typename... Args>
__host__ __device__ void Print(X&& x, Y&& y, Args&&... args) {
    Print(std::forward<X>(x));
    Print(std::forward<Y>(y), std::forward<Args>(args)...);
}

} // namespace eta

#endif
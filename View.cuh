#ifndef ETA_VIEW_CUH
#define ETA_VIEW_CUH

#include "define.cuh"

namespace eta {

struct View {
    void* base{ nullptr };
    int coeff[3]{ 0 };

    template<typename... Args>
    __host__ __device__ int get_index(Args... indices) const {
        return this->get_index<sizeof...(indices)>(indices...);
    }

    template<int dim> __host__ __device__ int get_index() const { return 0; }

    template<int dim, typename... Args>
    __host__ __device__ int get_index(int index, Args... indices) const {
        return index * this->coeff[dim - sizeof...(indices) - 1] +
               this->get_index<dim>(indices...);
    }

    template<typename T, typename... Args>
    __host__ __device__ T& get(Args... indices) {
        return *reinterpret_cast<T*>(static_cast<char*>(this->base) +
                                     this->get_index(indices...));
    }

    template<typename T, typename... Args>
    __host__ __device__ const T& get(Args... indices) const {
        return *reinterpret_cast<T*>(static_cast<char*>(this->base) +
                                     this->get_index(indices...));
    }
};

} // namespace eta

#endif

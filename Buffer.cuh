#ifndef ETA_BUFFER_CUH
#define ETA_BUFFER_CUH

#include "define.cuh"

namespace eta {

template<typename D> class Buffer {
public:
    static_assert(std::is_same<D, CPU>::value || std::is_same<D, GPU>::value);

    template<typename D_> friend class Buffer;

    using device = D;

    /* return current size of data */
    __host__ __device__ int size() const { return this->size_; }

    /* return current capacity of data */
    __host__ __device__ int capacity() const { return this->capacity_; }

    /* return current data */
    template<typename T = void> __host__ __device__ T* data() const {
        return static_cast<T*>(this->data_);
    }

    Buffer(int size = 0): size_{ size }, capacity_{ size } {
        this->data_ = Malloc<D>(size);
    }

    Buffer(const Buffer& buffer) = delete;

    Buffer(Buffer&& buffer):
        size_{ buffer.size_ }, capacity_{ buffer.capacity_ }, data_{
            buffer.data_
        } {
        buffer.size_ = 0;
        buffer.capacity_ = 0;
        buffer.data_ = nullptr;
    }

    ~Buffer() { Free<D>(this->data_); }

    /*
     * insure size of data at least size bytes, if not remalloc
     * return data
    */
    __host__ void* Resize(int size) {
        this->size_ = size;

        if (this->capacity_ < size) {
            this->capacity_ = size;

            Free<D>(this->data_);
            this->data_ = Malloc<D>(this->size_);
        }

        return this->data_;
    }

    /*
     * insure size of data at least size bytes, if not, remalloc
     * if remalloc, move data to new, synchronized
     * return data
    */
    __host__ void* ResizeMove(int size) {
        if (this->capacity_ < size) {
            this->capacity_ = size;

            void* data{ Malloc<D>(size) };
            MemcpyAsync<D, D>(data, this->data_, this->size_);
            this->data_ = data;
        }

        this->size_ = size;

        return this->data_;
    }

    __host__ void clear() {
        this->size_ = 0;
        this->capacity_ = 0;

        Free<D>(this->data_);
        this->data_ = nullptr;
    }

    __host__ void Clear() {
        this->size_ = 0;
        this->capacity_ = 0;

        Free<D>(this->data_);
        this->data_ = nullptr;
    }

    template<typename D_>
    __host__ void PassTo(Buffer<D_>& buffer, int size,
                         cudaStream_t stream) const {
        if (this->size_ < size) { ETA_throw("this->size_ < size\n"); }

        buffer.Resize(size);

        MemcpyAsync<D_, D>(buffer.data_, // dst
                           this->data_, // src
                           size, // size
                           stream // stream
        );
    }

private:
    int size_;
    int capacity_;
    void* data_;
};

} // namespace eta

#endif

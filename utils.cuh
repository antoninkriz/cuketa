#include <cstdint>
#include <stdexcept>
#include <iostream>

typedef int32_t i32;

#define CHECK_CUDA_ERROR(call) { \
    auto result = (call);        \
    if (result != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(result) << std::endl; \
        throw std::runtime_error("CUDA error");                                                                          \
    }                            \
}

template<typename T>
inline T *cuda_malloc(i32 size) {
    T *ptr;
    CHECK_CUDA_ERROR(cudaMalloc(&ptr, size * sizeof(T)));
    return ptr;
}

inline void cuda_free(void *ptr) {
    CHECK_CUDA_ERROR(cudaFree(ptr));
}

template<typename T>
class cuda_arr {
    const i32 n;
    T *ptr;
public:
    explicit cuda_arr(i32 n) : ptr(cuda_malloc<T>(n)), n(n) {}

    i32 size() const {
        return n;
    }

    T operator[](i32 i) const {
        return ptr[i];
    }

    T &operator[](i32 i) {
        return ptr[i];
    }

    T *begin() const {
        return ptr;
    }

    T *end() const {
        return ptr + n;
    }

    T *get() const {
        return ptr;
    }

    void free() {
        cuda_free(ptr);
    }
};

template<typename T>
inline void cuda_memcpy(cuda_arr<T> &dst, const T *src, i32 size) {
    CHECK_CUDA_ERROR(cudaMemcpyAsync(dst.get(), src, size * sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice))
}

template<typename T>
inline void cuda_memcpy(cuda_arr<T> &dst, i32 dst_offset, const T *src, i32 src_offset, i32 size) {
    CHECK_CUDA_ERROR(cudaMemcpyAsync(dst.get() + dst_offset, src + src_offset, size * sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice))
}

template<typename T>
inline void cuda_memcpy(cuda_arr<T> &dst, const cuda_arr<T> &src, i32 size) {
    CHECK_CUDA_ERROR(cudaMemcpyAsync(dst.get(), src.get(), size * sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToDevice))
}

template<typename T>
inline void cuda_memcpy(T *dst, const cuda_arr<T> &src, i32 size) {
    CHECK_CUDA_ERROR(cudaMemcpyAsync(dst, src.get(), size * sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost))
}

template<typename T>
inline void cuda_memset(cuda_arr<T> &dst, T val) {
    CHECK_CUDA_ERROR(cudaMemsetAsync(dst.get(), val, dst.size() * sizeof(T)))
}

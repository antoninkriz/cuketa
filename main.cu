#include "utils.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <vector>
#include <thrust/sort.h>

#ifdef _BLOCK_SIZE_INIT
constexpr i32 BLOCK_SIZE_INIT_FIT = _BLOCK_SIZE_INIT;
constexpr i32 BLOCK_SIZE_INIT_QUANTILES = _BLOCK_SIZE_INIT;
constexpr i32 BLOCK_SIZE_INIT_TRANSFORM = _BLOCK_SIZE_INIT;
#else
constexpr i32 BLOCK_SIZE_INIT_FIT = 512;
constexpr i32 BLOCK_SIZE_INIT_QUANTILES = 512;
constexpr i32 BLOCK_SIZE_INIT_TRANSFORM = 512;
#endif

#ifdef _BLOCK_SIZE_C_XXX
constexpr i32 BLOCK_SIZE_C_A = _BLOCK_SIZE_C_XXX;
constexpr i32 BLOCK_SIZE_C_G = _BLOCK_SIZE_C_XXX;
#else
constexpr i32 BLOCK_SIZE_C_A = 256;
constexpr i32 BLOCK_SIZE_C_G = 256;
#endif

#ifdef _BLOCK_SIZE_QUANTILE_BIASES
constexpr i32 BLOCK_SIZE_QUANTILE_BIASES = _BLOCK_SIZE_QUANTILE_BIASES;
#else
constexpr i32 BLOCK_SIZE_QUANTILE_BIASES = 512;
#endif

#ifdef  _BLOCK_SIZE_PPV_PX
constexpr i32 BLOCK_SIZE_PPV = _BLOCK_SIZE_PPV_PX;
#else
constexpr i32 BLOCK_SIZE_PPV = 256;
#endif

#define BLOCK_SIZE(x, b) ((x + b - 1) / b), b

constexpr i32 NUM_KERNELS = 84;
constexpr i32 INDICES_COUNT = 3;

constexpr i32 INDICES[NUM_KERNELS][INDICES_COUNT] = {
        {1, 2, 3},
        {1, 2, 4},
        {1, 2, 5},
        {1, 2, 6},
        {1, 2, 7},
        {1, 2, 8},
        {1, 2, 9},
        {1, 3, 4},
        {1, 3, 5},
        {1, 3, 6},
        {1, 3, 7},
        {1, 3, 8},
        {1, 3, 9},
        {1, 4, 5},
        {1, 4, 6},
        {1, 4, 7},
        {1, 4, 8},
        {1, 4, 9},
        {1, 5, 6},
        {1, 5, 7},
        {1, 5, 8},
        {1, 5, 9},
        {1, 6, 7},
        {1, 6, 8},
        {1, 6, 9},
        {1, 7, 8},
        {1, 7, 9},
        {1, 8, 9},
        {2, 3, 4},
        {2, 3, 5},
        {2, 3, 6},
        {2, 3, 7},
        {2, 3, 8},
        {2, 3, 9},
        {2, 4, 5},
        {2, 4, 6},
        {2, 4, 7},
        {2, 4, 8},
        {2, 4, 9},
        {2, 5, 6},
        {2, 5, 7},
        {2, 5, 8},
        {2, 5, 9},
        {2, 6, 7},
        {2, 6, 8},
        {2, 6, 9},
        {2, 7, 8},
        {2, 7, 9},
        {2, 8, 9},
        {3, 4, 5},
        {3, 4, 6},
        {3, 4, 7},
        {3, 4, 8},
        {3, 4, 9},
        {3, 5, 6},
        {3, 5, 7},
        {3, 5, 8},
        {3, 5, 9},
        {3, 6, 7},
        {3, 6, 8},
        {3, 6, 9},
        {3, 7, 8},
        {3, 7, 9},
        {3, 8, 9},
        {4, 5, 6},
        {4, 5, 7},
        {4, 5, 8},
        {4, 5, 9},
        {4, 6, 7},
        {4, 6, 8},
        {4, 6, 9},
        {4, 7, 8},
        {4, 7, 9},
        {4, 8, 9},
        {5, 6, 7},
        {5, 6, 8},
        {5, 6, 9},
        {5, 7, 8},
        {5, 7, 9},
        {5, 8, 9},
        {6, 7, 8},
        {6, 7, 9},
        {6, 8, 9},
        {7, 8, 9}
};

__global__ void init_fit(const float *X, i32 x_id, float *A, float *G, float *C, i32 input_length) {
    i32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= input_length)
        return;

    A[i] = X[x_id + i] * -1;
    G[i] = X[x_id + i] * 3;
    C[i] = A[i];
}

__global__ void init_transform(const float *X, i32 x_id, float *A, float *G, i32 input_length) {
    i32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= input_length)
        return;

    A[i] = X[x_id + i] * -1;
    G[i] = X[x_id + i] * 3;
}

__global__ void C_A_add(float *C, const float *A, i32 input_length, i32 dilation, i32 s, i32 e) {
    i32 i = blockIdx.x * blockDim.x + threadIdx.x;

    float tmpIn = i < input_length ? A[i] : 0;
    for (i32 g = 1; g <= 9 / 2; g++) {
        i32 d = e + dilation * (g - 1);
        if (i < d)
            atomicAdd(&C[input_length - d + i], tmpIn);
    }

    float tmpOut = 0;
    for (i32 g = (9 / 2) + 2; g <= 9; ++g) {
        i32 d = s + dilation * (g - ((9 / 2) + 2));
        if (i < input_length - d)
            tmpOut += A[i + d];
    }
    atomicAdd(&C[i], tmpOut);
}

__global__ void
C_G(float *C, const float *G, i32 input_length, i32 dilation, i32 s, i32 e, i32 ind0, i32 ind1, i32 ind2) {
    i32 i = blockIdx.x * blockDim.x + threadIdx.x;

    float gi = i < input_length ? G[i] : 0;

    if (ind0 < 5) {
        i32 d = e + dilation * (ind0 - 1);
        if (i < d)
            atomicAdd(&C[input_length - d + i], gi);
    } else if (ind0 > 5) {
        i32 d = s + dilation * (ind0 - (9 / 2 + 2));
        if (i < input_length - d)
            atomicAdd(&C[i], G[d + i]);
    } else {
        if (i < input_length)
            atomicAdd(&C[i], gi);
    }

    if (ind1 < 5) {
        i32 d = e + dilation * (ind1 - 1);
        if (i < d)
            atomicAdd(&C[input_length - d + i], gi);
    } else if (ind1 > 5) {
        i32 d = s + dilation * (ind1 - (9 / 2 + 2));
        if (i < input_length - d)
            atomicAdd(&C[i], G[d + i]);
    } else {
        if (i < input_length)
            atomicAdd(&C[i], gi);
    }

    if (ind2 < 5) {
        i32 d = e + dilation * (ind2 - 1);
        if (i < d)
            atomicAdd(&C[input_length - d + i], gi);
    } else if (ind2 > 5) {
        i32 d = s + dilation * (ind2 - (9 / 2 + 2));
        if (i < input_length - d)
            atomicAdd(&C[i], G[d + i]);
    } else {
        if (i < input_length)
            atomicAdd(&C[i], gi);
    }
}

__global__ void init_quantiles(
        float *quantiles,
        i32 length
) {
    i32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) return;

    quantiles[i] = fmodf((i + 1) * ((sqrtf(5.0f) + 1.0f) / 2.0f), 1.0f);
}

__global__ void quantile_biases(
        float *biases,
        const float *C,
        const float *quantiles,
        i32 input_length,
        i32 feature_index_start,
        i32 feature_index_end
) {
    i32 i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= feature_index_end - feature_index_start) return;

    float index = (input_length - 1) * quantiles[feature_index_start + i];
    i32 lowerIndex = static_cast<i32>(index);
    i32 upperIndex = min(lowerIndex + 1, input_length - 1);
    float fraction = index - lowerIndex;

    biases[feature_index_start + i] = C[lowerIndex] + fraction * (C[upperIndex] - C[lowerIndex]);
}

cuda_arr<float> fit_biases(
        const cuda_arr<float> &X,
        i32 num_examples,
        i32 input_length,
        const std::vector<i32> &dilations,
        const std::vector<i32> &num_features_per_dilation,
        const cuda_arr<float> quantiles
) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, num_examples - 1);

    i32 feature_index_start = 0;

    cuda_arr<float> biases(quantiles.size());
    cuda_arr<float> A(input_length), G(input_length), C(input_length);

    for (i32 dilation_index = 0; dilation_index < dilations.size(); dilation_index++) {
        i32 dilation = dilations[dilation_index];
        i32 padding = ((9 - 1) * dilation) / 2;
        i32 num_features_this_dilation = num_features_per_dilation[dilation_index];

        for (i32 kernel_index = 0; kernel_index < NUM_KERNELS; kernel_index++) {
            i32 feature_index_end = feature_index_start + num_features_this_dilation;

            i32 random_idx = distrib(gen);

            init_fit<<<BLOCK_SIZE(input_length, BLOCK_SIZE_INIT_FIT)>>>(
                    X.get(), random_idx * input_length, A.get(), G.get(), C.get(), input_length
            );

            i32 s = dilation;
            i32 e = input_length - padding;

            C_A_add<<<BLOCK_SIZE(input_length, BLOCK_SIZE_C_A)>>>(C.get(), A.get(), input_length, dilation, s, e);

            const auto &[ind0, ind1, ind2] = INDICES[kernel_index];

            C_G<<<BLOCK_SIZE(input_length, BLOCK_SIZE_C_G)>>>(
                    C.get(), G.get(), input_length, dilation, s, e, ind0, ind1, ind2
            );

            thrust::sort(thrust::cuda::par_nosync, C.begin(), C.end());

            quantile_biases<<<BLOCK_SIZE(feature_index_end - feature_index_start, BLOCK_SIZE_QUANTILE_BIASES)>>>(
                    biases.get(),
                    C.get(),
                    quantiles.get(),
                    input_length,
                    feature_index_start,
                    feature_index_end
            );

            feature_index_start = feature_index_end;
        }
    }

    A.free();
    G.free();
    C.free();

    return biases;
}

template<typename T>
std::vector<i32> logspace_floored(T base, T start, T stop, i32 n) {
    auto arr = std::vector<i32>(n);
    T step = (stop - start) / (n - 1);
    for (i32 i = 0; i < n; i++) {
        T exponent = start + i * step;
        arr[i] = pow(base, exponent);
    }

    return arr;
}

typedef struct {
    std::vector<i32> unique_elements;
    std::vector<i32> counts;
} unique_counts_result;

unique_counts_result sorted_unique_counts(const std::vector<i32> &arr) {
    i32 uniq_count = 1;
    for (i32 i = 1; i < arr.size(); ++i) {
        if (arr[i] != arr[i - 1]) {
            uniq_count++;
        }
    }

    unique_counts_result result{
            std::vector<i32>(uniq_count),
            std::vector<i32>(uniq_count)
    };

    i32 pos = 0;
    result.unique_elements[0] = arr[0];
    result.counts[0] = 1;

    for (i32 i = 1; i < arr.size(); ++i) {
        if (arr[i] == arr[i - 1]) {
            result.counts[pos]++;
        } else {
            pos++;
            result.unique_elements[pos] = arr[i];
            result.counts[pos] = 1;
        }
    }

    return result;
}

typedef struct {
    std::vector<i32> dilations;
    std::vector<i32> num_features_per_dilation;
} dilations_result;

dilations_result fit_dilations(
        i32 input_length,
        i32 num_features,
        i32 max_dilations_per_kernel
) {
    i32 num_features_per_kernel = num_features / NUM_KERNELS;
    i32 true_max_dilations_per_kernel = std::min(num_features_per_kernel, max_dilations_per_kernel);
    float multiplier = (float) num_features_per_kernel / true_max_dilations_per_kernel;

    float max_exponent = log2f((input_length - 1) / (9.0f - 1.0f));

    auto logspace = logspace_floored(2.0f, 0.f, max_exponent, true_max_dilations_per_kernel);
    auto unique_counts = sorted_unique_counts(logspace);

    dilations_result result = {
            unique_counts.unique_elements,
            unique_counts.counts
    };

    for (auto &x: result.num_features_per_dilation)
        x *= multiplier;

    i32 num_features_per_dilation_sum = 0;
    for (const auto &x: result.num_features_per_dilation)
        num_features_per_dilation_sum += x;

    i32 remainder = num_features_per_kernel - num_features_per_dilation_sum;

    i32 i = 1;
    while (remainder > 0) {
        result.num_features_per_dilation[i - 1] += 1;
        i = (i % result.num_features_per_dilation.size()) + 1;
        remainder -= 1;
    }

    return result;
}

typedef struct {
    std::vector<i32> dilations;
    std::vector<i32> num_features_per_dilation;
    cuda_arr<float> biases;
} fit_result;

fit_result fit(
        const cuda_arr<float> &X,
        i32 num_examples,
        i32 input_length,
        i32 num_features = 10000,
        i32 max_dilations_per_kernel = 32
) {
    auto [dilations, num_features_per_dilation] = fit_dilations(
            input_length,
            num_features,
            max_dilations_per_kernel
    );

    i32 num_features_per_kernel = 0;
    for (const auto &x: num_features_per_dilation)
        num_features_per_kernel += x;

    auto true_num_features = NUM_KERNELS * num_features_per_kernel;

    auto quantiles = cuda_arr<float>(true_num_features);
    init_quantiles<<<BLOCK_SIZE(true_num_features, BLOCK_SIZE_INIT_QUANTILES)>>>(quantiles.get(), true_num_features);

    auto biases = fit_biases(X, num_examples, input_length, dilations, num_features_per_dilation, quantiles);

    quantiles.free();

    return {
            dilations,
            num_features_per_dilation,
            biases
    };
}

template<unsigned int blockSize>
__device__ inline constexpr void warp_reduce(volatile float *sdata, int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

__global__ void ppv(
        float *features,
        i32 item_idx,
        i32 num_examples,
        i32 feature_index_start,
        i32 num_features_this_dilation,
        const float *C,
        i32 input_length,
        i32 padding,
        const float *biases
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float bias;
    __shared__ float shared_features[BLOCK_SIZE_PPV];

    int valid_input_length = input_length - 2 * padding;

    for (int feature_count = 0; feature_count < num_features_this_dilation; feature_count++) {
        bias = biases[feature_index_start + feature_count];
        __syncthreads();

        if (i < valid_input_length)
            shared_features[threadIdx.x] = C[padding + i] > bias
                                           ? 1.0f
                                           : 0.0f;
        else
            shared_features[threadIdx.x] = 0.0f;

        __syncthreads();

        // Perform parallel reduction within the block
        for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
            if (threadIdx.x < stride)
                shared_features[threadIdx.x] += shared_features[threadIdx.x + stride];
            __syncthreads();
        }

        if (threadIdx.x < 32)
            warp_reduce<BLOCK_SIZE_PPV>(shared_features, threadIdx.x);

        if (threadIdx.x == 0)
            atomicAdd(
                    &features[(item_idx * num_examples) + feature_index_start + feature_count],
                    shared_features[0] / valid_input_length
            );

        __syncthreads();
    }
}

cuda_arr<float> transform(
        const cuda_arr<float> &X,
        i32 num_examples,
        i32 input_length,
        const std::vector<i32> &dilations,
        const std::vector<i32> &num_features_per_dilation,
        const cuda_arr<float> biases
) {
    auto features = cuda_arr<float>(num_examples * biases.size());
    cuda_memset(features, 0.f);

    auto C_alpha = cuda_arr<float>(input_length);
    auto C = cuda_arr<float>(input_length);
    auto A = cuda_arr<float>(input_length);
    auto G = cuda_arr<float>(input_length);

    for (i32 item_idx = 0; item_idx < num_examples; item_idx++) {
        init_transform<<<BLOCK_SIZE(input_length, BLOCK_SIZE_INIT_TRANSFORM)>>>(
                X.get(),
                item_idx * input_length,
                A.get(),
                G.get(),
                input_length
        );

        i32 feature_index_start = 0;
        for (i32 dilation_index = 0; dilation_index < dilations.size(); dilation_index++) {
            i32 dilation = dilations[dilation_index];
            i32 padding = ((9 - 1) * dilation) / 2;
            i32 num_features_this_dilation = num_features_per_dilation[dilation_index];

            cuda_memcpy(C_alpha, A, input_length);

            i32 s = dilation;
            i32 e = input_length - padding;

            C_A_add<<<BLOCK_SIZE(input_length, BLOCK_SIZE_C_A)>>>(C_alpha.get(), A.get(), input_length, dilation, s, e);

            i32 _padding0 = dilation_index % 2;
            for (i32 kernel_index = 0; kernel_index < NUM_KERNELS; kernel_index++) {
                i32 feature_index_end = feature_index_start + num_features_this_dilation;

                cuda_memcpy(C, C_alpha, input_length);

                const auto &[ind0, ind1, ind2] = INDICES[kernel_index];

                C_G<<<BLOCK_SIZE(input_length, BLOCK_SIZE_C_G)>>>(C.get(), G.get(), input_length, dilation, s, e, ind0,
                                                                  ind1, ind2);

                i32 _padding1 = (_padding0 + kernel_index) % 2;
                ppv<<<BLOCK_SIZE(input_length, BLOCK_SIZE_PPV)>>>(
                        features.get(),
                        item_idx,
                        num_examples,
                        feature_index_start,
                        num_features_this_dilation,
                        C.get(),
                        input_length,
                        _padding1 ? padding : 0,
                        biases.get()
                );

                feature_index_start = feature_index_end;
            }
        }
    }

    C_alpha.free();
    C.free();
    A.free();
    G.free();

    return features;
}

i32 main() {
    constexpr i32 Ns[][2] = {
            // Test block sizes
            {10, 100000},
//            // 1851 ms
//            {100, 10000},
//            // 3274 ms
//            {100, 100000},
//            // 27515 ms, 558 MB VRAM, 456 MB RAM
//            {100, 1000000},
    };

    for (const auto &[num_examples, input_length]: Ns) {
        auto rng = std::mt19937(1337);
        auto X0 = new float[num_examples * input_length];
        for (i32 i = 0; i < num_examples * input_length; i++)
            X0[i] = std::uniform_real_distribution<float>(-1, 1)(rng);

        auto start = std::chrono::high_resolution_clock::now();

        auto X = cuda_arr<float>(num_examples * input_length);
        cuda_memcpy(X, X0, num_examples * input_length);

        auto [dilations, num_features_per_dilation, biases] = fit(X, num_examples, input_length);
        auto transform_res = transform(X, num_examples, input_length, dilations, num_features_per_dilation, biases);

        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << duration << " ms" << std::endl;

        X.free();
        transform_res.free();
    }
    return 0;
}

#include <iostream>
#include <cassert>
#include <vector>

#include <chrono>

template<typename T>
__global__ void vectorAdd(const T* A, const T* B, T* C, size_t N) {
    size_t index = threadIdx.x;

    if (index < N) {
        C[index] = A[index] + B[index];
    }
}

template<typename T>
__global__ void matrixAdd(const T* A, const T* B, T* C, size_t M, size_t N) {
    // TODO: Is the type of A, B correct?
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    size_t col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        C[row * N + col] = A[row * N + col] + B[row * N + col];
    }
}

template<typename T>
__global__ void matrixMul(const T* A, const T* B, T* C, size_t M, size_t K, size_t N) {
    // TODO: Is the type of A, B correct?
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    size_t col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        T ret = 0;
        for (size_t i = 0; i < K; ++i) {
            ret += A[row * K + i] * B[i * row + col];
        }
        C[row * N + col] = ret;
    }
}

template<typename T>
std::vector<T> vectorAddUsingGPU(const std::vector<T>& a, const std::vector<T>& b) {
    auto h_A = a.data();
    auto h_B = b.data();

    size_t N = a.size();
    assert(N == b.size());
    size_t size = N * sizeof(T);
    T* h_C = new T[size];

    T* d_A = nullptr;
    T* d_B = nullptr;
    T* d_C = nullptr;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    size_t threadsPerBlock = 256; // A common choice for the number of threads in a bloc
    size_t blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; // Watch out the 'N - 1'
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    auto ret = std::vector<T>{h_C, h_C + a.size()};
    free(h_C);
    return ret;
}

template<typename T>
std::vector<std::vector<T>>
matrixAddUsingGPU(std::vector<std::vector<T>> a, std::vector<std::vector<T>> b) {
    int row_size = a.size();
    int col_size = a.front().size();

    assert(row_size == b.size());

    auto h_A = new T[row_size * col_size]{};
    auto h_B = new T[row_size * col_size]{};
    auto h_C = new T[row_size * col_size]{};

    for (int i = 0; i < row_size; ++i) {
        assert(col_size == a[i].size() && col_size == b[i].size());

        for (int j = 0; j < col_size; ++j) {
            h_A[i * col_size + j] = a[i][j];
            h_B[i * col_size + j] = b[i][j];
        }
    }

    T* d_A = nullptr;
    T* d_B = nullptr;
    T* d_C = nullptr;
    auto size = row_size * col_size * sizeof(T);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(row_size / threadsPerBlock.x + 1, col_size / threadsPerBlock.y + 1);
    matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, row_size, col_size);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::vector<std::vector<T>> ret(row_size);
    for (int i = 0; i < row_size; ++i) {
        ret[i] = std::vector<T>{h_C + i * col_size, h_C + (i + 1) * col_size};
    }

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return ret;
}

template<typename T>
std::vector<std::vector<T>>
matrixMulUsingGPU(std::vector<std::vector<T>> a, std::vector<std::vector<T>> b) {
    int m = a.size();
    int k = a.front().size();
    int n = b.front().size();

    assert(k == b.size());

    auto h_A = new T[m * k]{};
    auto h_B = new T[k * n]{};
    auto h_C = new T[m * n]{};

    for (int i = 0; i < m; ++i) {
        assert(k == a[i].size());

        for (int j = 0; j < k; ++j) {
            h_A[i * k + j] = a[i][j];
        }
    }

    for (int i = 0; i < k; ++i) {
        assert(n == b[i].size());

        for (int j = 0; j < n; ++j) {
            h_B[i * n + j] = b[i][j];
        }
    }

    T* d_A = nullptr;
    T* d_B = nullptr;
    T* d_C = nullptr;

    cudaMalloc(&d_A, m * k * sizeof(T));
    cudaMalloc(&d_B, n * k * sizeof(T));
    cudaMalloc(&d_C, m * n * sizeof(T));

    cudaMemcpy(d_A, h_A, m * k * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * k * sizeof(T), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(m / threadsPerBlock.x + 1, k / threadsPerBlock.y + 1);
    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, m, k, n);


    cudaMemcpy(h_C, d_C, m * n * sizeof(T), cudaMemcpyDeviceToHost);

    std::vector<std::vector<T>> ret(m);
    for (int i = 0; i < m; ++i) {
        ret[i] = std::vector<T>{h_C + i * n, h_C + (i + 1) * n};
    }

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return ret;
}

template<typename T>
void printVector(const std::vector<T>& vec) {
    for (const auto& ele: vec) {
        std::cout << ele << ' ';
    }
    std::cout << std::endl;
}


template<typename T>
void printMatrix(const std::vector<std::vector<T>>& matrix) {
    for (const auto& vec: matrix) {
        for (const auto& ele: vec) {
            std::cout << ele << ' ';
        }
        std::cout << std::endl;
    }
}

int main() {
    // Vector Addition
    std::vector<int> a_1 = {1, 2, 3};
    std::vector<int> b_1 = {2, 2, 2};
    auto ret_1 = vectorAddUsingGPU(a_1, b_1);
    printVector(ret_1);

    // Matrix Addition
    std::vector<std::vector<int>> a_2 = {{1, 2},
                                         {3, 4}};
    std::vector<std::vector<int>> b_2 = {{2, 2},
                                         {2, 2}};
    auto ret_2 = matrixAddUsingGPU(a_2, b_2);
    printMatrix(ret_2);

    // Matrix multiplication
    std::vector<std::vector<int>> a_3 = {{1, 1},
                                         {1, 1},
                                         {1, 1}};
    std::vector<std::vector<int>> b_3 = {{2, 2, 2},
                                         {2, 2, 2}};
    auto ret_3 = matrixMulUsingGPU(a_3, b_3);
    printMatrix(ret_3);

    return 0;
}

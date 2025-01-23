#include <hip/hip_runtime.h>
#include <iostream>

// Define the dimensions of the matrices
#define M 2 // Rows of A and C
#define N 2 // Columns of B and C
#define K 3 // Columns of A and rows of B

// GPU kernel for matrix multiplication
__global__ void matrix_mul(const float *A, const float *B, float *C, int M,
                           int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
      sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

int main() {
  // Host matrices
  float h_A[M * K] = {1, 2, 3, 4, 5, 6};    // 2x3 matrix
  float h_B[K * N] = {7, 8, 9, 10, 11, 12}; // 3x2 matrix
  float h_C[M * N];                         // Result 2x2 matrix

  // Device matrices
  float *d_A, *d_B, *d_C;

  // Allocate device memory
  hipMalloc(&d_A, M * K * sizeof(float));
  hipMalloc(&d_B, K * N * sizeof(float));
  hipMalloc(&d_C, M * N * sizeof(float));

  // Copy host data to device
  hipMemcpy(d_A, h_A, M * K * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_B, h_B, K * N * sizeof(float), hipMemcpyHostToDevice);

  // Define block and grid dimensions
  dim3 blockDim(16, 16);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
               (M + blockDim.y - 1) / blockDim.y);

  // Launch the matrix multiplication kernel
  hipLaunchKernelGGL(matrix_mul, gridDim, blockDim, 0, 0, d_A, d_B, d_C, M, N,
                     K);

  // Copy the result back to host
  hipMemcpy(h_C, d_C, M * N * sizeof(float), hipMemcpyDeviceToHost);

  // Print the result
  std::cout << "Matrix C (Result):" << std::endl;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      std::cout << h_C[i * N + j] << " ";
    }
    std::cout << std::endl;
  }

  // Free device memory
  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_C);

  return 0;
}

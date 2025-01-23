#include <hip/hip_runtime.h>
#include <iostream>
#include <rocblas/rocblas.h>

// Function to check HIP errors
#define CHECK_HIP_ERROR(call)                                                  \
  {                                                                            \
    hipError_t err = call;                                                     \
    if (err != hipSuccess) {                                                   \
      std::cerr << "HIP Error: " << hipGetErrorString(err) << " at line "      \
                << __LINE__ << std::endl;                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

// Function to check rocBLAS errors
#define CHECK_ROCBLAS_ERROR(call)                                              \
  {                                                                            \
    rocblas_status err = call;                                                 \
    if (err != rocblas_status_success) {                                       \
      std::cerr << "rocBLAS Error: " << err << " at line " << __LINE__         \
                << std::endl;                                                  \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

int main() {
  // Matrix dimensions
  const int M = 2; // Rows of A and C
  const int N = 2; // Columns of B and C
  const int K = 3; // Columns of A and rows of B

  // Host matrices
  float h_A[M * K] = {1, 2, 3, 4, 5, 6};    // 2x3 matrix
  float h_B[K * N] = {7, 8, 9, 10, 11, 12}; // 3x2 matrix
  float h_C[M * N];                         // Result 2x2 matrix

  // Device matrices
  float *d_A, *d_B, *d_C;

  // Allocate device memory
  CHECK_HIP_ERROR(hipMalloc(&d_A, M * K * sizeof(float)));
  CHECK_HIP_ERROR(hipMalloc(&d_B, K * N * sizeof(float)));
  CHECK_HIP_ERROR(hipMalloc(&d_C, M * N * sizeof(float)));

  // Copy data from host to device
  CHECK_HIP_ERROR(
      hipMemcpy(d_A, h_A, M * K * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP_ERROR(
      hipMemcpy(d_B, h_B, K * N * sizeof(float), hipMemcpyHostToDevice));

  // Create rocBLAS handle
  rocblas_handle handle;
  CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

  // Scalar multipliers for the operation C = alpha * A * B + beta * C
  float alpha = 1.0f;
  float beta = 0.0f;

  // Perform matrix multiplication: d_C = alpha * d_A * d_B + beta * d_C
  CHECK_ROCBLAS_ERROR(rocblas_sgemm(handle, rocblas_operation_none,
                                    rocblas_operation_none, M, N, K, &alpha,
                                    d_A, M, d_B, K, &beta, d_C, M));

  // Copy the result from device to host
  CHECK_HIP_ERROR(
      hipMemcpy(h_C, d_C, M * N * sizeof(float), hipMemcpyDeviceToHost));

  // Print the result
  std::cout << "Result matrix C:" << std::endl;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      std::cout << h_C[i * N + j] << " ";
    }
    std::cout << std::endl;
  }

  // Clean up
  CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));
  CHECK_HIP_ERROR(hipFree(d_A));
  CHECK_HIP_ERROR(hipFree(d_B));
  CHECK_HIP_ERROR(hipFree(d_C));

  return 0;
}
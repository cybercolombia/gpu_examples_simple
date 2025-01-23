#include <stdio.h>
#include <stdlib.h>

#define M 2 // Rows of A and C
#define N 2 // Columns of B and C
#define K 3 // Columns of A and rows of B

void print_matrix(const char *name, float *mat, int rows, int cols) {
  printf("%s:\n", name);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%6.2f ", mat[i * cols + j]);
    }
    printf("\n");
  }
}

int main() {
  // Host matrices
  float A[M * K] = {1, 2, 3, 4, 5, 6};    // 2x3 matrix
  float B[K * N] = {7, 8, 9, 10, 11, 12}; // 3x2 matrix
  float C[M * N] = {0}; // Result 2x2 matrix, initialized to zero

// Offload matrices to GPU and perform computation
#pragma omp target data map(to : A[0 : M * K], B[0 : K * N])                   \
    map(from : C[0 : M * N])
  {
#pragma omp target teams distribute parallel for collapse(2)
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
          sum += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
      }
    }
  }

  // Print the results
  print_matrix("Matrix A", A, M, K);
  print_matrix("Matrix B", B, K, N);
  print_matrix("Matrix C (Result)", C, M, N);

  return 0;
}
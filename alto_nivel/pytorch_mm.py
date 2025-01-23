import torch

device = torch.device("hip")

A = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32).to(device)  # 2x3 matrix
B = torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.float32).to(
    device
)  # 3x2 matrix

C = torch.matmul(A, B)
print("\nResult of A x B:")
print(C_cpu)

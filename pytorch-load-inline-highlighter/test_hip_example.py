import torch
from torch.utils.cpp_extension import load_inline

# HIP code examples for AMD GPUs
hip_code = """
#include <hip/hip_runtime.h>

__global__ void vector_add_hip(float* a, float* b, float* c, int n) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
"""

hip_kernel_code = """
__global__ void matrix_multiply_hip(float* A, float* B, float* C, int N) {
    int row = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int col = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor launch_kernel(torch::Tensor a, torch::Tensor b) {
    auto c = torch::empty_like(a);
    
    dim3 threads(16, 16);
    dim3 blocks((a.size(1) + threads.x - 1) / threads.x,
                (a.size(0) + threads.y - 1) / threads.y);
    
    hipLaunchKernelGGL(matrix_multiply_hip, blocks, threads, 0, 0,
        a.data_ptr<float>(), b.data_ptr<float>(), 
        c.data_ptr<float>(), a.size(0));
    
    hipDeviceSynchronize();
    return c;
}
"""

# Test different variable naming patterns for HIP
my_hip_kernel = """
__device__ float compute_distance_hip(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return sqrtf(dx * dx + dy * dy);
}

__global__ void process_points_hip(float* points, float* distances, int n) {
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (tid < n - 1) {
        distances[tid] = compute_distance_hip(
            points[tid * 2], points[tid * 2 + 1],
            points[(tid + 1) * 2], points[(tid + 1) * 2 + 1]
        );
    }
}
"""

hip_sources_example = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>

torch::Tensor hip_function(torch::Tensor input) {
    // HIP-specific implementation
    return input.to(torch::kHIP);
}
"""

# Example with load_inline using HIP
with tempfile.TemporaryDirectory() as build_dir:
    hip_module = load_inline(
        name="hip_ops",
        cpp_sources=cpp_code,
        hip_sources=hip_kernel_code,  # Note: hip_sources parameter
        functions=["launch_kernel", "hip_function"],
        with_rocm=True,  # Enable ROCm/HIP compilation
        verbose=True,
        build_directory=build_dir,
    )
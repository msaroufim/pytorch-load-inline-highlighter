from torch.utils.cpp_extension import load_inline

cpp_code = """
torch::Tensor to_gray(torch::Tensor input);
"""

cuda_kernel_code = """
__global__ void to_gray_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 0.299f * input[idx * 3] + 
                     0.587f * input[idx * 3 + 1] + 
                     0.114f * input[idx * 3 + 2];
    }
}

torch::Tensor to_gray(torch::Tensor input) {
    auto output = torch::empty({input.size(0), input.size(1)}, input.options());
    
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;
    
    to_gray_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.size(0) * input.size(1)
    );
    
    return output;
}
"""

cpp_sources_example = """
#include <torch/extension.h>

torch::Tensor matrix_multiply(torch::Tensor a, torch::Tensor b) {
    return torch::mm(a, b);
}
"""

# Test different variable naming patterns
my_cpp_kernel = """
void process_data(torch::Tensor& data) {
    // C++ processing code
    for (int i = 0; i < data.size(0); ++i) {
        data[i] *= 2.0f;
    }
}
"""

cuda_implementation = """
__device__ float compute_value(float x, float y) {
    return sqrtf(x * x + y * y);
}

__global__ void vector_add(float* a, float* b, float* c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}
"""

cuda_module = load_inline(
    name="to_gray_cuda",
    cpp_sources=cpp_sources_example, 
    cuda_sources=cuda_kernel_code, 
    functions=["to_gray", "matrix_multiply"],
    with_cuda=True,
    verbose=True,
    extra_cflags=["-std=c++17"],
    extra_cuda_cflags=["-arch=sm_89"],
)


# PyTorch load_inline() Syntax Highlighter

A Visual Studio Code extension that provides syntax highlighting for C++, CUDA, and HIP/ROCm code within Python string literals used with PyTorch's `load_inline()` function.

## Features

- **Automatic Detection**: Recognizes C++/CUDA/HIP/ROCm code in various contexts:
  - `cpp_sources="..."`, `cuda_sources="..."`, and `hip_sources="..."` parameters
  - Variables ending with `_cpp`, `_cuda`, `_hip`, `_kernel`
  - Variables starting with `cpp_`, `cuda_`, or `hip_`
  - Content-based detection (e.g., `torch::Tensor`, `__global__`, `blockIdx`, `hipBlockIdx_x`)

- **Multi-String Support**: Works with single quotes, double quotes, and triple quotes
- **Context-Aware**: Only activates within Python files
- **Lightweight**: Uses TextMate grammar injection for optimal performance

## Installation

### From Source

1. Clone this repository
2. Install dependencies: `npm install`
3. Compile: `npm run compile`
4. Install the extension in VS Code:
   - Open VS Code
   - Go to Extensions (Ctrl+Shift+X)
   - Click "..." → "Install from VSIX..."
   - Select the generated `.vsix` file

### Development

1. Open this project in VS Code
2. Press F5 to launch Extension Development Host
3. Open a Python file with `load_inline()` code
4. Verify syntax highlighting works

## Usage Examples

The extension automatically highlights C++/CUDA/HIP/ROCm code in these patterns:

```python
# Variable name patterns
cpp_code = """
torch::Tensor my_function(torch::Tensor input) {
    return input * 2.0;
}
"""

cuda_kernel_code = """
__global__ void my_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= 2.0f;
    }
}
"""

hip_kernel_code = """
__global__ void hip_kernel(float* data, int size) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx < size) {
        data[idx] *= 2.0f;
    }
}
"""

# Function parameters
load_inline(
    name="my_module",
    cpp_sources="""
    #include <torch/extension.h>
    torch::Tensor process(torch::Tensor x) { return x; }
    """,
    cuda_sources="""
    __global__ void kernel(float* x, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) x[i] *= 2.0f;
    }
    """,
    hip_sources="""
    __global__ void hip_kernel(float* data, int size) {
        int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        if (idx < size) data[idx] *= 2.0f;
    }
    """,
    with_rocm=True
)
```

## Supported Patterns

### Variable Names
- `*_cpp`, `*_cuda`, `*_hip`, `*_kernel`
- `cpp_*`, `cuda_*`, `hip_*` 
- `cpp_code`, `cuda_kernel_code`, `hip_kernel_code`, `kernel_code`

### Function Parameters
- `cpp_sources=`
- `cuda_sources=`
- `hip_sources=`

### Content Detection
- C++: `torch::Tensor` declarations
- CUDA: `__global__`, `__device__`, `__host__`, `blockIdx`, `threadIdx`
- HIP: `hipBlockIdx_x`, `hipThreadIdx_x`, `hipLaunchKernelGGL`, `hipDeviceSynchronize`

## Requirements

- Visual Studio Code 1.74.0 or higher
- Existing C++ extension (built-in with VS Code)
- Optional: CUDA or HIP/ROCm extensions for enhanced GPU code features

## Publishing to VS Code Marketplace

To publish this extension to the VS Code Marketplace:

### 1. Create Publisher Account
- Go to https://marketplace.visualstudio.com/manage
- Sign in with Microsoft account
- Create a new publisher ID

### 2. Generate Personal Access Token
- Visit https://dev.azure.com/your-organization/_usersSettings/tokens
- Create new token with "Marketplace (Manage)" scope
- Copy the token (you won't see it again)

### 3. Update package.json
```json
{
  "publisher": "your-publisher-id",
  "repository": {
    "type": "git",
    "url": "https://github.com/your-username/pytorch-load-inline-highlighter"
  }
}
```

### 4. Create LICENSE file
```bash
echo "MIT License

Copyright (c) 2024 Your Name

Permission is hereby granted..." > LICENSE
```

### 5. Install vsce globally
```bash
npm install -g @vscode/vsce
```

### 6. Login to publisher
```bash
vsce login your-publisher-id
# Enter your Personal Access Token when prompted
```

### 7. Publish the extension
```bash
vsce publish
# Or specify version: vsce publish minor
```

### 8. Update existing extension
```bash
vsce publish patch  # for bug fixes (0.2.0 -> 0.2.1)
vsce publish minor  # for new features (0.2.0 -> 0.3.0)
vsce publish major  # for breaking changes (0.2.0 -> 1.0.0)
```

## License

MIT
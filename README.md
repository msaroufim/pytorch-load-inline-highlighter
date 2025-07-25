# PyTorch load_inline() Syntax Highlighter

A Visual Studio Code extension that provides syntax highlighting for C++ and CUDA code within Python string literals used with PyTorch's `load_inline()` function.

![screen.png](img/screen.png)

## Installation

It's this guy
https://marketplace.visualstudio.com/items?itemName=msaroufim.pytorch-load-inline-highlighter

## Features

- **Automatic Detection**: Recognizes C++/CUDA code in various contexts:
  - `cpp_sources="..."` and `cuda_sources="..."` parameters
  - Variables ending with `_cpp`, `_cuda`, `_kernel`
  - Variables starting with `cpp_` or `cuda_`
  - Content-based detection (e.g., `torch::Tensor`, `__global__`, `blockIdx`)

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

The extension automatically highlights C++/CUDA code in these patterns:

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
    """
)
```

## Supported Patterns

### Variable Names
- `*_cpp`, `*_cuda`, `*_kernel`
- `cpp_*`, `cuda_*` 
- `cpp_code`, `cuda_kernel_code`, `kernel_code`

### Function Parameters
- `cpp_sources=`
- `cuda_sources=`

### Content Detection
- C++: `torch::Tensor` declarations
- CUDA: `__global__`, `__device__`, `__host__`, `blockIdx`, `threadIdx`

## Requirements

- Visual Studio Code 1.74.0 or higher
- Existing C++ and CUDA extensions for full language support

## Acknowledgements

Thank you to Steven Arellano for giving me the idea to just go ahead and do this and thank you Claude for doing it.

## License

MIT
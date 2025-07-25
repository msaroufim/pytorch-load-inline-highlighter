# Installation Instructions

## Quick Install

### Option 1: Install from VSIX (Recommended)
1. Download the `pytorch-load-inline-highlighter-0.1.0.vsix` file
2. Open VS Code
3. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac) to open the command palette
4. Type "Extensions: Install from VSIX..." and select it
5. Navigate to and select the downloaded `.vsix` file
6. The extension will be installed and activated automatically

### Option 2: Using Command Line
```bash
code --install-extension pytorch-load-inline-highlighter-0.1.0.vsix
```

## Development Setup

If you want to modify or contribute to the extension:

### Prerequisites
- Node.js 16+ 
- npm
- Visual Studio Code

### Steps
1. Clone this repository
2. Install dependencies:
   ```bash
   npm install
   ```
3. Compile the extension:
   ```bash
   npm run compile
   ```
4. Open the project in VS Code
5. Press `F5` to launch a new Extension Development Host window
6. Open a Python file with `load_inline()` code to test

### Building from Source
```bash
# Compile TypeScript
npm run compile

# Package the extension
npm run package

# Install the packaged extension
npm run install-extension
```

## Verification

After installation, create a test Python file with this content:

```python
from torch.utils.cpp_extension import load_inline

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
```

You should see C++ syntax highlighting in the `cpp_code` string and CUDA syntax highlighting in the `cuda_kernel_code` string.

## Troubleshooting

### Extension Not Working
1. Ensure the file has a `.py` extension
2. Check that C++ and CUDA extensions are installed for full language support
3. Try reloading VS Code window (`Ctrl+Shift+P` → "Developer: Reload Window")

### No Syntax Highlighting
1. Verify the variable names match supported patterns (see README.md)
2. Check that strings contain recognizable C++/CUDA syntax
3. Ensure you're using supported string formats (single, double, or triple quotes)

### Performance Issues
The extension uses lightweight TextMate grammar injection and should have minimal performance impact. If you experience issues, try disabling other language extensions temporarily.

## Uninstallation

To remove the extension:
1. Open VS Code
2. Go to Extensions (`Ctrl+Shift+X`)
3. Find "PyTorch load_inline() Syntax Highlighter"
4. Click the gear icon and select "Uninstall"
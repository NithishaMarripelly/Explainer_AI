https://medicaid-chieac.streamlit.app/

# 📘 Complete Guide: Installing Unstructured.io on Windows

> [!NOTE]
> This guide is curated for beginners to help you install the `unstructured` library on Windows without the common headaches. It consolidates solutions to all the issues we've faced, including DLL errors, missing system dependencies, and library conflicts.

## 📋 Prerequisites

Before installing the Python libraries, you **must** have the following system tools installed. 90% of errors come from skipping these steps.

### 1. Microsoft Visual C++ Redistributable
**Why?** Required by PyTorch and other deep learning libraries. Without this, you will get `WinError 1114`.
- **Download**: [VC_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe)
- **Action**: Run the installer and **restart your computer**.

### 2. Build Tools for Visual Studio
**Why?** Needed to compile some Python packages (like Detectron2) from source.
- **Download**: [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- **Action**: Install "Desktop development with C++".

### 3. Poppler (for PDF Processing)
**Why?** Unstructured uses this to read PDF files and extract images.
- **Download**: [Release 24.02.0 (or latest)](https://github.com/oschwartz10612/poppler-windows/releases/)
- **Steps**:
  1. Download the `.7z` or `.zip` file.
  2. Extract it to a permanent location, e.g., `C:\Program Files\poppler`.
  3. **Critical**: Add `C:\Program Files\poppler\Library\bin` to your System **PATH** environment variable.

### 4. Tesseract OCR
**Why?** Required to extract text from images or scanned PDFs.
- **Download**: [Tesseract Installer](https://github.com/UB-Mannheim/tesseract/wiki)
- **Steps**:
  1. Run the installer.
  2. Install to `C:\Program Files\Tesseract-OCR`.
  3. **Critical**: Add `C:\Program Files\Tesseract-OCR` to your System **PATH**.

---

## 🛠️ Installation Steps

We will use a virtual environment to keep your system clean. Open **PowerShell** and follow these steps exactly.

### Step 1: Create and Activate Environment
```powershell
# Create the virtual environment named 'venv'
python -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1
```
> [!TIP]
> If you see an error about scripts being disabled, run this command first:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Step 2: Upgrade Basic Tools
Older versions of pip can cause installation failures.
```powershell
python -m pip install --upgrade pip setuptools wheel
```

### Step 3: Install Unstructured
We install the library with local inference support to run everything on your machine.
```powershell
pip install "unstructured[local-inference]"
```

### Step 4: Install PyTorch (CPU Version)
Windows usually lacks CUDA (GPU) support out of the box, so we explicitly install the CPU version to avoid massive downloads and compatibility issues.
```powershell
# Uninstall any existing version first to be safe
pip uninstall torch torchvision torchaudio -y

# Install CPU-only version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 5: Install Helper Libraries
These handle specific file types and image processing.
```powershell
# For image handling and layout analysis
pip install opencv-python
pip install "layoutparser[layoutmodels,tesseract]"

# For file type detection
pip install python-magic-bin

# For text processing
pip install nltk
```

### Step 6: Download NLTK Data
Run this simple Python one-liner to download necessary language models.
```powershell
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

---

## 🛑 Common Issues & Fixes

We have encountered and solved these specific issues during our development.

### 1. `OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed`
- **Cause**: Missing Microsoft Visual C++ Redistributable.
- **Fix**: Install the [VC_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe) and **restart**.

### 2. `AttributeError: np.float_ was removed in the NumPy 2.0 release`
- **Cause**: Some libraries (like ChromaDB) aren't ready for NumPy 2.0 yet.
- **Fix**: Downgrade NumPy.
  ```powershell
  pip install "numpy<2.0"
  ```

### 3. `PDFInfoNotInstalledError` or `TesseractNotFoundError`
- **Cause**: Poppler or Tesseract is not in your PATH.
- **Fix**:
  1. Check if you installed them.
  2. Search Windows for "Edit the system environment variables".
  3. Click "Environment Variables".
  4. Under "System variables", find `Path` and click "Edit".
  5. Add the full paths to the `bin` folders (e.g., `C:\Program Files\poppler\Library\bin`).
  6. **Restart your terminal** for changes to take effect.

### 4. Detectron2 Installation Fails
- **Cause**: Detectron2 is complex to build on Windows.
- **Fix**: If `pip install` fails, try installing from a pre-built wheel or use git:
  ```powershell
  git clone https://github.com/facebookresearch/detectron2.git
  cd detectron2
  pip install -e .
  ```
  *Note: This requires the Visual Studio Build Tools mentioned in Prerequisites.*

---

## ✅ Verification

To verify everything is working, create a file named `test_unstructured.py` with this content:

```python
from unstructured.partition.auto import partition
import torch

print(f"PyTorch Version: {torch.__version__}")
print("PyTorch CUDA Available:", torch.cuda.is_available())

try:
    # Create a dummy file to test
    with open("test.txt", "w") as f:
        f.write("This is a test document for unstructured.")

    elements = partition("test.txt")
    print("\n✅ Unstructured successfully partitioned the file!")
    print(f"Found {len(elements)} elements.")
    print(f"Content: {elements[0].text}")

except Exception as e:
    print(f"\n❌ Error: {e}")
```

Run it:
```powershell
python test_unstructured.py
```

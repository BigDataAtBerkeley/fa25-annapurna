#!/bin/bash
# Install all possible packages that generated code might use in the Neuron venv
# Run this on the Trainium EC2 instance to ensure all packages are available

set -e

NEURON_PYTHON="/opt/aws_neuronx_venv_pytorch_2_8_nxd_training/bin/python3"
NEURON_PIP="/opt/aws_neuronx_venv_pytorch_2_8_nxd_training/bin/pip3"

echo "📦 Installing all packages in Neuron venv for generated code"
echo "============================================================"
echo ""

# Verify Neuron venv exists
if [ ! -f "$NEURON_PYTHON" ]; then
    echo "❌ Error: Neuron venv not found at $NEURON_PYTHON"
    echo "   Make sure you're running this on a Trainium instance with Neuron SDK installed"
    exit 1
fi

echo "Using Neuron Python: $NEURON_PYTHON"
echo ""

# Upgrade pip first
echo "⬆️  Upgrading pip..."
$NEURON_PIP install --upgrade pip

# Core packages (already in venv, but verify)
echo "🔍 Verifying core packages..."
$NEURON_PYTHON -c "import torch; import torch_xla.core.xla_model as xm; print('✓ torch and torch_xla available')" || {
    echo "⚠️  torch or torch_xla not found - this is unexpected in Neuron venv"
}

# Install/upgrade packages that generated code uses
echo ""
echo "📚 Installing required packages..."

# HuggingFace libraries (for NLP tokenization)
echo "  Installing transformers..."
$NEURON_PIP install transformers tokenizers

# Datasets (used internally by dataset_loader)
echo "  Installing datasets..."
$NEURON_PIP install datasets

# Torchvision (available but not used - install just in case)
echo "  Installing torchvision..."
$NEURON_PIP install torchvision

# NumPy (usually comes with torch, but ensure it's there)
echo "  Verifying numpy..."
$NEURON_PIP install numpy

echo ""
echo "⬇️  Pre-downloading common HuggingFace tokenizers (optional)..."
echo "   Tokenizers only (~1-5 MB each), not full models (100MB-10GB+)"
echo "   This prevents download delays during code execution"
echo ""

# Support non-interactive mode via environment variable
if [ "${SKIP_TOKENIZER_DOWNLOAD:-false}" = "true" ]; then
    DOWNLOAD_TOKENIZERS="n"
else
    # Interactive mode: prompt user
    if [ -t 0 ]; then
        read -p "Pre-download tokenizers? (y/n, default=y): " -n 1 -r
        echo ""
        DOWNLOAD_TOKENIZERS="$REPLY"
    else
        # Non-interactive mode: default to yes
        DOWNLOAD_TOKENIZERS="y"
    fi
fi

if [[ $DOWNLOAD_TOKENIZERS =~ ^[Yy]$ ]] || [[ -z $DOWNLOAD_TOKENIZERS ]]; then
    $NEURON_PYTHON << 'PYEOF'
from transformers import AutoTokenizer
import os

# Common tokenizers that generated code frequently uses
# NOTE: We only download tokenizers (~1-5 MB each), NOT full models (100MB-10GB+)
common_models = [
    'bert-base-uncased',           # Most common for NLP tasks (~1.5 MB)
    'bert-base-cased',              # Alternative casing (~1.5 MB)
    'distilbert-base-uncased',     # Lighter alternative (~1.5 MB)
    'roberta-base',                 # Alternative transformer (~1.5 MB)
    'gpt2',                         # Sometimes used for language modeling (~1 MB)
]

print("Pre-downloading tokenizers (vocab files only, not model weights)...")
cache_dir = os.path.expanduser('~/.cache/huggingface/')
total_size = 0

for model_name in common_models:
    try:
        print(f"  Downloading {model_name} tokenizer...", end=' ', flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        # Get size of tokenizer files (vocab.json, tokenizer_config.json, etc.)
        import os
        tokenizer_dir = os.path.join(cache_dir, 'hub', f'models--{model_name.replace("/", "--")}')
        if os.path.exists(tokenizer_dir):
            size_mb = sum(os.path.getsize(os.path.join(dirpath, f)) 
                         for dirpath, dirnames, filenames in os.walk(tokenizer_dir)
                         for f in filenames) / (1024 * 1024)
            total_size += size_mb
            print(f"✓ (~{size_mb:.1f} MB)")
        else:
            print(f"✓")
    except Exception as e:
        print(f"⚠️  Failed: {e}")

print(f"\n✅ Tokenizers cached at: {cache_dir}")
print(f"   Total size: ~{total_size:.1f} MB (negligible)")
print("   These will be reused automatically on subsequent runs")
PYEOF
else
    echo "⏭️  Skipping tokenizer pre-download"
    echo "   Tokenizers will be downloaded on-demand during code execution"
fi

echo ""
echo "🔍 Verifying all installations..."
$NEURON_PYTHON << 'PYEOF'
import sys

packages_to_check = {
    'torch': 'PyTorch',
    'torch_xla': 'PyTorch/XLA',
    'torchvision': 'TorchVision',
    'transformers': 'Transformers',
    'datasets': 'Datasets',
    'numpy': 'NumPy',
}

print("Checking installed packages:")
all_ok = True
for package, name in packages_to_check.items():
    try:
        if package == 'torch_xla':
            import torch_xla.core.xla_model as xm
            print(f"  ✓ {name} (torch_xla)")
        else:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {name} ({package}) - version {version}")
    except ImportError:
        print(f"  ❌ {name} ({package}) - NOT INSTALLED")
        all_ok = False

if all_ok:
    print("\n✅ All packages verified and ready!")
else:
    print("\n⚠️  Some packages are missing!")
    sys.exit(1)
PYEOF

echo ""
echo "📊 Package locations:"
echo "  Neuron Python: $NEURON_PYTHON"
echo "  PyTorch: $($NEURON_PYTHON -c 'import torch; print(torch.__file__)')"
echo "  Transformers cache: ~/.cache/huggingface/"
echo ""
echo "✅ Installation complete!"
echo ""
echo "💡 All generated code will use packages from the Neuron venv"
echo "   No additional package installations needed during code execution"


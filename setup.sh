#!/bin/bash
# Setup script for Book2MDProcessor
# Usage: bash setup.sh
#        bash setup.sh --with-eval   (also installs Page2MDBench and evaluation deps)
set -e

echo "==> Installing system dependencies..."
apt-get install -y poppler-utils

echo "==> Installing Python dependencies..."
pip install -r requirements.txt

if [[ "$1" == "--with-eval" ]]; then
    echo "==> Cloning Page2MDBench..."
    if [ ! -d "Page2MDBench" ]; then
        git clone https://github.com/Hipsterfil998/Page2MDBench.git
    else
        echo "    Page2MDBench already exists, skipping clone."
    fi
    echo "==> Installing evaluation dependencies..."
    pip install rapidfuzz sacrebleu mistune bert-score
fi

echo "==> Done."

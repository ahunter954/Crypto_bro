#!/bin/bash

# Création du virtualenv si besoin
if [ ! -d "venv" ]; then
  python3 -m venv venv
fi

# Activation du venv
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Installation des dépendances avec contournement de PEP 668
pip install --break-system-packages -r requirements.txt

# Compilation Cython
python setup.py build_ext --inplace

echo "✅ Environnement prêt !"

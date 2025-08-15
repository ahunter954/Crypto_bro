# Makefile pour projet Crypto RL

VENV_DIR=venv
PYTHON=$(VENV_DIR)/bin/python

# Création de l'environnement virtuel
venv:
	python3 -m venv $(VENV_DIR)

# Installation des dépendances sans activation manuelle
install: venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install --break-system-packages -r requirements.txt

# Compilation du module Cython
cython-build:
	$(PYTHON) setup.py build_ext --inplace

# Installation complète : deps + compilation
setup: install cython-build

# Entraînement PPO
train:
	$(PYTHON) train.py

# Backtest
backtest:
	$(PYTHON) backtest.py

# Métriques de performance
metrics:
	$(PYTHON) metrics.py

# Nettoyage des fichiers compilés
clean:
	rm -rf build *.so *.c *.pyd __pycache__/

.PHONY: venv install cython-build setup train backtest metrics clean

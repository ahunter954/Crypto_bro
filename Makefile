# Makefile pour ton projet RL Trading avec Cython accéléré

VENV_DIR=venv
PYTHON=$(VENV_DIR)/bin/python
PIP=$(VENV_DIR)/bin/pip

# Création du virtualenv
venv:
	python3 -m venv $(VENV_DIR)

# Installation des dépendances
install: venv
	source $(VENV_DIR)/bin/activate && \
	$(PIP) install --upgrade pip && \
	$(PIP) install --break-system-packages -r requirements.txt

# Compilation du Cython
cython-build:
	$(PYTHON) setup.py build_ext --inplace

# Tout en une commande
setup: install cython-build

# Lancement de l'entraînement
train:
	$(PYTHON) train.py

# Lancement du backtest
backtest:
	$(PYTHON) backtest.py

# Calcul des métriques
metrics:
	$(PYTHON) metrics.py

# Nettoyage des fichiers compilés
clean:
	rm -rf build *.so *.c *.pyd

.PHONY: venv install cython-build setup train backtest metrics clean

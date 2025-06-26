# 🧠 Crypto Trading RL with PPO

Un projet de trading algorithmique avec renforcement (PPO) dans un environnement personnalisé Gymnasium, optimisé en Cython pour les performances.

---

## 🚀 Démarrage rapide

### 1. Cloner le repo et se placer dans le dossier

```bash
git clone <url-du-repo>
cd Crypto_bro
```

### 2. Créer l'environnement Python (venv)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install --upgrade pip
pip install --break-system-packages -r requirements.txt
```

### 4. Compiler le module Cython

```bash
python setup.py build_ext --inplace
```

OU en une seule commande :

```bash
make setup
```

---

## 🧪 Utilisation

### Entraînement

```bash
make train
```

### Backtest

```bash
make backtest
```

### Analyse de performance

```bash
make metrics
```

---

## 📁 Structure du projet

```bash
Crypto_bro/
├── data/               # Données historiques
├── envs/               # Environnements Gym (Python et Cython)
├── models/             # Modèles sauvegardés
├── utils/              # Fonctions utilitaires (indicateurs, API Binance)
├── train.py            # Script d'entraînement PPO
├── backtest.py         # Script de backtest
├── metrics.py          # Analyse des performances
├── requirements.txt    # Dépendances Python
├── setup.py            # Compilation Cython
├── install.sh          # Script d'installation complet
├── Makefile            # Commandes automatisées
```

---

## 💡 Objectifs

* Implémenter un environnement de trading personnalisé Gymnasium
* Entraîner un agent PPO avec Stable-Baselines3
* Accélérer l'exécution avec Cython
* Comparer la stratégie RL avec le Buy & Hold
* Mesurer rendement, drawdown, Sharpe ratio et ratio gain/perte

---

## 🛠 Requirements

* Python 3.9+
* Cython
* NumPy, pandas, gymnasium, stable-baselines3
* matplotlib (pour visualisation)

---

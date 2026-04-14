#!/usr/bin/env bash
# Contexte : purge cache DVC + re-track données après fork/changement de remote
# À lancer manuellement lors de changement de compte
set -euo pipefail

echo "[1/5] Suppression des fichiers .dvc et dvc.lock..."
find . \( -name "*.dvc" -o -name "dvc.lock" \) \
  -not -path "./.git/*" \
  -not -path "./.dvc" \
  -not -path "./.dvc/*" \
  -print -delete

echo "[2/5] Purge du cache et tmp DVC..."
rm -rf .dvc/cache .dvc/tmp

echo "[3/5] DVC add + push..."
dvc add data/raw/rakuten/X_train_update.csv \
        data/raw/rakuten/Y_train_CVw08PX.csv \
        data/raw/product_categories.csv
dvc push

echo "[4/5] Git add + commit + push..."
git add .
git commit -m "Réinit DVC — purge cache, re-track data, new remote"
git push origin feature/airflow

echo ""
echo "Réinit complète."
#!/bin/bash
set -e

echo "[Setup] Configuring Git and DVC..."

# ============================================================================
# 1. Git Configuration
# ============================================================================
if [ -n "$GIT_AUTHOR_NAME" ]; then
    git config --global user.name "$GIT_AUTHOR_NAME"
    echo "[Git] user.name = $GIT_AUTHOR_NAME"
fi

if [ -n "$GIT_AUTHOR_EMAIL" ]; then
    git config --global user.email "$GIT_AUTHOR_EMAIL"
    echo "[Git] user.email = $GIT_AUTHOR_EMAIL"
fi

git config --global core.fileMode false
git config --global init.defaultBranch main

# ============================================================================
# 2. SSH Configuration (GitHub)
# ============================================================================
if [ -f "/root/.ssh/id_github" ]; then
    export GIT_SSH_COMMAND="ssh -i /root/.ssh/id_github -o StrictHostKeyChecking=no -o IdentitiesOnly=yes"
    git config --global core.sshCommand "ssh -i /root/.ssh/id_github -o StrictHostKeyChecking=no -o IdentitiesOnly=yes"
    echo "[SSH] Git configured with id_github"
else
    echo "[Warning] /root/.ssh/id_github not found"
fi

git config --global url."git@github.com:".insteadOf "https://github.com/"

# ============================================================================
# 3. DVC Configuration (DagsHub)
# ============================================================================
if [ -d "/app/.dvc" ]; then
    # Activer l'autostage
    dvc config core.autostage true
    
    # --- AJOUT : Configuration dynamique du Remote ---
    # Si le remote 'storage' n'existe pas, on le crée
    if ! dvc remote list | grep -q "storage"; then
        echo "[DVC] Adding missing remote 'storage'..."
        # On utilise les variables d'environnement pour construire l'URL
        # On suppose que DAGSHUB_USER et DAGSHUB_REPO sont passés au conteneur
        REMOTE_URL="https://dagshub.com/${DAGSHUB_USER}/${DAGSHUB_REPO}.dvc"
        dvc remote add -d storage "$REMOTE_URL" || echo "[Error] Could not add remote"
    fi

    # --- AJOUT : Authentification  ---
    if [ -n "$DAGSHUB_TOKEN" ]; then
        echo "[DVC] Configuring authentication for 'storage'..."
        dvc remote modify --local storage user "$DAGSHUB_USER"
        dvc remote modify --local storage password "$DAGSHUB_TOKEN"
        dvc remote modify --local storage auth basic
    fi

    echo "[DVC] Current remotes:"
    dvc remote list
else
    echo "[Warning] .dvc directory not found"
fi

# ============================================================================
# Ready!
# ============================================================================
echo "[Setup] ✓ Git + DVC ready!"
[ -n "$GITHUB_USER"  ] && echo "  GitHub:  $GITHUB_USER"
[ -n "$DAGSHUB_USER" ] && echo "  DagsHub: $DAGSHUB_USER"
echo ""

exec "$@"
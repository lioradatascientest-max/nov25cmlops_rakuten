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
# 2. SSH Configuration
# ============================================================================

# Pas de chmod — le mount WSL2 est en lecture seule pour les permissions
# SSH fonctionne en root dans Docker sans vérification stricte des permissions

if [ -f "/root/.ssh/id_github" ]; then
    export GIT_SSH_COMMAND="ssh -i /root/.ssh/id_github -o StrictHostKeyChecking=no -o IdentitiesOnly=yes"
    git config --global core.sshCommand "ssh -i /root/.ssh/id_github -o StrictHostKeyChecking=no -o IdentitiesOnly=yes"
    echo "[SSH] Git configured with id_github"
else
    echo "[Warning] /root/.ssh/id_github not found — git push via SSH may fail"
fi

git config --global url."git@github.com:".insteadOf "https://github.com/"
echo "[Git] Configured to use SSH for GitHub"

# ============================================================================
# 3. DVC Configuration
# ============================================================================

if [ -d "/app/.dvc" ]; then
    if [ -n "$DAGSHUB_USER" ] && [ -n "$DAGSHUB_REPO" ]; then
        #DVC_REMOTE_URL="https://dagshub.com/${DAGSHUB_USER}/${DAGSHUB_REPO}.s3"
        DVC_REMOTE_URL="https://dagshub.com/${DAGSHUB_USER}/${DAGSHUB_REPO}.dvc"
        dvc remote add -d origin "$DVC_REMOTE_URL" >/dev/null 2>&1 || \
        dvc remote modify origin url "$DVC_REMOTE_URL" >/dev/null 2>&1 || \
        echo "[Warning] Unable to configure DVC remote origin"
        echo "[DVC] origin = $DVC_REMOTE_URL"
    fi

    if [ -n "$DAGSHUB_TOKEN" ]; then
        dvc remote modify origin --local access_key_id "$DAGSHUB_TOKEN" >/dev/null 2>&1 || \
        echo "[Warning] Unable to set DVC access_key_id"
        dvc remote modify origin --local secret_access_key "$DAGSHUB_TOKEN" >/dev/null 2>&1 || \
        echo "[Warning] Unable to set DVC secret_access_key"
    fi

    dvc config core.autostage true >/dev/null 2>&1 || echo "[Warning] Unable to enable DVC autostage"
    echo "[DVC] autostage configured"
    echo "[DVC] Configured remotes:"
    dvc remote list || echo "[Warning] Unable to list DVC remotes"
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
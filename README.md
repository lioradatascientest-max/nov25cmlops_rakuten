# MLOps Rakuten

Classification de types de produits pour Rakuten France

> Projet MLOps : pipeline complète d'entrainement, exposition via API sécurisée, orchestration en batch, monitoring et versioning.

---

## Table des matières

- [Architecture globale](#architecture-globale)
- [Modes d'exécution](#modes-dexécution)
- [Project Organization](#project-organization)
- [Installation](#installation)
- [Contribuer — configuration Git/SSH requise](#contribuer--configuration-gitssh-requise)
- [Structure du pipeline de données](#structure-du-pipeline-de-données)
- [Sécurité et Gateway](#sécurité-et-gateway)
- [Suivi d'expériences et versioning](#suivi-dexpériences-et-versioning)
- [Containerisation via Docker](#containerisation-via-docker)
- [Lancer l'application avec Docker](#lancer-lapplication-avec-docker)
- [Orchestration batch avec Airflow](#orchestration-batch-avec-airflow)
- [Monitoring](#monitoring)
- [Drift Monitoring — Evidently](#drift-monitoring--evidently)
- [Tests](#tests)
- [Commandes Makefile](#commandes-makefile)

---

## Architecture globale

Le projet suit une architecture microservices conteneurisée. Le même pipeline de données et d'entraînement est accessible via trois modes d'exécution distincts, selon le contexte (développement, API, automatisation batch). Une interface Streamlit est positionnée devant la gateway.

```
┌───────────────────────────────────────────────────────────────────────┐
│               MODES DE DÉCLENCHEMENT                                  │
│                                                                       │
│   Mode 1 — CLI local      make init-dvc / make ingest-dvc / make train│
│   Mode 2 — API            curl / Swagger → nginx → gateway → api-*    │
│            └── EXECUTION_MODE=cli    : subprocess (dans le container) │
│            └── EXECUTION_MODE=docker : docker-in-docker (DID)         │
│   Mode 3 — Airflow batch  DAG → DockerOperator → containers éphémères │
└───────────────────────────┬───────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────────┐
│                    CLIENT (curl / Swagger / Streamlit / Airflow)   │
└─────────────────────────┬──────────────────────────────────────────┘
                          │ HTTPS (TLS auto-signé)
                          ▼
┌────────────────────────────────────────────────────────────────────┐
│                       NGINX (Reverse Proxy)                        │
│  • Terminaison TLS                                                 │
│  • Routage vers l'API Gateway                                      │
└─────────────────────────┬──────────────────────────────────────────┘
                          │ HTTP interne
                          ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                         API GATEWAY (FastAPI)                                 │
│  • Authentification OAuth2 / Bearer Token                                     │
│  • Routage vers les services internes                                         │
└──────┬──────────────┬────────────────────┬──────────────┬──────────────┬──────┘
       │              │                    │              │              │
       ▼              ▼                    ▼              ▼              ▼
┌──────────┐  ┌──────────────┐  ┌──────────────────┐  ┌────────┐  ┌──────────────┐
│  Ingest  │  │    Train     │  │     Predict      │  │  Init  │  │   Monitor    │
│ Service  │  │   Service    │  │    Service       │  │Service │  │   Service    │
│(FastAPI) │  │              │  │                  │  │        │  │              │
│          │  │ • Pipeline   │  │ • Chargement     │  │• Initialisation
       │  │ • Evidently  │                                raw data
│ • Merge  │  │   complète   │  │   modèle MLflow  │  │        │  │ • Drift      │
│  datasets│  │ • MLflow     │  │ • Inférence      │  │        │  │   report     │
│          │  │   tracking   │  │ • Top-K résult.  │  │        │  │ • MLflow     │
└──────────┘  └──────────────┘  └──────────────────┘  └────────┘  └──────────────┘
       │              │
       └──────────────┼──────────────────────────────
                      │
           ┌──────────┴──────────┐
           ▼                     ▼
┌────────────────────┐  ┌──────────────────────────────┐
│  Stockage local    │  │  MLflow + DagsHub            │
│  (volumes Docker)  │  │  • Tracking expériences      │
│  • data/           │  │  • Métriques / artefacts     │
│  • models/         │  │  • DVC remote (données)      │
└────────────────────┘  └──────────────────────────────┘
                                    │
             ┌──────────────────────┤
             ▼                      ▼
┌──────────────────┐     ┌────────────────────────┐
│   Prometheus     │     │  Grafana               │
│  • Métriques     │     │  • Dashboards          │
└──────────────────┘     └────────────────────────┘
```

---

## Modes d'exécution

Le projet expose **trois modes d'exécution** qui diffèrent par leur déclencheur et leur transport, mais partagent la même logique métier via les modules `pipelines/` et `modules/`.

### Vue d'ensemble

```
┌─────────────────┬────────────────────┬────────────────────────────────────────┐
│ Mode            │ Déclencheur        │ Transport                              │
├─────────────────┼────────────────────┼────────────────────────────────────────┤
│ CLI local       │ make / terminal    │ subprocess direct (hors Docker)        │
│ API             │ humain / Swagger   │ HTTP → nginx → gateway → api-*         │
│ Airflow batch   │ scheduler / manuel │ DockerOperator → containers éphémères  │
└─────────────────┴────────────────────┴────────────────────────────────────────┘
```

```
                     ┌─────────────────────────────────────────┐
                     │           Logique métier partagée        │
                     │  pipelines/ · modules/ · monitoring/     │
                     └──────────┬──────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────────┐
        ▼                       ▼                           ▼
┌──────────────────┐  ┌──────────────────────┐  ┌──────────────────────────┐
│  Mode 1 — CLI    │  │   Mode 2 — API        │  │   Mode 3 — Airflow       │
│                  │  │                       │  │                          │
│  main.py (Typer) │  │  FastAPI (api-*)       │  │  DAG rakuten_ml_pipeline │
│  make ingest-dvc │  │  POST /ingest, /train  │  │  DockerOperator          │
│  make train-dvc  │  │                       │  │  → containers éphémères  │
└──────────────────┘  └──────────┬────────────┘  └──────────────────────────┘
  subprocess direct               │
  (machine locale)     ┌──────────┴──────────┐
                       ▼                     ▼
             ┌──────────────────┐  ┌─────────────────────┐
             │ EXECUTION_MODE   │  │  EXECUTION_MODE      │
             │     = cli        │  │     = docker         │
             │                  │  │                      │
             │ subprocess       │  │  docker exec         │
             │ → dvc/git        │  │  → dvc-runner        │
             │   (in container) │  │  → git-runner        │
             └──────────────────┘  └─────────────────────┘
```

---

### Mode 1 — CLI local

Exécution directe sur la machine de développement, sans Docker. Utilisé pour le développement et les tests rapides.

```bash
# Ingestion d'un batch
python mlops_rakuten/main.py ingest data/raw/rakuten/seeds/rakuten_batch_0001.csv

# Entraînement
python mlops_rakuten/main.py train

# Via Makefile
make ingest-dvc CSV=rakuten_batch_0001.csv
make train-dvc
```

**Transport :** `main.py` (Typer) appelle directement `utils/cli.py`, qui exécute `dvc` et `git` via `subprocess`.

---

### Mode 2 — API Docker

Deux sous-modes contrôlés par la variable `EXECUTION_MODE` :

#### EXECUTION_MODE=cli

Les services FastAPI (`api-ingest`, `api-train`) exécutent DVC et Git via **subprocess** directement dans leur container. Aucun container supplémentaire.

```bash
make docker-up-cli
```

Lance : `nginx` + `gateway` + `api-ingest` + `api-train` + `api-predict` + `api-monitor`
      + `prometheus` + `grafana` + `nginx_exporter` + `node-exporter`

#### EXECUTION_MODE=docker (docker-in-docker)

Les services FastAPI délèguent DVC et Git à des containers dédiés (`dvc-runner`, `git-runner`) via `docker exec`.

```bash
make docker-up-docker
```

Lance : tous les services + `dvc-runner` + `git-runner`

```
EXECUTION_MODE=cli                       EXECUTION_MODE=docker
────────────────────────────────────     ────────────────────────────────────
api-ingest                               api-ingest
  └── subprocess → dvc/git                 └── docker exec → rakuten-dvc-runner
        (dans le container api-ingest)                      → rakuten-git-runner
```

**Appels API une fois les containers lancés :**

```bash
make api-token                              # récupérer un JWT
make api-init                               # initialiser le dataset seed
make api-ingest CSV=rakuten_batch_0001.csv  # ingérer un batch
make api-train                              # entraîner
make api-predict TEXT="Vélo électrique" TOPK=3
```

---

### Mode 3 — Airflow batch

Orchestration automatisée du pipeline complet via le DAG `rakuten_ml_pipeline`.

**Transport :** contrairement aux modes 1 et 2, le worker Airflow ne fait pas de subprocess directement. Il utilise le `DockerOperator`, qui lance un **container Docker éphémère** pour chaque tâche. C'est à l'intérieur de ce container que `python -m mlops_rakuten.main ...` s'exécute.

```
Airflow worker (Celery)
    └── DockerOperator
          ├── tâche ingest   → container éphémère (image api-ingest)
          │                        └── subprocess → main.py ingest
          ├── tâche train    → container éphémère (image api-train)
          │                        └── subprocess → main.py train
          ├── tâche validate → container éphémère (image api-predict)
          │                        └── subprocess → main.py predict
          └── tâche drift    → container éphémère (image api-ingest)
                                   └── run_drift_report() [appel Python direct]
```

Chaque container éphémère monte les mêmes **volumes nommés** que la stack principale (`rakuten_models`, `rakuten_reports`, etc.) ainsi qu'un **bind mount** sur le répertoire du projet et sur `~/.ssh`.

#### Bind mounts — configuration requise

Les mounts hôte sont résolus dynamiquement dans le DAG :

```python
# PROJECT_ROOT : priorité à la variable d'env, sinon déduit depuis __file__
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parent.parent))

# SSH_DIR : priorité à la variable d'env, sinon $HOME/.ssh
SSH_DIR = os.environ.get("SSH_DIR", str(Path.home() / ".ssh"))
```

Pour forcer un chemin spécifique (machine partagée, CI), ajouter dans `.env` ou dans la config Airflow :

```dotenv
PROJECT_ROOT=/chemin/absolu/vers/nov25cmlops_rakuten
SSH_DIR=/home/mon_user/.ssh
```

> Ces chemins doivent être valides sur **l'hôte Docker** (là où tourne le daemon), pas dans le container Airflow.

#### DAG — rakuten_ml_pipeline

```
Déclenchement : manuel (schedule_interval=None)
→ En prod, remplacer par un FileSensor sur data/uploads/

ingest → train → validate → drift_monitor
                               ↑
                    trigger_rule="all_done"
                    (tourne même si validate échoue)
```

| Tâche | Image | Commande | Notes |
|---|---|---|---|
| `ingest` | `api-ingest` | `main.py ingest <csv>` | Itère sur tous les CSV de `data/uploads/`, skip si vide |
| `train` | `api-train` | `main.py train` | dvc repro + sync Git/MLflow |
| `validate` | `api-predict` | `main.py predict <texte>` | Smoke-test du modèle `@production` |
| `drift_monitor` | `api-ingest` | `run_drift_report()` | Rapport Evidently, lié au run MLflow courant |

#### Démarrage Airflow

```bash
# 1. Initialiser la base Airflow (une seule fois)
make airflow-init

# 2. Lancer la stack complète avec Airflow
make docker-up-batch

# 3. Interface Airflow
make airflow-ui    # → http://localhost:8080  (admin/admin)
``` 

### Commits Git par mode

Chaque mode produit des commits identifiables dans l'historique Git :
```
git log --oneline

a3f1c2e  Airflow:train          — model v26, f1_macro=0.7750  ← automatique (Airflow)
58ee89f  Airflow:ingest         — batch=rakuten_batch_0003     ← automatique (Airflow)
3a2f1c4  Docker-in-Docker:train — model v25, f1_macro=0.7697  ← manuel (API / Docker-DID)
9b4e2d1  Docker-in-Docker:ingest— batch=rakuten_batch_0002     ← manuel (API / Docker-DID)
3a2f1c4  Docker-CLI:train       — model v25, f1_macro=0.7697  ← manuel (API / CLI)
9b4e2d1  CLI:ingest             — batch=rakuten_batch_0002     ← manuel (API / CLI)
3a2f1c4  CLI-local:train        — model v25, f1_macro=0.7697  ← local (main.py / make)
9b4e2d1  CLI-local:ingest       — batch=rakuten_batch_0002     ← local (main.py / make)
```

| Préfixe de commit       | Mode                          | Déclencheur        |
|-------------------------|-------------------------------|--------------------|
| `CLI-local:train`       | Mode 1 — local                | make / terminal    |
| `Docker-CLI:train`      | Mode 2 — subprocess container | curl / Swagger     |
| `Docker-in-Docker:train`| Mode 2 — docker-in-docker     | curl / Swagger     |
| `Airflow:ingest`        | Mode 3 — batch                | scheduler / cron   |
| `Airflow:train`         | Mode 3 — batch                | scheduler / cron   |

### Résumé des profiles Docker

| Commande Makefile         | Services                               | Usage                      |
|---------------------------|----------------------------------------|----------------------------|
| `make docker-up-cli`      | EXECUTION_MODE=cli                     | API                        |
| `make docker-up-docker`   | EXECUTION_MODE=docker                  | Démo docker-in-docker      |
| `make docker-up-batch`    | stack cli + airflow (profile batch)    | Batch automatisé           |

---

## Project Organization

```
├── Makefile                   <- Commandes utilitaires
├── README.md
├── .env                       <- Variables d'environnement (non versionné)
├── Dockerfile                 <- Image commune à tous les services
├── entrypoint.sh              <- Init Git/DVC/SSH au démarrage des containers
├── docker-compose.yml         <- Orchestration (tous modes)
│
├── data/
│   ├── interim/               <- Données intermédiaires
│   ├── processed/             <- Datasets finaux pour l'entraînement
│   └── raw/
│       └── rakuten/
│           └── seeds/         <- Batches CSV (rakuten_batch_0001.csv … 0010.csv)
│
├── deployments/
│   ├── airflow/
│   │   ├── docker-compose.airflow.yml
│   │   └── dags/
│   │       └── rakuten_batch_pipeline.py
│   ├── certs/                 <- Certificats TLS
│   ├── nginx/
│   │   └── nginx.conf
│   └── prometheus/
│       └── prometheus.yml
│
├── models/                    <- Modèles entraînés
├── reports/                   <- Métriques et rapports
├── logs/                      <- Logs applicatifs
│
├── tests/
│
└── mlops_rakuten/
    ├── main.py                <- Point d'entrée CLI
    ├── monitoring/            ← Logique Evidently
    ├── services/              <- API FastAPI (gateway, ingest, train, predict)
    ├── auth/                  <- OAuth2
    ├── config/                <- config.yml, entités, constantes
    ├── modules/               <- Logique métier
    ├── pipelines/             <- Pipelines orchestrant les modules
    └── utils/
        ├── cli.py             <- Transport subprocess (Modes 1 et 3)
        ├── docker.py          <- Transport docker exec (Mode 2 docker)
        └── sync_core.py       <- Logique Git/DVC commune
```

---

## Installation

### 1. Environnement Python

```bash
uv --version   # vérifier uv, sinon : https://docs.astral.sh/uv/
make create_environment
source .venv/bin/activate
make requirements
python -c "import pandas, typer, mlops_rakuten; print('OK')"
```

### 2. Configuration des données

```bash
mkdir -p data/raw/rakuten

# Copier les fichiers :
# product_categories.csv  →  data/raw/
# X_train_update.csv      →  data/raw/rakuten/
# Y_train_CVw08PX.csv     →  data/raw/rakuten/
```

### 3. Fichier .env

```dotenv
# DagsHub / MLflow
DAGSHUB_USER=shiff-oumi
DAGSHUB_REPO=nov25cmlops_rakuten_dag
DAGSHUB_TOKEN=<token_dagshub>
MLFLOW_TRACKING_URI=https://dagshub.com/shiff-oumi/nov25cmlops_rakuten_dag.mlflow
MLFLOW_TRACKING_USERNAME=shiff-oumi
MLFLOW_TRACKING_PASSWORD=<token_dagshub>

# Git / GitHub
GIT_AUTHOR_NAME=Rakuten MLOps
GIT_AUTHOR_EMAIL=mlops@rakuten.local
GITHUB_USER=shiff-oumi
GITHUB_TOKEN=<token_github>
```
## Contribuer — configuration Git/SSH requise

Le pipeline commite et pousse automatiquement vers GitHub à chaque ingestion et entraînement. Chaque personne doit configurer son environnement une seule fois avant de lancer le projet.

### Clé SSH

La clé SSH est montée dans les containers via le volume `~/.ssh:/root/.ssh`. Elle doit être active et reconnue par GitHub :

Commencer par vérifier les clés existantes :
```bash
ls ~/.ssh/
# → id_github, id_github.pub, id_rsa, known_hosts...
```

Si `id_github` est déjà présente, vérifier qu'elle est reconnue par GitHub :
```bash
ssh -i ~/.ssh/id_github -T git@github.com
# → Hi ! You've successfully authenticated.
```

Si `id_github` n'existe pas encore :
```bash
# Créer la clé dédiée
ssh-keygen -t ed25519 -f ~/.ssh/id_github -C "mlops@rakuten"

# Copier la clé publique → GitHub > Settings > SSH Keys > New SSH Key
cat ~/.ssh/id_github.pub
```

### Fork et remote

Le push automatique cible le remote `origin` par défaut (configuré dans `sync_git_dvc` de `utils/sync_core.py`). Il faut bien vérifier que la bonne cible est bien paramétré.

```bash
git remote -v   # vérifier : origin → repo principal
```

### Branche de travail

La branche est détectée automatiquement via `git rev-parse --abbrev-ref HEAD` — le container lit la branche active depuis le volume `.git/` monté et pousse dessus directement. Il suffit d'être sur la bonne branche avant de lancer :

```bash
git checkout -b feature/<nom>
make docker-up-cli    # le container voit ta branche courante
make api-train        # → commit + push automatique sur feature/<nom>
```

### Checklist rapide

| Étape | Commande |
|-------|----------|
| Clé SSH active | `ssh -i ~/.ssh/id_github -T git@github.com` |
| Fork ajouté | `git remote add myfork git@github.com:<user>/...` |
| Branche créée | `git checkout -b feature/<nom>` |
| Variables .env | copier `.env.example` → `.env` et renseigner les tokens |
| (Airflow) PROJECT_ROOT | renseigner dans `.env` si la détection auto échoue |

---

## Structure du pipeline de données

```
data/raw/  (X_train, Y_train, categories)
    │  dvc add
    ▼
  seed  →  data/raw/rakuten/seeds/  (10 batches × ~1000 lignes)
    │
    ▼  (après make api-ingest ou make api-init)
  data/interim/rakuten_train.csv    (dataset courant, tracké DVC)
    │  dvc repro
    ▼
  preprocess  →  data/interim/preprocessed_dataset.csv
    ▼
  transform   →  data/processed/  (TF-IDF, splits train/val)
    ▼
  train       →  models/text_classifier.pkl  +  MLflow (alias: pending)
    ▼
  evaluate    →  reports/  +  promotion MLflow (pending → production si +1% F1)
```

---

## Sécurité et Gateway

```
Internet  (HTTPS 443)
   ▼
NGINX  →  terminaison TLS, redirect HTTP→HTTPS
   ▼
API Gateway (FastAPI)  →  POST /token, validation Bearer Token, routing
   ▼
api-ingest / api-train / api-predict
```

**Génération du certificat auto-signé :**

```bash
mkcert -key-file deployments/certs/nginx.key \
       -cert-file deployments/certs/nginx.crt \
       localhost 127.0.0.1 ::1
```

**Utilisateurs configurés :**

| Utilisateur | Rôle  | Mot de passe |
|-------------|-------|--------------|
| jane        | user  | password     |
| john        | user  | password     |
| julien      | admin | admin123     |
| claudia     | admin | admin456     |
| samuel      | admin | admin789     |

---

## Suivi d'expériences et versioning

### Ce qui est versionné où

| Objet                            | Outil  | Destination        |
|----------------------------------|--------|--------------------|
| Données brutes (CSV)             | DVC    | DagsHub S3         |
| Données intermédiaires (npz/npy) | DVC    | DagsHub S3         |
| `mlflow_run_metadata.json`       | DVC    | DagsHub S3         |
| Code source                      | Git    | GitHub             |
| `.dvc`, `dvc.lock`, `dvc.yaml`   | Git    | GitHub             |
| Hyperparamètres, métriques       | MLflow | DagsHub MLflow     |
| Modèle                           | MLflow | DagsHub S3         |
| Vectorizer, LabelEncoder         | MLflow | DagsHub S3         |
| Alias modèle (pending, prod…)    | MLflow | DagsHub MLflow     |

### Système d'alias MLflow

```
text_classifier_tfidf_m
├── @pending     → Nouvellement entraîné, en attente d'évaluation
├── @production  → Meilleur modèle validé, utilisé par Prediction
└── @archived    → Ancienne version, conservée pour traçabilité
```

Logique de promotion :

```
Après training  →  alias "pending" attribué

Après évaluation
  ├── Pas de @production → promotion directe
  └── @production existe
        ├── val_f1_macro (nouveau) ≥ val_f1_macro (prod) + 0.01 → promotion
        └── Amélioration insuffisante → pas de changement
```

### Configuration DVC remote

```bash
dvc remote add origin s3://dvc
dvc remote add origin https://dagshub.com/lioradatascientest/nov25cmlops_rakuten.s3
dvc remote modify origin --local access_key_id your_token
dvc remote modify origin --local secret_access_key your_token

dvc remote default origin

dvc push   # → DagsHub S3
dvc pull   # ← DagsHub S3
```

---

## Containerisation via Docker

### Services

| Service       | Rôle                                    | Port    | Profile    |
|---------------|-----------------------------------------|---------|------------|
| `nginx`       | Reverse proxy + TLS                     | 443     | toujours   |
| `gateway`     | Auth + routage                          | interne | toujours   |
| `api-ingest`  | Ingestion datasets                      | interne | toujours   |
| `api-train`   | Entraînement                            | interne | toujours   |
| `api-predict` | Inférence                               | interne | toujours   |
| `prometheus`  | Métriques                               | 9090    | toujours   |
| `grafana`     | Dashboards                              | 3000    | toujours   |
| `streamlit`   | Interface Graphique                     | 8501    | toujours   |
| `api-monitor` | Drift monitoring Evidently              | interne | toujours   |
| `dvc-runner`  | DVC isolé (docker-in-docker)            | interne | `docker`   |
| `git-runner`  | Git isolé (docker-in-docker)            | interne | `docker`   |
| `airflow-*`   | Orchestration batch                     | 8080    | `airflow`  |

### SSH et commits automatiques

```yaml
# docker-compose.yml
volumes:
  - ~/.ssh:/root/.ssh   # clé SSH montée dans les containers
```

---

## Lancer l'application avec Docker

### Mode CLI

```bash
make docker-up-cli

make api-init
make api-ingest CSV=rakuten_batch_0001.csv
make api-train
make api-predict TEXT="Vélo électrique pliable" TOPK=3
```

`make docker-up-cli` démarre aussi automatiquement la stack de monitoring
`Prometheus + Grafana`, ainsi que `nginx_exporter` et `node-exporter`.
Les dashboards Grafana et la datasource Prometheus sont provisionnés
automatiquement au démarrage.

### Mode Docker-in-Docker

```bash
make docker-up-docker

make api-ingest CSV=rakuten_batch_0002.csv
make api-train
```

### Swagger

```bash
make swagger   # → https://localhost/docs
```

---

## Orchestration batch avec Airflow

### Architecture du Mode 3

Le Mode 3 se distingue fondamentalement des modes 1 et 2 par son transport. Le worker Airflow (Celery) ne fait **pas** de subprocess directement : il instancie un `DockerOperator` qui lance un container Docker éphémère pour chaque tâche. C'est **à l'intérieur** de ce container que le code Python s'exécute.

```
Airflow worker (Celery)
    └── DockerOperator (via /var/run/docker.sock)
          │
          ├── tâche ingest    → container éphémère, image api-ingest
          │                         └── subprocess → main.py ingest <csv>
          │                               └── utils/cli.py → dvc, git
          │
          ├── tâche train     → container éphémère, image api-train
          │                         └── subprocess → main.py train
          │                               └── utils/cli.py → dvc repro, git, MLflow
          │
          ├── tâche validate  → container éphémère, image api-predict
          │                         └── subprocess → main.py predict <texte>
          │                               └── MLflow → chargement @production
          │
          └── tâche drift     → container éphémère, image api-ingest
                                    └── run_drift_report() [appel Python direct]
                                          └── Evidently + MLflow logging
```

Chaque container éphémère est supprimé après exécution (`auto_remove="success"`). Les données persistent via les volumes nommés Docker partagés avec la stack principale.

### DAG — rakuten_ml_pipeline

```
Déclenchement : manuel (schedule_interval=None)
→ En prod : remplacer par un FileSensor sur data/uploads/

ingest ──► train ──► validate ──► drift_monitor
                                       ↑
                           trigger_rule="all_done"
                           tourne même si validate échoue
```

### Bind mounts

Chaque container éphémère a besoin d'accéder au code source et aux clés SSH sur l'hôte. Ces chemins sont résolus dynamiquement :

```python
# Déduit depuis l'emplacement du DAG lui-même
PROJECT_ROOT = os.environ.get("PROJECT_ROOT",
    str(Path(__file__).resolve().parent.parent))

# $HOME/.ssh par défaut
SSH_DIR = os.environ.get("SSH_DIR",
    str(Path.home() / ".ssh"))
```

Si la détection automatique ne convient pas (utilisateur différent, CI, chemin non standard), définir dans `.env` :

```dotenv
PROJECT_ROOT=/chemin/absolu/vers/nov25cmlops_rakuten
SSH_DIR=/home/mon_user/.ssh
```

> Ces chemins doivent être valides sur l'**hôte Docker**, pas dans le container Airflow.

### Démarrage

```bash
make airflow-init      # initialise la DB Airflow (une seule fois)
make docker-up-batch   # lance stack cli + Airflow
make airflow-ui        # → http://localhost:8080  (admin/admin)
```

### Commandes Airflow

```bash
make airflow-trigger          # déclencher manuellement
make airflow-runs             # voir les derniers runs
make airflow-set-batch N=3    # configurer le prochain batch
make airflow-ui               # ouvrir l'interface
```

---

### DAG drift_monitor

Le drift monitoring est intégré comme étape finale du pipeline principal :

```
ingest >> train >> predict >> drift_monitor
```

La tâche `drift_monitor` utilise un `DockerOperator` qui appelle directement
`mlops_rakuten.monitoring.drift_report` — même logique que le endpoint `POST /drift`.

```python
drift_monitor = DockerOperator(
    task_id="drift_monitor",
    image="nov25cmlops_rakuten-gateway:latest",
    trigger_rule="all_done",   # non bloquant — s'exécute même si predict échoue
    ...
)

ingest >> train >> predict >> drift_monitor
```

Le paramètre `trigger_rule="all_done"` garantit que le monitoring tourne
toujours en fin de pipeline, même en cas d'échec partiel des étapes précédentes.


### Commandes Airflow

```bash
make airflow-trigger          # déclencher manuellement
make airflow-runs             # voir les derniers runs
make airflow-set-batch N=3    # configurer le prochain batch
make airflow-ui               # ouvrir l'interface
```

---

## Monitoring

Le projet intègre trois couches de monitoring complémentaires :

```
MLflow + Evaluation   → performance modèle (accuracy, f1, classification report)
Evidently             → drift données (distribution, qualité, dérive texte)
Prometheus / Grafana  → métriques infra (latence HTTP, CPU, requêtes nginx)
```

- **Prometheus** : métriques système et applicatives → [http://localhost:9090](http://localhost:9090)
- **Grafana** : dashboards → [http://localhost:3000](http://localhost:3000) (admin/admin)

Au démarrage via `make docker-up-cli`, `make docker-up-docker` ou `make docker-up`,
la stack monitoring est lancée automatiquement. Grafana charge aussi
automatiquement la datasource Prometheus et les dashboards versionnés dans
`deployments/grafana/`.

---

### Drift Monitoring — Evidently

Le service `api-monitor` expose un endpoint de drift monitoring basé sur [Evidently](https://www.evidentlyai.com/).
Il compare le dataset de référence (`rakuten_train.csv`, 74k lignes) contre le dernier batch ingéré.

#### Architecture

```
POST /drift (via Gateway)
    ↓
api-monitor
    ↓ détecte automatiquement le dernier batch
    ├── uploads/rakuten_batch_*.csv  (prioritaire — ingest récent)
    └── raw/rakuten/seeds/rakuten_batch_*.csv  (fallback)
    ↓
drift_report.py (Evidently)
    ├── DataDriftPreset()              → dérive distribution features
    ├── TargetDriftPreset()            → dérive distribution catégories
    └── ColumnDistributionMetric()     → distribution par catégorie Rakuten
    ↓
    ├── rapport HTML → reports/drift/drift_report.html (volume rakuten_reports)
    └── métriques   → MLflow experiment "train_rakuten_model_mlflow"
                      tags: train_run_id, model_version, current_batch
```

#### Métriques exposées

| Métrique | Description | Type |
|---|---|---|
| `drift_share` | Part de colonnes avec drift détecté (0.0 → 1.0) | float |
| `drift_count` | Nombre de colonnes driftées | float |
| `dataset_drift` | Drift global détecté (0 ou 1) | float |
| `current_batch` | Nom du batch comparé | tag MLflow |
| `train_run_id` | Run MLflow du dernier entraînement | tag MLflow |

#### Endpoints

| Méthode | Endpoint | Rôle requis | Description |
|---|---|---|---|
| `POST` | `/drift` | admin | Lance le rapport Evidently |
| `GET` | `/drift/report` | user | Retourne le rapport HTML |

#### Mapping des catégories

Les codes `prdtypecode` sont traduits en noms lisibles pour le rapport :

| Code | Catégorie |
|---|---|
| 10 | Livres |
| 40 | Films |
| 1140 | Jouets |
| 1280 | Peluches |
| 2705 | Livres jeunesse |
| 2905 | Jeux vidéo |
| ... | (voir `data/raw/product_categories.csv`) |

Le mapping est chargé dynamiquement depuis `data/raw/product_categories.csv`.

#### Intégration MLflow

Le run drift est loggé dans le **même experiment** que l'entraînement, avec un tag `train_run_id` qui le lie au dernier run d'entraînement. Le rapport HTML est également archivé comme artifact MLflow sous `drift_reports/drift_report.html`.

```
DagsHub MLflow → train_rakuten_model_mlflow
    ├── run: training        run_id=ea5512d...   ← entraînement
    └── run: drift_monitoring                    ← drift
          tags:
            train_run_id  = ea5512d...
            model_version = 39
            current_batch = rakuten_batch_0008.csv
```
---

## Tests

```bash
pytest tests/                           # tous les tests
pytest tests/test_model_trainer.py      # module spécifique
```

| Fichier                      | Couverture                        |
|------------------------------|-----------------------------------|
| `test_pipelines.py`          | Pipeline complète (intégration)   |
| `test_data_ingestion.py`     | Fusion des datasets               |
| `test_data_preprocessing.py` | Nettoyage                         |
| `test_data_transformation.py`| TF-IDF + split                    |
| `test_model_trainer.py`      | Entraînement                      |
| `test_model_evaluation.py`   | Métriques et matrice de confusion |
| `test_prediction.py`         | Inférence et format de sortie     |

---

## Commandes Makefile

### Stack Docker

| Commande                | Description                                          |
|-------------------------|------------------------------------------------------|
| `make docker-up-cli`    | Stack CLI (EXECUTION_MODE=cli, sans runners)         |
| `make docker-up-docker` | Stack Docker-in-Docker (avec runners)                |
| `make docker-up-batch`  | Stack CLI + Airflow                                  |
| `make docker-down`      | Arrêt (volumes conservés)                            |
| `make docker-down-v`    | Arrêt + suppression des volumes                      |
| `make docker-ps`        | État des containers                                  |
| `make docker-logs`      | Logs en temps réel                                   |
| `make docker-mode`      | Affiche EXECUTION_MODE actif                         |

### API

| Commande                              | Description                    |
|---------------------------------------|--------------------------------|
| `make api-health`                     | Santé de tous les services     |
| `make api-token`                      | Récupère un JWT                |
| `make api-init`                       | Initialise le dataset seed     |
| `make api-init-force`                 | Réinitialise (force)           |
| `make api-ingest CSV=<fichier>`       | Ingère un batch CSV            |
| `make api-train`                      | Lance l'entraînement           |
| `make api-predict TEXT=<t> TOPK=<n>`  | Prédiction                     |
| `make api-info`                       | Informations modèle actif      |
| `make api-drift`                      | Lance le rapport Evidently     |
| `make api-drift-report`               | Retourne le fichier HTML       |

### Airflow

| Commande                      | Description                        |
|-------------------------------|------------------------------------|
| `make airflow-trigger`        | Déclenche le DAG manuellement      |
| `make airflow-runs`           | Liste les derniers runs            |
| `make airflow-set-batch N=3`  | Configure le prochain batch        |
| `make airflow-ui`             | Ouvre http://localhost:8080        |

### Local

| Commande                  | Description                           |
|---------------------------|---------------------------------------|
| `make create_environment` | Crée le venv Python avec `uv`         |
| `make requirements`       | Installe les dépendances              |
| `make init-dvc`           | Initialise le dataset seed (Mode 1)   |
| `make init_force-dvc`     | Réinitialise (force, Mode 1)          |
| `make ingest-dvc CSV=<f>` | Ingestion locale (Mode 1)             |
| `make train-dvc`          | Entraînement local (Mode 1)           |
| `make predict-dvc TEXT=<t> TOPK=<n>` | Prédiction locale (Mode 1) |
| `make swagger`            | Ouvre https://localhost/docs          |

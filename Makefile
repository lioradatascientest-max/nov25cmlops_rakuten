#################################################################################
# GLOBALS
#################################################################################

PROJECT_NAME := nov25cmlops_rakuten
PYTHON_VERSION := 3.12
PYTHON_INTERPRETER := python

# Docker compose command (supports docker-compose v1 or docker compose v2)
COMPOSE_CMD := $(shell \
	if command -v docker-compose >/dev/null 2>&1; then \
		echo docker-compose; \
	else \
		echo docker compose; \
	fi \
)

# Services
SVC_NGINX   := nginx
SVC_GATEWAY := gateway
SVC_PREDICT := api-predict
SVC_TRAIN   := api-train
SVC_INGEST  := api-ingest
SVC_DVC	 := dvc-runner
SVC_GIT	 := git-runner

# Variables par défaut
CSV       :=
SEEDS_DIR ?= data/raw/rakuten/seeds
TEXT      ?= Super aspirateur sans fil Dyson
TOPK      ?= 5
API_URL   ?= https://localhost
SVC       ?= api-train

# Auth JWT — récupéré automatiquement pour les appels curl
ADMIN_USER ?= claudia
ADMIN_PASS ?= admin456
TOKEN = $(shell curl -s -k -X POST "$(API_URL)/token" \
	-d "username=$(ADMIN_USER)&password=$(ADMIN_PASS)" \
	| python -c "import sys,json; print(json.load(sys.stdin).get('access_token',''))" 2>/dev/null)

#################################################################################
# PYTHON (LOCAL)
#################################################################################

## Install Python dependencies (local)
.PHONY: requirements
requirements:
	uv pip install -r requirements.txt
	uv pip install -r requirements-dev.txt

## Create local venv
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo "Activate with: source ./.venv/bin/activate"

## Lint using ruff
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

## Run tests
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest tests

## Clean python caches
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Fix permissions on data/ and models/ after Docker root usage
.PHONY: fix-permissions
fix-permissions:
	@sudo chown -R $(USER):$(USER) data/ models/ reports/ 2>/dev/null || true
	@echo "Permissions fixed (owner=$(USER))"

#################################################################################
# DVC & DAGHUB     			                                       				#
#################################################################################

## Initialize DVC config.local with credentials 
.PHONY: dvc-credentials
dvc-credentials:
	@echo "Création du fichier .dvc/config.local avec les identifiants DagsHub"
	@read -p "Enter DagsHub Access Key ID: " ACCESS_KEY; \
	read -p "Enter DagsHub Secret Access Key: " SECRET_KEY; \
	dvc remote modify origin --local access_key_id $$ACCESS_KEY; \
	dvc remote modify origin --local secret_access_key $$SECRET_KEY;
	@echo ".dvc/config.local crée avec succès."

## Test DVC connection to Dagshub s3 remote storage
.PHONY: dvc-test
dvc-test:
	@echo "Test connexion DVC vers DagsHub..."
	@dvc status && echo "Connected to DagsHub" || echo "Connection failed"	


#################################################################################
# CLI version for debugging, can be used inside containers or locally			
#################################################################################
.PHONY: init-dvc
init-dvc:
	$(PYTHON_INTERPRETER) mlops_rakuten/main.py init

.PHONY: init_force-dvc
init_force-dvc:
	$(PYTHON_INTERPRETER) mlops_rakuten/main.py init --force


#make ingest-dvc CSV=data/uploads/rakuten_batch_008.csv
.PHONY: ingest-dvc
ingest-dvc:
	$(PYTHON_INTERPRETER) mlops_rakuten/main.py ingest $(CSV)

.PHONY: train-dvc
train-dvc:
	$(PYTHON_INTERPRETER) mlops_rakuten/main.py train


#make predict-dvc TEXT="Vélo électrique pliable" TOPK=3
.PHONY: predict-dvc
predict-dvc:
	$(PYTHON_INTERPRETER) mlops_rakuten/main.py predict "$(TEXT)" --top-k $(TOPK)



#################################################################################
# Mode API CURL (HTTP → nginx → gateway → api-*)
# Requiert : make docker-up-cli ou make docker-up-docker (selon le profil choisi utilisation de subprocess CLI ou Docker exec pour les services)
# Auth JWT récupérée automatiquement via /token
# Simule les clics sur les boutons de Swagger UI, mais en ligne de commande avec curl
# Usage : make api-init / make api-ingest CSV=... / make api-train
#################################################################################

## curl : GET /health — vérifie la disponibilité de l'API
.PHONY: api-health
api-health:
	@echo "GET $(API_URL)/health"
	@curl -s -k "$(API_URL)/health" | python -m json.tool

## curl : GET /token — affiche le token JWT admin
.PHONY: api-token
api-token:
	@echo "POST $(API_URL)/token"
	@curl -s -k -X POST "$(API_URL)/token" \
		-d "username=$(ADMIN_USER)&password=$(ADMIN_PASS)" \
		| python -m json.tool

## curl : POST /init (auth admin requise)
.PHONY: api-init
api-init:
	@echo "POST $(API_URL)/init"
	@curl -s -k -X POST "$(API_URL)/init" \
		-H "Authorization: Bearer $(TOKEN)" \
		| python -m json.tool

## curl : POST /init?force=true (auth admin requise)
.PHONY: api-init-force
api-init-force:
	@echo "POST $(API_URL)/init?force=true"
	@curl -s -k -X POST "$(API_URL)/init?force=true" \
		-H "Authorization: Bearer $(TOKEN)" \
		| python -m json.tool

## curl : POST /ingest  : make api-ingest CSV=rakuten_batch_0001.csv (auth admin requise)
.PHONY: api-ingest
api-ingest:
	@test -n "$(CSV)"              || (echo "Usage: make api-ingest CSV=rakuten_batch_NNNN.csv" && exit 1)
	@test -f "$(SEEDS_DIR)/$(CSV)" || (echo "File not found: $(SEEDS_DIR)/$(CSV)" && exit 1)
	@echo "POST $(API_URL)/ingest — $(CSV)"
	@curl -s -k -X POST "$(API_URL)/ingest" \
		-H "Authorization: Bearer $(TOKEN)" \
		-F "file=@$(SEEDS_DIR)/$(CSV);type=text/csv" \
		| python -m json.tool

## curl : POST /train (auth admin requise)
.PHONY: api-train
api-train:
	@echo "POST $(API_URL)/train"
	@curl -s -k -X POST "$(API_URL)/train" \
		-H "Authorization: Bearer $(TOKEN)" \
		| python -m json.tool

## curl : POST /predict → make api-predict TEXT="Vélo électrique pliable" TOPK=3
.PHONY: api-predict
api-predict:
	@echo "POST $(API_URL)/predict — $(TEXT)"
	@curl -s -k -X POST "$(API_URL)/predict" \
		-H "Authorization: Bearer $(TOKEN)" \
		-H "Content-Type: application/json" \
		-d "{\"designation\": \"$(TEXT)\", \"top_k\": $(TOPK)}" \
		| python -m json.tool

## curl : GET /info — infos modèle chargé
.PHONY: api-info
api-info:
	@curl -s -k "$(API_URL)/info" \
		-H "Authorization: Bearer $(TOKEN)" \
		| python -m json.tool

## curl : POST /reload — recharge le modèle MLflow manuellement
.PHONY: api-reload
api-reload:
	@curl -s -k -X POST "$(API_URL)/reload" \
		-H "Authorization: Bearer $(TOKEN)" \
		| python -m json.tool


# curl : POST /drift - compare les distribtions entre le dernier batch ingéré et les données d'entraînement 
.PHONY: api-drift
api-drift:
	@echo "POST $(API_URL)/drift"
	@curl -s -k -X POST "$(API_URL)/drift" \
		-H "Authorization: Bearer $(TOKEN)" \
		| python -m json.tool

.PHONY: api-drift-report
api-drift-report:
	explorer.exe reports/drift/drift_report.html

#################################################################################
# DOCKER COMPOSE
#################################################################################

## Build all images
.PHONY: docker-build
docker-build:
	$(COMPOSE_CMD) build

## Build airflow-init image (one time before first docker-up-airflow)
.PHONY: docker-build-airflow
docker-build-airflow:
	$(COMPOSE_CMD) --profile airflow build airflow-init

## Start stack — Mode 2 CLI subprocess (sans runners)
.PHONY: docker-up-cli
docker-up-cli:
	EXECUTION_MODE=cli $(COMPOSE_CMD) up -d --build

## Start stack — Mode 2 Docker exec (avec dvc/git runners)
.PHONY: docker-up-docker
docker-up-docker:
	EXECUTION_MODE=docker $(COMPOSE_CMD) --profile docker up -d --build

## Start stack — Mode 3 Airflow batch (stack cli + CeleryExecutor)
.PHONY: docker-up-batch
docker-up-batch:
	EXECUTION_MODE=cli $(COMPOSE_CMD) --profile airflow up -d --build

## Stop all services, keep volumes
.PHONY: docker-down
docker-down:
	$(COMPOSE_CMD) --profile docker --profile airflow down

## Stop all services + remove volumes (DANGER)
.PHONY: docker-down-v
docker-down-v:
	$(COMPOSE_CMD) --profile docker --profile airflow down -v

## Show containers status
.PHONY: docker-ps
docker-ps:
	$(COMPOSE_CMD) ps

## Tail logs (all services)
.PHONY: docker-logs
docker-logs:
	$(COMPOSE_CMD) logs -f

## Tail logs for one service → make docker-logs-svc SVC=api-train
.PHONY: docker-logs-svc
docker-logs-svc:
	$(COMPOSE_CMD) logs -f $(SVC)

## Show EXECUTION_MODE currently running in each service
.PHONY: docker-mode
docker-mode:
	@echo "Mode actif par service :"
	@docker inspect rakuten-ingest --format '{{range .Config.Env}}{{println .}}{{end}}' \
		2>/dev/null | grep EXECUTION_MODE || echo "  api-ingest: not running"
	@docker inspect rakuten-train --format '{{range .Config.Env}}{{println .}}{{end}}' \
		2>/dev/null | grep EXECUTION_MODE || echo "  api-train:  not running"

#################################################################################
# AIRFLOW — Mode 3
#################################################################################

## Initialize Airflow DB and admin user (one time setup)
.PHONY: airflow-init
airflow-init:
	EXECUTION_MODE=cli $(COMPOSE_CMD) --profile airflow run --rm airflow-init

## Trigger DAG manually
.PHONY: airflow-trigger
airflow-trigger:
	@docker exec rakuten-airflow-scheduler \
		airflow dags trigger rakuten_ml_pipeline

## List last DAG runs
.PHONY: airflow-runs
airflow-runs:
	@docker exec rakuten-airflow-scheduler \
		airflow dags list-runs -d rakuten_ml_pipeline --limit 10

## Set next batch number → make airflow-set-batch N=3
.PHONY: airflow-set-batch
airflow-set-batch:
	@docker exec rakuten-airflow-scheduler \
		airflow variables set next_batch_index $(N)

## Open Airflow UI in browser
.PHONY: airflow-ui
airflow-ui:
	@open http://localhost:8080 2>/dev/null \
		|| xdg-open http://localhost:8080 2>/dev/null \
		|| explorer.exe http://localhost:8080 2>/dev/null \
		|| echo "Open http://localhost:8080 in your browser"

#################################################################################
# STREAMLIT
#################################################################################

## Open Streamlit dashboard in browser
.PHONY: streamlit-ui
streamlit-ui:
	@open http://localhost:8501 2>/dev/null \
		|| xdg-open http://localhost:8501 2>/dev/null \
		|| explorer.exe http://localhost:8501 2>/dev/null \
		|| echo "Open http://localhost:8501 in your browser"

## Tail Streamlit logs
.PHONY: streamlit-logs
streamlit-logs:
	$(COMPOSE_CMD) logs -f streamlit

## Restart Streamlit service
.PHONY: streamlit-restart
streamlit-restart:
	$(COMPOSE_CMD) restart streamlit


#################################################################################
# UIs
#################################################################################

## Open MLflow UI (DagsHub)
.PHONY: mlflow-ui
mlflow-ui:
	@open https://dagshub.com/lioradatascientest/nov25cmlops_rakuten.mlflow 2>/dev/null \
		|| xdg-open https://dagshub.com/lioradatascientest/nov25cmlops_rakuten.mlflow 2>/dev/null \
		|| echo "Open https://dagshub.com/lioradatascientest/nov25cmlops_rakuten.mlflow in your browser"

## Open Grafana UI
.PHONY: grafana-ui
grafana-ui:
	@open http://localhost:3000 2>/dev/null \
		|| xdg-open http://localhost:3000 2>/dev/null \
		|| explorer.exe http://localhost:3000 2>/dev/null \
		|| echo "Open http://localhost:3000 in your browser"

## Open Swagger UI
.PHONY: swagger
swagger:
	@open https://localhost/docs 2>/dev/null \
		|| xdg-open https://localhost/docs 2>/dev/null \
		|| explorer.exe https://localhost/docs 2>/dev/null \
		|| echo "Open https://localhost/docs in your browser"

#################################################################################
# SMOKE TESTS
#################################################################################

## Quick health check through nginx
.PHONY: smoke-health
smoke-health:
	curl -k -s https://localhost/health | cat

#################################################################################
# HELP
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z0-9_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:30}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

.PHONY: help
help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)

#################################################################################
# GLOBALS
#################################################################################

PROJECT_NAME := nov25cmlops_rakuten
PYTHON_VERSION := 3.12
PYTHON_INTERPRETER := python

# Docker compose command (supports docker-compose v1 or docker compose v2)
COMPOSE_CMD := $(shell \
	if command -v docker-compose >/dev/null 2>&1; then \
		echo docker compose; \
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

# Variable 
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

## Force reset DVC tracking for debbugging, if DVC cache is corrupted or to start fresh (WARNING: will lose tracked data/models)
.PHONY: dvc-reset
dvc-reset:
	@echo "Reset DVC tracking (WARNING: will lose tracked data/models)"
	dvc repro --force 
	


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

#################################################################################
# DOCKER COMPOSE
#################################################################################

## Build all images
.PHONY: docker-build
docker-build:
	$(COMPOSE_CMD) build

## Start stack — mode CLI subprocess (sans dvc/git runner)
.PHONY: docker-up-cli
docker-up-cli:
	EXECUTION_MODE=cli $(COMPOSE_CMD) up -d --build

## Start stack — mode Docker exec (avec dvc/git runner via --profile docker)
.PHONY: docker-up-docker
docker-up-docker:
	EXECUTION_MODE=docker $(COMPOSE_CMD) --profile docker up -d --build

## Start stack — défaut (EXECUTION_MODE depuis .env, sans profil)
.PHONY: docker-up
docker-up:
	$(COMPOSE_CMD) up -d --build

## Stop services (keep volumes)
.PHONY: docker-down
docker-down:
	$(COMPOSE_CMD) --profile docker down

## Stop services + remove volumes (DANGER)
.PHONY: docker-down-v
docker-down-v:
	$(COMPOSE_CMD) --profile docker down -v

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
# MLflow UI (Dagshub)
#################################################################################
.PHONY: mflow-ui
mlflow-ui:
	open https://dagshub.com/shiff-oumi/nov25cmlops_rakuten_dag.mlflow \
	  2>/dev/null || xdg-open https://dagshub.com/shiff-oumi/nov25cmlops_rakuten_dag.mlflow		

#################################################################################
# QUICK SMOKE TESTS
#################################################################################

## Check gateway health (through nginx). Uses -k for self-signed TLS.
.PHONY: smoke-health
smoke-health:
	curl -k -s https://localhost/health | cat

## Open Swagger in browser (macOS). If not macOS, just open https://localhost/docs manually.
.PHONY: swagger
swagger:
	open https://localhost/docs

#################################################################################
# HELP
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z0-9_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
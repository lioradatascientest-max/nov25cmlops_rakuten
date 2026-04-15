from components.auth import is_admin, is_authenticated
import streamlit as st

if not is_authenticated():
    st.warning("Veuillez vous connecter.")
    st.stop()


st.title("Presentation du projet")
st.caption("NOV25CMLOPS_RAKUTEN - classification produit et plateforme MLOps")

st.markdown(
    """
    Nous sommes partis d'un problème de classification e-commerce :
    prédire automatiquement le type d'un produit Rakuten à partir de
    ses informations catalogue. Notre contribution n'est pas seulement
    un modèle ML, mais une plateforme MLOps de bout en bout.
    """
)

tabs = st.tabs(
    [
        "1. Contexte",
        "2. Valeur",
        "3. Architecture",
        "4. Pipeline ML",
        "5. Demo",
        "6. Perspectives",
    ]
)


with tabs[0]:
    st.header("Contexte et problématique")

    st.markdown(
        """
        Le projet s'appuie sur le challenge **Rakuten France Multimodal Product
        Data Classification**. L'objectif est de prédire le code type produit
        (`prdtypecode`) d'un nouveau produit dans le catalogue.

        Sur une marketplace, une bonne catégorisation conditionne la recherche,
        les recommandations, la qualité du catalogue et l'intégration de
        nouveaux vendeurs. Une approche manuelle ou uniquement basée sur des
        règles ne passe pas à l'échelle.
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset challenge", "~99k produits")
    col2.metric("Classes", "27 prdtypecode")
    col3.metric("Metrique", "F1 weighted")

    st.info(
        "Dans ce repo, nous nous concentrons sur la modalité texte, "
        "c'est-à-dire la désignation produit."
    )


with tabs[1]:
    st.header("Proposition de valeur")

    st.markdown(
        """
        La proposition de valeur est de transformer un classifieur produit en
        service exploitable :

        - prédiction temps réel pour les utilisateurs ;
        - ingestion de nouveaux lots pour les administrateurs ;
        - ré-entrainement reproductible ;
        - versioning des données et des modèles ;
        - monitoring technique et data drift.
        """
    )

    user_col, admin_col = st.columns(2)

    with user_col:
        st.subheader("Utilisateur")
        st.success(
            """
            - Soumettre une désignation produit
            - Recevoir une catégorie prédite
            - Consulter un top-k avec probabilités
            """
        )

    with admin_col:
        st.subheader("Administrateur")
        st.warning(
            """
            - Ingérer un batch CSV
            - Lancer un training
            - Vérifier les métriques
            - Surveiller le drift, l'API et le modèle
            """
        )

    st.subheader("Requis fonctionnels et technologies")
    st.dataframe(
        [
            {
                "Requis": "Prédire une catégorie à partir d'une désignation",
                "Implementation": "/predict + page Prediction",
                "Technos": "Streamlit, FastAPI, scikit-learn",
            },
            {
                "Requis": "Sécuriser les rôles",
                "Implementation": "Gateway JWT user/admin",
                "Technos": "FastAPI OAuth2, python-jose",
            },
            {
                "Requis": "Versionner les données",
                "Implementation": "DVC remote",
                "Technos": "DVC, DagsHub S3, Git",
            },
            {
                "Requis": "Tracker les modèles",
                "Implementation": "runs + registry + alias",
                "Technos": "MLflow, DagsHub",
            },
            {
                "Requis": "Orchestrer le batch",
                "Implementation": "DAG ingest/train/validate/drift",
                "Technos": "Airflow, DockerOperator",
            },
            {
                "Requis": "Monitorer",
                "Implementation": "/metrics + drift report",
                "Technos": "Prometheus, Grafana, Evidently",
            },
        ],
        use_container_width=True,
        hide_index=True,
    )


with tabs[2]:
    st.header("Architecture actuelle")

    st.markdown(
        """
        L'architecture a évolué. Elle est aujourd'hui
        organisée autour d'une UI Streamlit, d'une gateway securisée, de
        microservices FastAPI et de briques MLOps dédiées.
        """
    )

    st.graphviz_chart(
        """
        digraph {
          rankdir=LR;
          node [shape=box, style="rounded"];

          User [label="User / Admin"];
          UI [label="Streamlit UI"];
          Nginx [label="Nginx HTTPS"];
          Gateway [label="FastAPI Gateway\\nAuth JWT + routing"];
          Predict [label="api-predict\\n/predict /info /reload"];
          Ingest [label="api-ingest\\n/init /ingest"];
          Train [label="api-train\\n/train"];
          Monitor [label="api-monitor\\n/drift"];
          DVC [label="DVC + DagsHub S3\\nData versioning"];
          MLflow [label="MLflow / DagsHub\\nExperiments + registry"];
          Airflow [label="Airflow\\nBatch orchestration"];
          Prometheus [label="Prometheus\\n/metrics scrape"];
          Grafana [label="Grafana dashboards"];

          User -> UI;
          UI -> Gateway;
          User -> Nginx;
          Nginx -> Gateway;
          Gateway -> Predict;
          Gateway -> Ingest;
          Gateway -> Train;
          Gateway -> Monitor;
          Ingest -> DVC;
          Train -> DVC;
          Train -> MLflow;
          Predict -> MLflow;
          Monitor -> MLflow;
          Airflow -> Ingest;
          Airflow -> Train;
          Airflow -> Monitor;
          Gateway -> Prometheus;
          Predict -> Prometheus;
          Ingest -> Prometheus;
          Train -> Prometheus;
          Monitor -> Prometheus;
          Prometheus -> Grafana;
        }
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.info("Gateway : point de contrôle auth et routage.")
    col2.info("MLflow/DVC : traçabilité modèle et données.")
    col3.info("Prometheus/Grafana/Evidently : observabilité.")


with tabs[3]:
    st.header("Pipeline ML")

    st.graphviz_chart(
        """
        digraph {
          rankdir=TB;
          node [shape=box, style="rounded"];

          Raw [label="Raw data\\nX_train + Y_train + categories"];
          Seed [label="seed\\nmerge + batches"];
          Ingest [label="ingest\\nappend CSV + deduplicate"];
          Preprocess [label="preprocess\\nclean text + labels"];
          Transform [label="transform\\nTF-IDF + label encoding + split"];
          Train [label="train\\nLogisticRegression"];
          Evaluate [label="evaluate\\nmetrics + promotion"];
          Registry [label="MLflow Registry\\n@pending -> @production"];

          Raw -> Seed;
          Seed -> Ingest;
          Ingest -> Preprocess;
          Preprocess -> Transform;
          Transform -> Train;
          Train -> Evaluate;
          Evaluate -> Registry;
        }
        """
    )

    st.subheader("Choix du modèle")
    st.markdown(
        """
        Le modèle retenu est un compromis volontairement industrialisable :
        **TF-IDF + regression logistique**.

        - rapide à entrainer ;
        - compatible avec `predict_proba` pour le top-k ;
        - facilement versionnable ;
        - interprétable et robuste pour une demo MLOps.
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Validation accuracy", "0.7818")
    col2.metric("Validation F1 macro", "0.7635")
    col3.metric("Validation F1 weighted", "0.7849")

    st.code(
        """
Modele: LogisticRegression
Vectorisation: TF-IDF, 20 000 features, n-grams 1-2
Split: validation 20%, stratifié, random_state=42
Promotion: production si F1 macro >= production + 0.01
        """.strip()
    )


with tabs[4]:
    st.header("Deroulé de démo")

    demo_user, demo_admin = st.columns(2)

    with demo_user:
        st.subheader("Use case utilisateur")
        st.markdown(
            """
            1. Se connecter avec un compte `user`.
            2. Aller sur la page Prediction.
            3. Entrer une désignation produit.
            4. Choisir `top_k=3`.
            5. Montrer le code catégorie, le nom et la probabilité.
            """
        )
        st.code("Velo electrique pliable compact")
        st.code("Lot de 4 couverts inox pour cuisine")
        st.code("Roman policier edition poche")

    with demo_admin:
        st.subheader("Use case administrateur")
        st.markdown(
            """
            1. Se connecter avec un compte `admin`.
            2. Montrer les liens DagsHub, MLflow, Airflow, Grafana.
            3. Lancer ou expliquer un ingest batch.
            4. Lancer le training et le reload modèle.
            5. Consulter `/info`, MLflow, Evidently et Grafana.
            """
        )
        if is_admin():
            st.success("Vous êtes connecté en admin : démo complète possible.")
        else:
            st.info("Connecté en user : seul le parcours prediction est visible.")

    st.subheader("Message à dire")
    st.write(
        """
        Pour l'utilisateur final, la complexite MLOps disparait. Pour
        l'administrateur, la plateforme expose le cycle de vie complet :
        données, entrainement, versioning, promotion et monitoring.
        """
    )


with tabs[5]:
    st.header("Limites et perspectives")

    limits, next_steps = st.columns(2)

    with limits:
        st.subheader("Limites à assumer")
        st.markdown(
            """
            - Le challenge initial est multimodal, mais le repo utilise le texte.
            - L'authentification est adaptée à la démo, pas à une production.
            - Les chemins Airflow sont encore liés à un environnement hôte.
            - Le drift mesure surtout des distributions de données.
            - Les images ne sont pas encore exploitées.
            """
        )

    with next_steps:
        st.subheader("Perspectives")
        st.markdown(
            """
            - Ajouter la modalité image avec CNN/ViT.
            - Tester CamemBERT ou sentence-transformers.
            - Ajouter CI/CD et déploiement cloud.
            - Mettre en place un secret management réel.
            - Ajouter alerting Prometheus/Grafana.
            - Monitorer la performance avec labels retardés.
            """
        )

    st.success(
        """
        Conclusion : le projet montre la transformation d'un modèle de
        classification en systeme MLOps exploitable, versionné, securisé,
        monitorable et ré-entrainable.
        """
    )

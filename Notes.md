
make docker-up-docker

make swagger

samuel
admin789

make docker-logs-svc SVC=api-train

Le Seigneur Des Anneaux Figurine En Plomb À Peindre Avec Son Socle : Sam

make docker-down-v   

# Pour débloquer votre pipeline et supprimer le fichier de verrouillage
docker exec rakuten-train rm -f /app/.dvc/tmp/rwlock

# définir DagsHub comme stockage par défaut
docker exec rakuten-train dvc remote add -d storage https://dagshub.com/samuel.beau/nov25cmlops_rakuten_dag.dvc


# test token
curl -H "Authorization: Bearer b4542ab2a907ad40875ae0c056d7b9c225a52262" \
     https://dagshub.com/api/v1/samuel.beau


# Tout arrêter et nettoyer (Profil inclus) :
docker compose --profile docker down
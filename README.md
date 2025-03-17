# Mini-projet-R5.C.06
Mini projet R5.C.06 

## Lancer le script python

### Prérequis
- Docker fonctionnel sur sa machine

### Créer l'image
Se rendre dans le projet :
```bash
cd 'Mini-projet-R5.C.06'
```
Créer l'image
```docker
docker build --pull --rm -f './Dockerfile' -t 'miniprojet:latest' .
``` 

### Lancer l'image
Pour lancer l'image, ouvrir un terminal :
```
docker run --rm -v ./plots:/plots miniprojet
```

### Résultats
Les résultats de l'exécution de l'image se touvent dans le dossier **plots/**.
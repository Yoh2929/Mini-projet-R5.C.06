# Utiliser l'image officielle de Python
FROM python:3.9-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers requirements.txt dans le conteneur
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste des fichiers de l'application dans le conteneur
COPY . .

# Définir un volume pour sauvegarder les plots
VOLUME /plots

# Définir la commande par défaut pour exécuter l'application
CMD ["python", "app.py", "/plots"]
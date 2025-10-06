# Projet d'Analyse Immobilière

## Journal de bord

### 2025-10-04
- Création des fichiers de notebooks Jupyter
- Le contenu représente des suggestions
- **Note importante** : Veuillez créer des fonctions pour chaque étape, c'est plus propre que du code en vrac

## À venir
<!-- Ajoutez les prochaines étapes ici -->
oiuz

# Tutoriel de Base Git

## Authentification à GitHub

### Méthode recommandée : Authentification SSH (plus sécurisée)
Plus d'informations dans la [documentation officielle GitHub sur SSH](https://docs.github.com/fr/authentication/connecting-to-github-with-ssh).

### Alternative : GitHub Desktop
GitHub Desktop est une solution simple :

1. Téléchargez et installez [GitHub Desktop](https://desktop.github.com/)
2. Connectez-vous à votre compte GitHub
3. Clonez le dépôt via l'interface
4. Toutes les opérations Git peuvent être effectuées via l'application

GitHub Desktop gère automatiquement l'authentification pour vous.


## Configuration Initiale
### Cloner le Dépôt
Pour créer une copie locale du dépôt sur votre machine :
```bash
git clone https://github.com/jenzah/prog_web.git
cd nom-du-depot
```


## Gestion des Branches
### Afficher Toutes les Branches
```bash
git branch -a
```

### Créer une Nouvelle Branche
```bash
git checkout -b feature/votre-nouvelle-fonctionnalite
```
Cela crée une nouvelle branche et y bascule immédiatement.

### Basculer Entre les Branches
```bash
git checkout nom-de-branche
```


## Workflow
### Obtenir les Dernières Modifications
Avant de commencer à travailler, récupérez toujours les dernières modifications :
```bash
git pull
```

### Vérifier le Statut
Pour voir quels fichiers ont été modifiés :
```bash
git status
```

### Préparer les Modifications
Pour ajouter des fichiers modifiés à votre prochain commit :
```bash
git add fichier.php          # Ajouter un fichier spécifique
git add dossier/             # Ajouter tous les fichiers d'un dossier
git add .                    # Ajouter tous les fichiers modifiés
```

### Valider les Modifications
Enregistrez vos modifications préparées avec un message descriptif :
```bash
git commit -m "Votre message de commit descriptif"
```

### Pousser les Modifications
Téléchargez vos commits vers le dépôt distant :
```bash
git push
```

Pour une nouvelle branche qui n'a jamais été poussée auparavant :
```bash
git push -u origin feature/votre-nouvelle-fonctionnalite
```

## Opérations Avancées
### Fusionner les Modifications
Pour fusionner une autre branche dans votre branche actuelle :
```bash
git merge nom-de-branche
```
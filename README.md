# Projet - Caractères

## Introduction:
Le fichier ***pictures*** contient les differentes images pour les chiffres avec un certain nombre d'occurrences de chaque lettre. 
Le fichier **Pictures.ipynb** contient le programme permettant d'interagir avec les differentes images sous **Jupyter notebook** (**Pictures.py** étant la version python de base).

## Travailler sur le projet:

#Instalation:
Pour avoir le projet sur le votre machine, veuillez d'abord installer **Git** en exécutant les commandes suivantes (sur linux):
```bash
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install git
```
À partir de maintenant, **il est indispensable d'être situé dans le dossier 'projet-maths'**

# Prémière importation:
Ensuite, importer le projet ***pour la prémière fois*** en exécutant la commande : 
```bash
git clone https://github.com/MaGhAnBi/projet-maths.git
```
# Exporter son travail le depot git du projet: On le surnommera souvent 'faire un push'

```bash
git add -A
git commit -m "UN COURT MESSAGE POUR ETIQUETTER LE TRAVAIL OU SINON A LAISSER VIDE"
git push
```

# Importer l'état actuel du dépôt: (Mettre à jour son depot local surnommé 'pull')

```bash
git pull
```

# Remarque :

Quand on essaye de faire un **push** alors qu'il y a eu des mises à jour sur le depot principal depuis notre dernière importation, le système vous obligera à faire un **pull** avant, afin de mettre à jour votre dépot local. 

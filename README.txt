# Description du Projet

## Fichiers Importants

- `Convolution_chat_chien.ipynb`: Le fichier principal contenant le code du projet à exécuter.

Les autres fichiers sont des scripts d'entraînement et des exemples utilisés pour comprendre les concepts et développer le projet.

## À Propos du Projet

Ce projet permet d'importer une image de chien ou de chat via une page web locale, afin que le réseau de neurones puisse identifier s'il s'agit d'un chien ou d'un chat.

Le réseau de neurones a été entraîné avec un ensemble de données de chats et de chiens fourni par Microsoft et disponible sur Kaggle.

## Architecture du Modèle

Le modèle est un réseau de neurones convolutif qui comprend trois couches convolutives suivies de couches entièrement connectées. La sortie du modèle donne la probabilité que l'image soit celle d'un chat ou d'un chien.

## Processus d'Entraînement

Les données sont prétraitées et normalisées avant d'être fournies au réseau de neurones pour l'entraînement. Nous utilisons la Mean Squared Error (MSE) comme fonction de perte et Adam comme optimiseur pour l'entraînement du modèle. Le modèle est évalué en calculant la précision sur un ensemble de test.

Le code est compatible avec CUDA, ce qui permet d'utiliser un GPU pour accélérer l'entraînement et l'inférence du modèle.

## Utilisation de Node.js

### Démarrage du Serveur

1. Ouvrez votre terminal.
2. Naviguez vers le répertoire du projet.
3. Exécutez la commande: `http-server`.

### Accès à la Page Web

1. Ouvrez votre navigateur web préféré.
2. Entrez l'URL suivante: `http://localhost:8080/HTML_CNN_chat_chien.html`.

La page web devrait maintenant être accessible, et vous pouvez l'utiliser pour télécharger des images de chats ou de chiens depuis internet (ou utilisez celles à disposition dans le dossier) afin d'obtenir la classification du réseau de neurones.

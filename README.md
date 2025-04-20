# ChatPDF — Agent conversationnel intelligent pour fichiers PDF

## Objectif du projet 

concevoir et réaliser un agent conversationnel (chatbot) capable d’interagir avec des documents PDF en s'appuyant sur un **grand modèle de langage local (LLM)** et une approche **RAG (Retrieval-Augmented Generation)**.  
Le but est de surmonter les limitations contextuelles des LLM en enrichissant les réponses par du contenu provenant de documents personnalisés.

## Stack technique 
- Python
- Streamlit - Pour l'interface web interactive
- Langchain - Pour la gestion des chaînes RAG et la mémoire conversationnelle
- ChromaDB - Base vectorielle pour stocker les embeddings
- Sentence Transformers - Modèle d'embedding
- Ollama - Pour exécuter les LLMs localement
- PDFMiner - Pour extraire du texte à partir des fichiers PDF

## Fonctionnement
  - L'utilisateur importe un fichier PDF
  - Le texte du fichier est extrait et découpé en chunks
  - Ces fragments sont transformés en embeddigns et stockés dans ChromaDB
  - à chaque question posée:
      - Le système retrouve les morceaux de texte les plus pertinents
      - et les injecte dans un prompt pour le modèle LLM
      - Le modèle génére une réponse concise, contextualisé et pertinente
   
## Structure du projet
.

├── app.py                  # Fichier principal (Streamlit app)

├── htmlTemplates.py        # Templates HTML (interface conversation)

├── /pdfs                   # Répertoire pour stocker les files PDF

├── /db                    # Base de données ChromaDB
├── requirements.txt        # Dépendances

├── README.md               # ce fichier
   
## Fonctionnalités
  - Import de fichiers PDF
  - Indexation automatique via PDFMiner et embeddings
  - Récupération contextuelle (RAG)
  - Réponses basées uniquement sur le document fourni
  - Historique de conversation grace  langchain memory
  - Interface Web simple et intuitive via Streamlit
 
## Perspective
    - Ajout de la sélection du LLM (multi-languages)
    - Ajout de l'authentification de l'utilisateur
    - Amélioration de l'interface

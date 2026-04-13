# 🎓 AI Professor — Guide de démarrage complet

Ce projet est un **professeur IA local** qui lit tes cours (PDF / PowerPoint), les stocke dans une base de données vectorielle, et répond aux questions des étudiants en temps réel via WebSocket — texte, audio et synchronisation labiale inclus.

Tout tourne **en local** sur ta machine : aucune clé API externe, aucun coût, données privées.

---

## 📋 Ce dont tu as besoin avant de commencer

| Outil | Version minimale | Lien |
|---|---|---|
| Docker Desktop | 4.x | https://www.docker.com/products/docker-desktop |
| Git | n'importe laquelle | https://git-scm.com |

> **RAM recommandée :** 16 Go minimum. Le modèle LLM (Llama 3) a besoin d'espace pour tourner.

---

## 📁 Structure du projet

```
professor/
├── pipeline/               ← scripts de traitement des documents
│   ├── config.py           ← chemins et variables d'environnement
│   ├── spark_session.py    ← création de la session Spark
│   ├── bronze.py           ← extraction du texte brut (PDF/PPTX)
│   ├── silver.py           ← enrichissement des métadonnées
│   ├── gold.py             ← découpage sémantique + embeddings
│   ├── run_pipeline.py     ← orchestrateur principal du pipeline
│   ├── index_to_chroma.py  ← indexation dans ChromaDB
│   ├── extractor.py        ← extraction PDF et PPTX
│   ├── metadata.py         ← détection langue, sujet, professeur
│   └── requirements.txt    ← dépendances Python du pipeline
├── backend/
│   └── api/
│       └── main.py         ← API WebSocket FastAPI
├── data/
│   └── landing/            ← 📂 dépose tes PDF/PPTX ici
├── docker-compose.yml      ← tous les services
└── Dockerfile.api          ← image Docker de l'API
```

---

## 🚀 Étape 1 — Démarrer tous les services

Cette commande lance les 7 services en arrière-plan (PostgreSQL, ChromaDB, MinIO, Ollama, Spark, Airflow, API).

```bash
docker compose up -d --build
```

> ⏳ La première fois, Docker télécharge toutes les images. Cela peut prendre 5 à 15 minutes selon ta connexion.

Pour vérifier que tout est bien démarré :

```bash
docker compose ps
```

Tu dois voir tous les conteneurs avec le statut `running`.

---

## 🤖 Étape 2 — Télécharger les modèles IA

Ces modèles tournent **localement** dans le conteneur Ollama.

```bash
# Modèle de chat — répond aux questions des étudiants (~4.7 Go)
docker compose exec ollama ollama pull llama3

# Modèle d'embeddings — transforme le texte en vecteurs (~270 Mo)
docker compose exec ollama ollama pull nomic-embed-text
```

> ⏳ Le téléchargement de Llama 3 peut prendre 10 à 30 minutes. C'est normal.

---

## 📦 Étape 3 — Installer les dépendances Python dans Spark

```bash
docker compose exec spark pip install -r /opt/app/pipeline/requirements.txt
```

---

## 📄 Étape 4 — Déposer tes documents de cours

Copie tes fichiers PDF ou PowerPoint dans le dossier `data/landing/`.

**Convention de nommage recommandée** (optionnelle mais conseillée) :

```
Professeur_Matiere_Chapitre.pdf
```

Exemples :

```
Martin_AlgebreLineaire_Ch1.pdf
Dupont_Thermodynamique_Semaine3.pptx
Smith_IntroductionPython_Ch5.pdf
```

> Si le fichier ne suit pas cette convention, le système utilisera automatiquement Ollama pour deviner la matière à partir du contenu.

---

## ⚙️ Étape 5 — Lancer le pipeline de traitement

Cette commande lit les fichiers dans `data/landing/`, extrait le texte, enrichit les métadonnées, découpe les documents en morceaux sémantiques et calcule les embeddings. Elle crée trois couches Delta Lake :

- **Bronze** — texte brut extrait
- **Silver** — texte + métadonnées (professeur, matière, langue…)
- **Gold** — morceaux découpés + vecteurs d'embeddings

```bash
docker compose exec spark python /opt/app/pipeline/run_pipeline.py
```

> ⏳ Selon le nombre et la taille de tes fichiers, cette étape peut prendre plusieurs minutes. L'embedding de chaque morceau fait un appel à Ollama.

---

## 🗃️ Étape 6 — Indexer dans ChromaDB

Cette commande lit la couche Gold et envoie tous les morceaux + vecteurs dans ChromaDB pour la recherche sémantique.

```bash
docker compose exec spark python /opt/app/pipeline/index_to_chroma.py
```

Tu verras la progression :

```
671 chunks loaded.
Upserted 100/671 chunks...
Upserted 200/671 chunks...
...
Indexing complete — 671 chunks in 'ai_professor'.
```

> ✅ Cette commande est **idempotente** : tu peux la relancer sans risque après avoir ajouté de nouveaux documents.

---

## 🧪 Étape 7 — Tester le professeur IA

Crée un fichier `test.py` et exécute-le pour envoyer une question au professeur :

```python
import asyncio
import websockets
import json

async def test():
    uri = "ws://localhost:8000/ws/professor"

    async with websockets.connect(uri) as websocket:
        # Envoie une question en français (ou dans n'importe quelle langue)
        await websocket.send(json.dumps({
            "text": "Explique-moi le théorème central limite."
        }))

        while True:
            try:
                raw = await websocket.recv()
                msg = json.loads(raw)

                if msg["is_final"]:
                    print("\n\n--- Réponse complète ---")
                    print(msg["text_chunk"])
                else:
                    # Affiche les tokens au fur et à mesure
                    print(msg["text_chunk"], end="", flush=True)

            except Exception as e:
                print("\nTerminé :", e)
                break

asyncio.run(test())
```

```bash
# Installer websockets si ce n'est pas déjà fait
pip install websockets

python test.py
```

Le professeur répondra **dans la même langue que ta question**.

---

## 🔄 Ajouter de nouveaux documents plus tard

Quand tu veux ajouter de nouveaux cours :

```bash
# 1. Copie tes nouveaux PDF/PPTX dans data/landing/

# 2. Relance le pipeline
docker compose exec spark python /opt/app/pipeline/run_pipeline.py

# 3. Réindexe dans ChromaDB
docker compose exec spark python /opt/app/pipeline/index_to_chroma.py
```

---

## 🌐 Interfaces disponibles

| Service | URL | Description |
|---|---|---|
| API WebSocket | `ws://localhost:8000/ws/professor` | Point d'entrée principal |
| API Health | http://localhost:8000/health | Vérifie que l'API tourne |
| Airflow | http://localhost:8080 | Orchestration (admin / admin) |
| MinIO | http://localhost:9001 | Stockage objets (minioadmin / minioadmin) |
| ChromaDB | http://localhost:8001 | Base vectorielle |

---

## 🛠️ Commandes utiles

```bash
# Voir les logs d'un service en temps réel
docker compose logs -f api
docker compose logs -f ollama
docker compose logs -f spark

# Redémarrer l'API après une modification de code
docker compose restart api

# Arrêter tous les services
docker compose down

# Arrêter ET supprimer les volumes (repart de zéro)
docker compose down -v

# Ouvrir un terminal dans un conteneur
docker compose exec spark bash
docker compose exec api bash
```

---

## ❗ Problèmes fréquents

**Les conteneurs ne démarrent pas**
> Vérifie que Docker Desktop est bien ouvert et que tu as suffisamment de RAM disponible.

**Erreur `No module named '...'` dans Spark**
> Réinstalle les dépendances :
> ```bash
> docker compose exec spark pip install -r /opt/app/pipeline/requirements.txt
> ```

**Le pipeline tourne mais les réponses ne tiennent pas compte du cours**
> L'indexation ChromaDB n'a peut-être pas été faite. Lance l'étape 6.

**Llama 3 répond lentement**
> C'est normal sans GPU. Sur CPU, attends 10 à 30 secondes par réponse.

**Erreur de schéma Delta Lake**
> Le schéma a changé entre deux runs. Le pipeline utilise `overwriteSchema=true` automatiquement, donc relancer suffit.

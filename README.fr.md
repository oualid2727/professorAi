# 🎓 AI Professor — Guide complet

**AI Professor** est un professeur IA qui tourne entièrement en local sur ta machine. Tu lui donnes des cours en PDF ou PowerPoint, et il répond aux questions des étudiants en temps réel avec de la voix, de la reconnaissance vocale, et de la synchronisation labiale pour un avatar 3D.

Aucune clé API externe. Aucun coût. Données 100 % privées.

---

## 🧠 Ce que fait le système

```
Étudiant parle  →  Whisper STT  →  question texte
                                        ↓
                              Recherche hybride (BM25 + vecteurs)
                              dans les cours indexés (ChromaDB)
                                        ↓
                              Llama 3 génère une réponse
                              en tenant compte de l'historique
                                        ↓
                    XTTS v2 synthétise la voix  +  visèmes pour l'avatar
                                        ↓
                    Réponse streamée en temps réel via WebSocket
```

---

## 📋 Prérequis

| Outil | Version | Lien |
|---|---|---|
| Docker Desktop | 4.x | https://www.docker.com/products/docker-desktop |
| Git | n'importe laquelle | https://git-scm.com |
| NVIDIA Container Toolkit | dernière | voir étape 0 ci-dessous |

> **RAM recommandée :** 16 Go minimum.
> **GPU :** NVIDIA recommandé. Le code détecte automatiquement le GPU au démarrage et bascule sur CPU si aucun GPU n'est disponible.

---

## 📁 Structure du projet

```
professor/
├── pipeline/                   ← traitement des documents (PySpark)
│   ├── config.py               ← chemins et variables d'environnement
│   ├── spark_session.py        ← session Spark + distribution des modules
│   ├── extractor.py            ← extraction PDF et PPTX
│   ├── metadata.py             ← détection langue, sujet, professeur
│   ├── bronze.py               ← couche Bronze : texte brut extrait
│   ├── silver.py               ← couche Silver : métadonnées enrichies
│   ├── gold.py                 ← couche Gold : découpage sémantique + embeddings
│   ├── run_pipeline.py         ← orchestrateur Bronze → Silver → Gold
│   ├── index_to_chroma.py      ← indexation dans ChromaDB
│   └── requirements.txt        ← dépendances Python du pipeline Spark
├── backend/
│   └── api/
│       ├── main.py             ← API WebSocket FastAPI
│       ├── chain.py            ← recherche hybride BM25 + vecteurs
│       ├── memory.py           ← mémoire de conversation (PostgreSQL)
│       ├── stt.py              ← reconnaissance vocale (Whisper)
│       ├── tts.py              ← synthèse vocale (XTTS v2) + visèmes
│       └── reference_voice/
│           └── professor.wav   ← (optionnel) voix de référence pour clonage
├── airflow/
│   └── dags/
│       └── professor_pipeline.py  ← DAG Airflow (surveillance automatique)
├── data/
│   └── landing/                ← 📂 dépose tes PDF/PPTX ici
├── docker-compose.yml
├── Dockerfile.api
└── backend/requirements.txt    ← dépendances Python de l'API
```

---

## 🚀 Installation

### Étape 0 — Installer le NVIDIA Container Toolkit (serveur GPU uniquement)

> **Passe cette étape si tu n'as pas de GPU NVIDIA.**
> Sur CPU, tout fonctionne mais la synthèse vocale est plus lente.

```bash
# Ajouter le dépôt NVIDIA
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Installer et redémarrer Docker
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Vérifier que le GPU est visible dans Docker
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

Tu dois voir le nom de ton GPU dans la sortie de `nvidia-smi`.

---

### Étape 1 — Démarrer tous les services

```bash
docker compose up -d --build
```

> ⏳ La première fois, Docker télécharge toutes les images (~5-15 min selon ta connexion).
> Le build de l'image API prend plus longtemps car elle installe PyTorch + XTTS.

Vérifier que tout tourne :

```bash
docker compose ps
```

Tous les conteneurs doivent être en statut `running`.

---

### Étape 2 — Télécharger les modèles IA dans Ollama

```bash
# Modèle de chat — génère les réponses (~4.7 Go)
docker compose exec ollama ollama pull llama3

# Modèle d'embeddings — transforme le texte en vecteurs (~270 Mo)
docker compose exec ollama ollama pull nomic-embed-text
```

> ⏳ Le téléchargement de Llama 3 peut prendre 10 à 30 minutes.

---

### Étape 3 — Attendre que l'API soit prête

XTTS v2 télécharge ses poids (~1.8 Go) au premier démarrage. Surveille les logs :

```bash
docker compose logs -f api
```

Attends ces lignes avant de continuer :

```
[stt] Whisper 'medium' ready on cuda.
[tts] XTTS v2 ready on cuda.
Application startup complete.
```

> ⏳ Premier démarrage : 5-15 minutes. Démarrages suivants : 30-60 secondes.

---

### Étape 4 — Installer les dépendances Python dans Spark

```bash
docker compose exec spark pip install -r /opt/app/pipeline/requirements.txt
```

---

### Étape 5 — Déposer tes documents de cours

Copie tes fichiers PDF ou PowerPoint dans le dossier `data/landing/`.

**Convention de nommage recommandée :**

```
Professeur_Matiere_Chapitre.pdf
```

Exemples :

```
Martin_AlgebreLineaire_Ch1.pdf
Dupont_Thermodynamique_Semaine3.pptx
Smith_IntroductionPython_Ch5.pdf
```

> Si le fichier ne suit pas cette convention, le système détecte automatiquement
> la matière à partir du contenu via Ollama.

---

### Étape 6 — Lancer le pipeline de traitement

```bash
docker compose exec spark python /opt/app/pipeline/run_pipeline.py
```

Cette commande crée trois couches Delta Lake :

- **Bronze** — texte brut extrait du PDF/PPTX
- **Silver** — texte + métadonnées (professeur, matière, langue…)
- **Gold** — morceaux sémantiques + vecteurs d'embeddings 768 dimensions

> ⏳ Quelques minutes selon la taille des fichiers.

---

### Étape 7 — Indexer dans ChromaDB

```bash
docker compose exec spark python /opt/app/pipeline/index_to_chroma.py
```

Tu verras la progression :

```
671 chunks loaded.
Upserted 100/671 chunks...
Indexing complete — 671 chunks in 'ai_professor'.
```

> ✅ Idempotent : relance sans risque après ajout de nouveaux documents.

---

### Étape 8 — Tester le professeur

```bash
pip install websockets
```

Crée un fichier `test.py` :

```python
import asyncio, websockets, json

async def test():
    async with websockets.connect("ws://localhost:8000/ws/professor") as ws:
        await ws.send(json.dumps({"text": "Explique-moi le théorème central limite."}))
        while True:
            try:
                msg = json.loads(await ws.recv())
                if msg.get("is_final"):
                    print("\n\n--- Réponse complète ---")
                    print(msg["text_chunk"])
                    break
                elif msg.get("text_chunk"):
                    print(msg["text_chunk"], end="", flush=True)
            except Exception as e:
                print("\nTerminé :", e)
                break

asyncio.run(test())
```

```bash
python test.py
```

---

### Étape 9 — Tester la voix (optionnel)

```bash
python test_tts.py
```

Le script sauvegarde les chunks audio dans `tts_output/chunk_01.wav`, `chunk_02.wav` etc.
et affiche les visèmes pour chaque phrase synthétisée.

---

## 🔄 Ajouter de nouveaux documents plus tard

```bash
# 1. Copie tes nouveaux fichiers dans data/landing/

# 2. Relance le pipeline
docker compose exec spark python /opt/app/pipeline/run_pipeline.py

# 3. Réindexe dans ChromaDB
docker compose exec spark python /opt/app/pipeline/index_to_chroma.py
```

> L'Airflow DAG fait ça automatiquement toutes les 5 minutes si tu déposes
> des fichiers dans `data/landing/`. Interface : http://localhost:8080 (admin / admin).

---

## 🌐 Interfaces disponibles

| Service | URL | Accès |
|---|---|---|
| **API WebSocket** | `ws://localhost:8000/ws/professor` | Point d'entrée principal |
| **API Health** | http://localhost:8000/health | Vérifie que l'API tourne |
| **Airflow** | http://localhost:8080 | admin / admin |
| **MinIO** | http://localhost:9001 | minioadmin / minioadmin |
| **ChromaDB** | http://localhost:8001 | — |

---

## 📡 Format du WebSocket

### Entrées acceptées

```json
// Question texte
{ "text": "C'est quoi le théorème de Bayes ?" }

// Question audio (base64)
{ "audio_b64": "<audio base64>", "mime_type": "audio/webm" }

// Reprendre une session
{ "session_id": "uuid-de-session", "text": "Peux-tu répéter ?" }

// Effacer l'historique
{ "command": "clear_history" }
```

### Messages reçus

```json
// Echo de session (premier message)
{ "session_id": "uuid", "is_final": false }

// Transcription audio (si entrée audio)
{ "transcript": "Ce que Whisper a entendu", "is_final": false }

// Token de texte (arrive en continu)
{ "text_chunk": "Le théorème", "audio_b64": null, "visemes": [], "is_final": false }

// Chunk audio + visèmes (après chaque phrase complète)
{ "text_chunk": "", "audio_b64": "<WAV base64>", "visemes": [{"id": 12, "timestamp": 0.0}], "is_final": false }

// Message final avec sources
{ "text_chunk": "<réponse complète>", "sources": [...], "is_final": true }
```

### IDs de visèmes (standard Oculus / Ready Player Me)

| ID | Visème | Phonèmes |
|---|---|---|
| 0 | Silence | — |
| 1 | PP | p, b, m |
| 2 | FF | f, v |
| 4 | DD | t, d |
| 5 | kk | k, g |
| 7 | SS | s, z |
| 10 | aa | voyelle ouverte |
| 11 | E | é, è, ə |
| 12 | I | i, y |
| 13 | O | o, ɔ |
| 14 | U | u, ou |

---

## 🎙️ Voix de référence pour le clonage vocal (optionnel)

Pour donner au professeur une voix cohérente :

1. Enregistre 10 à 30 secondes d'audio propre (pas de bruit de fond)
2. Exporte en WAV mono ou stéréo, n'importe quel sample rate
3. Place le fichier ici : `backend/api/reference_voice/professor.wav`

> Si le fichier n'existe pas, XTTS utilise sa voix française intégrée.

---

## 🛠️ Commandes utiles

```bash
# Voir les logs en temps réel
docker compose logs -f api
docker compose logs -f ollama
docker compose logs -f spark

# Redémarrer l'API après modification du code backend
docker compose restart api

# Ouvrir un terminal dans un conteneur
docker compose exec api bash
docker compose exec spark bash

# Arrêter tous les services
docker compose down

# Tout effacer et repartir de zéro (supprime les données !)
docker compose down -v
```

---

## ❗ Problèmes fréquents

**`[tts] Failed to load XTTS v2: EOF when reading a line`**
> La variable `COQUI_TOS_AGREED=1` est manquante dans le docker-compose.
> Vérifie qu'elle est bien présente dans la section `environment` du service `api`.

**`ModuleNotFoundError` dans Spark**
> Réinstalle les dépendances :
> ```bash
> docker compose exec spark pip install -r /opt/app/pipeline/requirements.txt
> ```

**`ValueError: Could not connect to tenant default_tenant` (ChromaDB)**
> Version du client chromadb trop ancienne dans Spark :
> ```bash
> docker compose exec spark pip install "chromadb>=0.6.0"
> ```

**`ModuleNotFoundError: No module named 'bronze'` dans Spark**
> Les modules ne sont pas distribués aux executors. Vérifie que `spark_session.py`
> liste bien tous les fichiers dans `spark.submit.pyFiles`.

**Le professeur répond mais ignore le contenu du cours**
> L'indexation ChromaDB n'a pas été faite ou est vide. Relance l'étape 7.

**`A schema mismatch detected` (Delta Lake)**
> Le schéma a changé entre deux runs. Relance simplement le pipeline,
> il utilise `overwriteSchema=true` automatiquement.

**Réponses très lentes (30-60s par phrase)**
> Normal sans GPU NVIDIA. Voir le tableau des performances ci-dessous.

---

## ⚡ Performances selon le matériel

| Composant | CPU seul | GPU NVIDIA |
|---|---|---|
| Whisper (transcription) | 3-5s | < 1s |
| XTTS v2 (synthèse par phrase) | 30-60s | 1-2s |
| Llama 3 (réponse complète) | 10-30s | 2-5s |
| Modèle Whisper conseillé | `small` | `medium` |

Sur GPU, le professeur répond en moins de 5 secondes au total.
Sur CPU, il vaut mieux désactiver la voix (`TTS_ENABLED: "false"` dans docker-compose)
et utiliser uniquement le texte pour les tests.

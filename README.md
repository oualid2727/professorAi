## AI Professor Lakehouse & 3D Integration

**Goal**: Local, RAG-based AI professor that ingests course materials through a medallion lakehouse, serves low-latency answers (text + audio + visemes) over WebSockets for a 3D classroom.

### Project Structure

- **docker-compose.yml**: Orchestrates PostgreSQL, ChromaDB, MinIO, Ollama, Spark, Airflow, and the FastAPI API service.
- **backend/**:
  - `api/main.py`: FastAPI app with `/ws/professor` WebSocket that streams `text_chunk`, `audio_b64`, `visemes`, and `is_final`.
  - `rag/chain.py`: LangChain-based RAG chain using Chroma and Ollama.
  - `evaluation/ragas_eval.py`: Ragas script to measure answer faithfulness.
  - `requirements.txt`: Python dependencies for the API/RAG/evaluation.
- **pipeline/**:
  - `pdf_to_delta.py`: PySpark script to move from landing -> bronze/silver/gold Delta tables.
- **data/**:
  - `landing/`: Drop PDFs/PPTX here for ingestion.

### Running the Stack

1. **Prerequisites**
   - Docker Desktop with Docker Compose.
   - Enough RAM/CPU to run local LLM (Ollama) and Spark.

2. **Start services**

```bash
docker compose up -d --build
```

3. **Prepare data**
   - Copy your course PDFs/PPTX into `data/landing/`.
   - From the `spark` container, run:

```bash
docker compose exec spark python /opt/app/pipeline/pdf_to_delta.py
```

This writes Delta tables under `/opt/app/data/delta/{bronze,silver,gold}`.

4. **RAG indexing (conceptual)**
   - Load the gold Delta table in a separate script, create `langchain` `Document` objects, and upsert them into Chroma with `collection_name="ai_professor"`. The RAG chain in `backend/rag/chain.py` expects this collection.

5. **Using the WebSocket API**

- Connect your Next.js 3D classroom client to:
  - `ws://localhost:8000/ws/professor`
- Send JSON:

```json
{ "text": "Explain the central limit theorem." }
```

- Receive streaming chunks shaped as:

```json
{
  "text_chunk": "Hello",
  "audio_b64": "...",
  "visemes": [
    { "id": "A", "timestamp": 0.1 },
    { "id": "O", "timestamp": 0.2 }
  ],
  "is_final": false
}
```

The current implementation uses a **dummy local TTS and viseme generator** to keep the interface stable; swap `synthesize_dummy_audio` and `generate_visemes` in `api/main.py` with Piper/Coqui + a phoneme-to-viseme mapping.

6. **Ragas evaluation**

- After logging question/answer/context triplets from real conversations, run:

```bash
docker compose exec api python -m evaluation.ragas_eval
```

Edit `evaluation/ragas_eval.py` to point to your stored interactions and inspect the printed `faithfulness` scores.


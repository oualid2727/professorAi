# backend/api/run_full_evaluation.py
#
# Harnais d'évaluation scientifique du système RAG — version résiliente v2.
#
# Nouveau changement (après avoir découvert que le seuil CRAG bugué rendait
# la sauvegarde précédente inutilisable pour recalculer les verdicts) :
#
#   3. Le score numérique brut du meilleur candidat (top_score) est
#      désormais sauvegardé pour chaque question, en plus du texte des
#      contextes. Cela permet de recalculer le verdict CRAG ET la
#      concordance a posteriori avec n'importe quel seuil, INSTANTANÉMENT,
#      sans rappeler ni la récupération ni la génération — voir
#      recompute_concordance.py. Sans ce score, un changement de seuil
#      oblige à tout refaire tourner depuis zéro, ce qui vient de coûter
#      plusieurs heures inutilement.
#
# USAGE (depuis le conteneur api, working dir = /app) :
#   docker compose exec -d api sh -c \
#     "EVAL_QUESTIONS_FILE=/app/api/questions_test.json \
#      EVAL_RESULTS_FILE=/app/api/evaluation_results.json \
#      python -m api.run_full_evaluation > /app/api/eval_log.txt 2>&1"

import json
import os
import time
from typing import Dict, List, Optional

import httpx

try:
    from api.chain import retrieve_context
except ImportError:
    from chain import retrieve_context  # exécution en script autonome / tests

try:
    from api.ragas_eval import build_dataset, evaluate_metrics
except ImportError:
    from ragas_eval import build_dataset, evaluate_metrics


# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "ollama")
OLLAMA_PORT  = int(os.getenv("OLLAMA_PORT", "11434"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

QUESTIONS_FILE = os.getenv("EVAL_QUESTIONS_FILE", "questions_test.json")
RESULTS_FILE   = os.getenv("EVAL_RESULTS_FILE", "evaluation_results.json")

GENERATION_TIMEOUT = float(os.getenv("EVAL_GENERATION_TIMEOUT", "180.0"))

VALID_VERDICTS = {"correct", "ambiguous", "incorrect"}

SOURCE_TEXT_FIELDS = ["chunk_text", "text", "content", "page_content"]
# Nom du champ contenant le score numérique dans les dictionnaires "sources"
# (voir chain.py : meta["score"] = round(score, 3))
SOURCE_SCORE_FIELDS = ["score"]


# ── Génération synchrone d'une réponse (hors WebSocket, pour l'évaluation) ────

def _build_simple_prompt(question: str, context: str) -> str:
    return (
        "Use the following course material to answer the student's question.\n\n"
        f"--- COURSE CONTEXT ---\n{context}\n--- END CONTEXT ---\n\n"
        f"Student question: {question}\n\n"
        "Answer concisely, based only on the context above."
    )


def generate_answer(question: str, context: str) -> str:
    prompt = _build_simple_prompt(question, context)
    try:
        resp = httpx.post(
            f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=GENERATION_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        print(f"[eval] generation error for {question!r}: {e}")
        return ""


def extract_source_text(source: Dict) -> Optional[str]:
    if not isinstance(source, dict):
        return None
    for field in SOURCE_TEXT_FIELDS:
        if field in source and source[field]:
            return str(source[field])
    return None


def extract_source_score(source: Dict) -> Optional[float]:
    """Récupère le score numérique d'une source, si présent."""
    if not isinstance(source, dict):
        return None
    for field in SOURCE_SCORE_FIELDS:
        if field in source and source[field] is not None:
            try:
                return float(source[field])
            except (TypeError, ValueError):
                continue
    return None


# ── Chargement des questions de test ──────────────────────────────────────────

def load_questions(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        questions = json.load(f)
    for q in questions:
        if q.get("expected_verdict") not in VALID_VERDICTS:
            raise ValueError(
                f"expected_verdict invalide pour {q.get('question')!r}: "
                f"{q.get('expected_verdict')!r} (doit être parmi {VALID_VERDICTS})"
            )
    return questions


# ── Sauvegarde incrémentale ────────────────────────────────────────────────────

def save_partial(per_question: List[Dict], path: str) -> None:
    matches = sum(1 for r in per_question if r["match"])
    total = len(per_question)
    partial = {
        "status": "in_progress",
        "total_questions_so_far": total,
        "concordance_matches_so_far": matches,
        "concordance_rate_so_far": (matches / total) if total else 0.0,
        "per_question": per_question,
    }
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(partial, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


# ── Boucle principale ──────────────────────────────────────────────────────────

def run_evaluation(questions: List[Dict], results_path: str = None) -> Dict:
    per_question = []
    ragas_questions, ragas_answers, ragas_contexts = [], [], []

    total = len(questions)
    for i, item in enumerate(questions, 1):
        question = item["question"]
        category = item.get("category", "")
        expected = item["expected_verdict"]

        print(f"[eval] ({i}/{total}) {question!r}", flush=True)

        row = {
            "question": question,
            "category": category,
            "expected_verdict": expected,
            "obtained_verdict": None,
            "top_score": None,     # nouveau — permet un recalcul instantané des seuils
            "match": False,
            "answer": None,
            "context_texts": [],
            "num_sources": 0,
            "error": None,
        }

        try:
            context, sources, grade = retrieve_context(question)
        except Exception as e:
            print(f"[eval] retrieve_context error: {e}")
            row["error"] = f"retrieve_context: {e}"
            per_question.append(row)
            if results_path:
                save_partial(per_question, results_path)
            continue

        row["obtained_verdict"] = grade
        row["match"] = (grade == expected)
        row["num_sources"] = len(sources) if sources else 0

        source_scores = [s for s in (extract_source_score(src) for src in (sources or [])) if s is not None]
        if source_scores:
            row["top_score"] = max(source_scores)
        elif not sources:
            print(f"[eval] pas de score trouvé dans les sources — clés disponibles: "
                  f"{list(sources[0].keys()) if sources else '(aucune source)'}")

        if grade != "incorrect" and context:
            answer = generate_answer(question, context)
            row["answer"] = answer

            context_texts = []
            for src in (sources or []):
                text = extract_source_text(src)
                if text:
                    context_texts.append(text)
                else:
                    print(f"[eval] champ de texte introuvable dans la source. Clés disponibles: {list(src.keys()) if isinstance(src, dict) else type(src)}")

            row["context_texts"] = context_texts

            if answer and context_texts:
                ragas_questions.append(question)
                ragas_answers.append(answer)
                ragas_contexts.append(context_texts)
            else:
                row["error"] = "answer ou contexts vide — échantillon exclu de RAGAS"

        per_question.append(row)

        if results_path:
            save_partial(per_question, results_path)

    matches = sum(1 for r in per_question if r["match"])
    concordance_rate = matches / total if total else 0.0

    ragas_scores = None
    ragas_error = None
    if ragas_questions:
        print(f"\n[eval] Calcul RAGAS sur {len(ragas_questions)} question(s) éligibles...", flush=True)
        try:
            dataset = build_dataset(ragas_questions, ragas_answers, ragas_contexts)
            result = evaluate_metrics(dataset)
            ragas_scores = dict(result)
        except Exception as e:
            print(f"[eval] RAGAS evaluation error: {e}")
            ragas_error = str(e)
    else:
        ragas_error = "Aucune question éligible pour RAGAS (toutes en refus, ou erreurs)"

    summary = {
        "status": "complete",
        "total_questions": total,
        "concordance_matches": matches,
        "concordance_rate": concordance_rate,
        "ragas_eligible_count": len(ragas_questions),
        "ragas_scores": ragas_scores,
        "ragas_error": ragas_error,
        "per_question": per_question,
    }
    return summary


def print_summary(summary: Dict) -> None:
    print("\n" + "=" * 60)
    print("RÉSUMÉ DE L'ÉVALUATION")
    print("=" * 60)
    print(f"Questions testées         : {summary['total_questions']}")
    print(f"Concordance CRAG          : {summary['concordance_matches']}/{summary['total_questions']} "
          f"({summary['concordance_rate']*100:.1f}%)")
    if summary["ragas_scores"]:
        print(f"Échantillons évalués RAGAS: {summary['ragas_eligible_count']}")
        for metric, score in summary["ragas_scores"].items():
            print(f"  {metric:20s}: {score:.3f}")
    else:
        print(f"RAGAS non calculé : {summary['ragas_error']}")
    print("=" * 60)

    mismatches = [r for r in summary["per_question"] if not r["match"] and not r["error"]]
    if mismatches:
        print(f"\n{len(mismatches)} désaccord(s) de verdict CRAG :")
        for r in mismatches:
            print(f"  - [{r['category']}] {r['question']!r}: attendu={r['expected_verdict']}, obtenu={r['obtained_verdict']}, score={r['top_score']}")


def main():
    questions = load_questions(QUESTIONS_FILE)
    t0 = time.time()
    summary = run_evaluation(questions, results_path=RESULTS_FILE)
    elapsed = time.time() - t0

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print_summary(summary)
    print(f"\nDurée totale : {elapsed/60:.1f} minutes")
    print(f"Résultats détaillés sauvegardés dans {RESULTS_FILE}")


if __name__ == "__main__":
    main()
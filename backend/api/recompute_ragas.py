# backend/api/recompute_ragas.py
#
# Recalcule les métriques RAGAS à partir des context_texts/answers déjà
# sauvegardés — SANS refaire la récupération ni la génération.
#
# Version diagnostique : teste chaque métrique SÉPARÉMENT (faithfulness
# seule, puis answer_relevancy seule) et capture la trace complète de toute
# erreur, au lieu du message tronqué qu'on obtient en les calculant
# ensemble. Un échec sur une métrique ne doit plus jamais masquer le succès
# de l'autre, ni cacher la vraie cause de l'échec.
#
# USAGE :
#   docker compose exec api python -m api.recompute_ragas

import json
import os
import traceback

from ragas.metrics import faithfulness, answer_relevancy

try:
    from api.ragas_eval import build_dataset, evaluate_metrics
except ImportError:
    from ragas_eval import build_dataset, evaluate_metrics

RESULTS_FILE = os.getenv("EVAL_RESULTS_FILE", "evaluation_results.json")


def collect_eligible(summary):
    questions, answers, contexts = [], [], []
    skipped = 0
    for row in summary.get("per_question", []):
        answer = row.get("answer")
        ctx = row.get("context_texts") or []
        if answer and ctx:
            questions.append(row["question"])
            answers.append(answer)
            contexts.append(ctx)
        else:
            skipped += 1
    return questions, answers, contexts, skipped


def try_metric(name, metric, questions, answers, contexts):
    print(f"\n--- {name} ---")
    try:
        dataset = build_dataset(questions, answers, contexts)
        result = evaluate_metrics(dataset, metrics=[metric])
        scores = dict(result)
        print(f"OK: {scores}")
        return scores, None
    except Exception as e:
        tb = traceback.format_exc()
        print(f"ÉCHEC: {e!r}")
        print(tb)
        return None, tb


def main():
    with open(RESULTS_FILE, encoding="utf-8") as f:
        summary = json.load(f)

    questions, answers, contexts, skipped = collect_eligible(summary)
    print(f"{len(questions)} échantillon(s) exploitable(s), {skipped} ignoré(s).")

    if not questions:
        print("Aucun échantillon exploitable — le fichier ne contient probablement pas "
              "context_texts (produit par une ancienne version du harnais).")
        return

    all_scores = {}
    all_errors = {}

    for name, metric in [("faithfulness", faithfulness), ("answer_relevancy", answer_relevancy)]:
        scores, tb = try_metric(name, metric, questions, answers, contexts)
        if scores:
            all_scores.update(scores)
        else:
            all_errors[name] = tb

    print("\n" + "=" * 60)
    print("RÉSULTAT FINAL")
    print("=" * 60)
    if all_scores:
        for k, v in all_scores.items():
            print(f"  {k:20s}: {v:.3f}")
    if all_errors:
        print(f"\nMétrique(s) en échec : {list(all_errors.keys())}")
        print("(trace complète affichée ci-dessus pour chacune)")

    summary["ragas_scores"] = all_scores if all_scores else None
    summary["ragas_error"] = None if not all_errors else f"failed metrics: {list(all_errors.keys())}"
    summary["ragas_eligible_count"] = len(questions)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSauvegardé dans {RESULTS_FILE}")


if __name__ == "__main__":
    main()
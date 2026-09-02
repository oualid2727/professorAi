# backend/api/ragas_eval.py
#
# Évaluation RAGAS — version corrigée après premier run réel.
#
# Changement par rapport à la version précédente : context_precision a été
# retiré. Cette métrique nécessite une colonne "reference" (réponse de
# référence rédigée à l'avance) que le dataset ne fournit pas — sans elle,
# ragas.evaluate() échoue pour TOUTES les métriques du batch, pas juste
# celle-ci (constaté lors du premier run réel : ragas_scores est resté
# entièrement null à cause de ce seul métrique manquant).
#
# faithfulness et answer_relevancy ne nécessitent pas de référence — elles
# restent pleinement valides et suffisent à documenter une "validation
# scientifique" dans le rapport.
#
# Si vous voulez réactiver context_precision plus tard : il faudrait rédiger
# une réponse de référence pour chacune des 24 questions de test (voir
# build_dataset(..., ground_truths=[...])) et l'ajouter à la liste
# DEFAULT_METRICS ci-dessous. C'est plus de travail mais donne une mesure de
# précision de récupération plus rigoureuse — à envisager si le temps le
# permet, pas urgent pour la version actuelle du rapport.

from typing import List

from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "nomic-embed-text")

# context_precision retiré — nécessite une colonne "reference" absente du
# dataset actuel (voir explication en tête de fichier).
DEFAULT_METRICS = [faithfulness, answer_relevancy]


def build_dataset(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: List[str] = None,
):
    """
    Construit un dataset RAGAS à partir de trois (ou quatre) listes parallèles.
    - questions      : les questions posées
    - answers        : les réponses générées par le système
    - contexts       : pour chaque question, la liste des passages récupérés
    - ground_truths  : optionnel, réponse de référence. Nécessaire uniquement
                        si vous réactivez context_precision ou context_recall.
    """
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    if ground_truths is not None:
        data["reference"] = ground_truths  # nom de colonne attendu par ragas
    return Dataset.from_dict(data)


def get_llm_and_embeddings():
    llm = LangchainLLMWrapper(Ollama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL))
    embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=EMBED_MODEL)
    )
    return llm, embeddings


def evaluate_metrics(dataset: Dataset, metrics=None):
    """
    Calcule les métriques RAGAS sur le dataset fourni.
    Par défaut : faithfulness, answer_relevancy (aucune référence requise).
    """
    if metrics is None:
        metrics = DEFAULT_METRICS
    llm, embeddings = get_llm_and_embeddings()
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
    )
    return result


if __name__ == "__main__":
    questions = ["What is the definition of a vector space?"]
    answers = [
        "A vector space is a set equipped with vector addition and scalar "
        "multiplication that satisfies specific axioms."
    ]
    contexts = [
        [
            "A vector space over a field F is a set V together with two "
            "operations that satisfy the eight axioms listed below."
        ]
    ]
    ds = build_dataset(questions, answers, contexts)
    res = evaluate_metrics(ds)
    print(res)
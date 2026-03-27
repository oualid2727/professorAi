from typing import List

from datasets import Dataset
from ragas.metrics import faithfulness
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings


def build_dataset(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
):
    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
    )


def evaluate_faithfulness(dataset: Dataset):
    llm = LangchainLLMWrapper(Ollama(base_url="http://ollama:11434", model="llama3"))
    embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(base_url="http://ollama:11434", model="llama3"))

    result = evaluate(
        dataset,
        metrics=[faithfulness],
        llm=llm,
        embeddings=embeddings,
    )
    return result


if __name__ == "__main__":
    questions = ["What is the definition of a vector space?"]
    answers = [
        "A vector space is a set equipped with vector addition and scalar multiplication that satisfies specific axioms."
    ]
    contexts = [
        [
            "A vector space over a field F is a set V together with two operations that satisfy the eight axioms listed below."
        ]
    ]
    ds = build_dataset(questions, answers, contexts)
    res = evaluate_faithfulness(ds)
    print(res)
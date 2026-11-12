"""
Evaluation & Benchmarking Framework
===================================
Evaluates the Energy Operations Document RAG Assistant against quantitative metrics:
- Retrieval: Recall@1, Recall@3, MRR (Mean Reciprocal Rank)
- Faithfulness / Groundedness: Average Groundedness Score, Hallucination Rate
- Safety: Out-of-Domain Refusal Accuracy
- Latency: Average retrieval, generation, and end-to-end latency
"""

import time
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from backend.rag_engine import EnergyOpsRAGPipeline

BASE_DIR = Path(__file__).resolve().parents[1]
QUESTIONS_CSV = BASE_DIR / "data" / "questions.csv"


class RAGEvaluator:
    def __init__(self, pipeline: EnergyOpsRAGPipeline = None):
        self.pipeline = pipeline or EnergyOpsRAGPipeline()

    def run_benchmark(self, csv_path: Path = QUESTIONS_CSV) -> Dict[str, Any]:
        df = pd.read_csv(csv_path)
        total_queries = len(df)

        hits_at_1 = 0
        hits_at_3 = 0
        reciprocal_ranks = []
        groundedness_scores = []
        hallucination_count = 0
        latencies = []
        refusal_tests = 0
        refusal_successes = 0

        query_details: List[Dict[str, Any]] = []

        for idx, row in df.iterrows():
            q = str(row["question"])
            expected_src = str(row["expected_source"]).strip()
            domain = str(row.get("domain", "General"))

            t0 = time.perf_counter()
            response = self.pipeline.answer_query(q, top_k=3)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed_ms)

            # Check if this is an out-of-domain test
            is_ood = (expected_src == "NONE" or "UNANSWERABLE" in str(row.get("expected_keywords", "")))

            if is_ood:
                refusal_tests += 1
                if response.is_refusal:
                    refusal_successes += 1
                hit_1 = False
                hit_3 = False
                rr = 0.0
            else:
                retrieved_sources = [c["doc_name"] for c in response.retrieved_chunks]

                # Check Hit@1
                hit_1 = len(retrieved_sources) > 0 and retrieved_sources[0] == expected_src
                if hit_1:
                    hits_at_1 += 1

                # Check Hit@3
                hit_3 = expected_src in retrieved_sources
                if hit_3:
                    hits_at_3 += 1

                # Calculate Reciprocal Rank
                if expected_src in retrieved_sources:
                    rank = retrieved_sources.index(expected_src) + 1
                    rr = 1.0 / rank
                else:
                    rr = 0.0
                reciprocal_ranks.append(rr)

            if not response.is_refusal:
                groundedness_scores.append(response.groundedness_score)
                if response.hallucination_detected:
                    hallucination_count += 1

            query_details.append({
                "id": idx + 1,
                "question": q,
                "domain": domain,
                "expected_source": expected_src,
                "retrieved_top_source": response.retrieved_chunks[0]["doc_name"] if response.retrieved_chunks else "None",
                "retrieval_score": response.retrieved_chunks[0]["combined_score"] if response.retrieved_chunks else 0.0,
                "hit_at_1": hit_1 if not is_ood else "N/A (OOD)",
                "hit_at_3": hit_3 if not is_ood else "N/A (OOD)",
                "is_refusal": response.is_refusal,
                "groundedness_score": response.groundedness_score,
                "status": response.verification_status,
                "latency_ms": round(elapsed_ms, 2)
            })

        # Summary calculations
        valid_eval_count = total_queries - refusal_tests
        recall_at_1 = (hits_at_1 / valid_eval_count) if valid_eval_count > 0 else 1.0
        recall_at_3 = (hits_at_3 / valid_eval_count) if valid_eval_count > 0 else 1.0
        mrr = (sum(reciprocal_ranks) / len(reciprocal_ranks)) if reciprocal_ranks else 1.0
        avg_groundedness = (sum(groundedness_scores) / len(groundedness_scores)) if groundedness_scores else 1.0
        hallucination_rate = (hallucination_count / len(groundedness_scores)) if groundedness_scores else 0.0
        refusal_acc = (refusal_successes / refusal_tests) if refusal_tests > 0 else 1.0

        return {
            "summary_metrics": {
                "total_queries_tested": total_queries,
                "in_domain_queries": valid_eval_count,
                "out_of_domain_queries": refusal_tests,
                "retrieval_recall_at_1": round(recall_at_1 * 100, 1),
                "retrieval_recall_at_3": round(recall_at_3 * 100, 1),
                "mean_reciprocal_rank_mrr": round(mrr, 3),
                "avg_groundedness_score_pct": round(avg_groundedness * 100, 1),
                "hallucination_rate_pct": round(hallucination_rate * 100, 1),
                "refusal_accuracy_pct": round(refusal_acc * 100, 1),
                "avg_latency_ms": round(sum(latencies) / len(latencies), 2)
            },
            "query_results": query_details
        }


if __name__ == "__main__":
    evaluator = RAGEvaluator()
    results = evaluator.run_benchmark()
    print("=== BENCHMARK RESULTS ===")
    for k, v in results["summary_metrics"].items():
        print(f"  {k}: {v}")


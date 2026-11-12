"""
Unit & Integration Test Suite for Energy Operations Document RAG Assistant
=========================================================================
Tests chunking, hybrid retrieval, citation binding, hallucination verifier,
and FastAPI endpoint logic directly.
"""

import unittest
from pathlib import Path

from backend.rag_engine import EnergyOpsRAGPipeline, DocumentProcessor, HybridRetriever, HallucinationVerifier
from backend.evaluator import RAGEvaluator
from backend.app import health_check, demo, run_case, query_rag, run_evaluation, list_documents, get_stats, Query


class TestEnergyOpsRAGPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pipeline = EnergyOpsRAGPipeline()

    def test_01_document_loading(self):
        """Verify that all operational SOP documents are loaded and non-empty."""
        self.assertGreaterEqual(len(self.pipeline.documents), 4)
        doc_names = [d["doc_name"] for d in self.pipeline.documents]
        self.assertIn("solar_inverter_sop.md", doc_names)
        self.assertIn("wind_turbine_maintenance.md", doc_names)
        self.assertIn("hydro_safety_brief.md", doc_names)
        self.assertIn("grid_operations_note.md", doc_names)

    def test_02_chunking_and_metadata(self):
        """Verify chunks have proper ID, doc_name, section, and text."""
        self.assertGreater(len(self.pipeline.chunks), 0)
        for chunk in self.pipeline.chunks:
            self.assertTrue(chunk.chunk_id)
            self.assertTrue(chunk.doc_name)
            self.assertTrue(chunk.text)
            self.assertGreater(chunk.word_count, 0)

    def test_03_inverter_derating_retrieval(self):
        """Verify Solar Inverter query retrieves solar_inverter_sop.md at rank 1."""
        query = "What should be checked for inverter derating?"
        retrieved, confident = self.pipeline.retriever.retrieve(query, top_k=3)
        self.assertTrue(confident)
        self.assertGreater(len(retrieved), 0)
        self.assertEqual(retrieved[0].chunk.doc_name, "solar_inverter_sop.md")
        self.assertIn("75C", retrieved[0].chunk.text)

    def test_04_gearbox_vibration_retrieval(self):
        """Verify Wind Gearbox query retrieves wind_turbine_maintenance.md at rank 1."""
        query = "How should gearbox vibration be investigated?"
        retrieved, confident = self.pipeline.retriever.retrieve(query, top_k=3)
        self.assertTrue(confident)
        self.assertGreater(len(retrieved), 0)
        self.assertEqual(retrieved[0].chunk.doc_name, "wind_turbine_maintenance.md")
        self.assertIn("vibration", retrieved[0].chunk.text.lower())

    def test_05_out_of_domain_refusal(self):
        """Verify out-of-domain unanswerable query triggers safety refusal."""
        query = "What is the recommended procedure for repairing a Boeing 737 aircraft engine?"
        resp = self.pipeline.answer_query(query)
        self.assertTrue(resp.is_refusal)
        self.assertIn("Safety & Compliance Notice", resp.answer)
        self.assertEqual(len(resp.citations), 0)

    def test_06_hallucination_verification_score(self):
        """Verify Groundedness score is high (>= 0.70) on valid in-domain answer."""
        query = "What safety steps are required before spillway gate inspection?"
        resp = self.pipeline.answer_query(query)
        self.assertFalse(resp.is_refusal)
        self.assertGreaterEqual(resp.groundedness_score, 0.70)
        self.assertEqual(resp.verification_status, "VERIFIED_GROUNDED")
        self.assertIn("hydro_safety_brief.md", [c["source"] for c in resp.citations])

    def test_07_api_health_and_demo(self):
        """Test health_check() and demo() endpoint functions directly."""
        health = health_check()
        self.assertEqual(health["status"], "healthy")
        self.assertGreaterEqual(health["total_documents"], 4)

        demo_data = demo()
        self.assertEqual(demo_data["case_study"], "Energy Operations Document RAG Assistant")
        self.assertIn("sample_queries", demo_data)

    def test_08_api_run_case(self):
        """Test run_case() function for starter code compatibility."""
        res = run_case(Query(text="What should be checked for inverter derating?"))
        self.assertIn("result", res)
        self.assertIn("citations", res)
        self.assertGreater(len(res["citations"]), 0)

    def test_09_api_query_full(self):
        """Test query_rag() function returning full RAGResponse."""
        resp = query_rag(Query(text="What should be reviewed for generation deviation before escalation?", top_k=3))
        self.assertEqual(resp.verification_status, "VERIFIED_GROUNDED")
        self.assertGreaterEqual(resp.groundedness_score, 0.70)
        self.assertIn("latency_breakdown_ms", resp.model_dump())

    def test_10_api_evaluate_benchmark(self):
        """Test run_evaluation() benchmark endpoint function."""
        eval_res = run_evaluation()
        metrics = eval_res["summary_metrics"]
        self.assertEqual(metrics["retrieval_recall_at_1"], 100.0)
        self.assertEqual(metrics["mean_reciprocal_rank_mrr"], 1.0)
        self.assertEqual(metrics["hallucination_rate_pct"], 0.0)
        self.assertEqual(metrics["refusal_accuracy_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()

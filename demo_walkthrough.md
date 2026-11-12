# Demo Walkthrough & Verification Notes

**Project:** Greenko Energy Operations Document RAG Assistant  
**Target:** Live Interview Case Study Presentation & Evaluation

---

## 1. Quick Verification Commands

Before presenting to the interviewer, run these two quick terminal commands to ensure the environment is 100% verified:

### Step 1: Run Automated Test Suite (10/10 Passed)
```bash
.venv\Scripts\python.exe -m unittest -v backend.test_pipeline
```
**Expected Output:**
```
test_01_document_loading ... ok
test_02_chunking_and_metadata ... ok
test_03_inverter_derating_retrieval ... ok
test_04_gearbox_vibration_retrieval ... ok
test_05_out_of_domain_refusal ... ok
test_06_hallucination_verification_score ... ok
test_07_api_health_and_demo ... ok
test_08_api_run_case ... ok
test_09_api_query_full ... ok
test_10_api_evaluate_benchmark ... ok

----------------------------------------------------------------------
Ran 10 tests in 0.130s

OK
```

### Step 2: Run Benchmark Suite
```bash
.venv\Scripts\python.exe -m backend.evaluator
```
**Expected Output:**
```
=== BENCHMARK RESULTS ===
  total_queries_tested: 7
  in_domain_queries: 6
  out_of_domain_queries: 1
  retrieval_recall_at_1: 100.0
  retrieval_recall_at_3: 100.0
  mean_reciprocal_rank_mrr: 1.0
  avg_groundedness_score_pct: 100.0
  hallucination_rate_pct: 0.0
  refusal_accuracy_pct: 100.0
  avg_latency_ms: 2.4
```

---

## 2. Live Application Demo Script (Tab-by-Tab)

Launch the app:
```bash
.venv\Scripts\python.exe -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```
Open **`http://localhost:8000`** in Chrome or Edge.

### Demo Action 1: In-Domain Operational Query (Solar Inverter Derating)
1. In the **Operations Assistant** tab, click the quick pill:  
   `☀️ Solar Inverter Derating` (`What should be checked for inverter derating?`).
2. Click **Run RAG Pipeline**.
3. **What to point out to the interviewer:**
   - **Grounded Operational Guidance:** Highlights the exact checks: cooling fans, dust filters, ambient temperature, DC input imbalance, priority maintenance ticket, SCADA alarms.
   - **Citation Badge:** Explicit provenance tag `📄 Source: solar_inverter_sop.md`.
   - **Factuality & Hallucination Audit:**
     - Groundedness Score = `100%`.
     - Status = `VERIFIED_GROUNDED`.
     - Claim breakdown shows each statement verified against the document text with high overlap.
   - **Retrieved SOP Evidence Chunks:** Shows Rank #1 `solar_inverter_sop.md` with Combined Score = `0.310` (Lexical: `0.287`, Semantic: `0.329`).
   - **Latency Telemetry:** Retrieval: `~1.1 ms`, Generation: `~0.3 ms`, Verification: `~0.4 ms`, Total: `~1.8 ms`.

### Demo Action 2: Out-of-Domain Adversarial Query (Safety Refusal)
1. Click the red pill:  
   `🚫 Out-of-Domain Safety Test` (`What is the recommended procedure for repairing a Boeing 737 aircraft engine?`).
2. Click **Run RAG Pipeline**.
3. **What to point out to the interviewer:**
   - Instead of hallucinating advice for aircraft maintenance, the system triggers the **Safety Refusal Boundary**:
     > *"⚠️ Safety & Compliance Notice: The provided renewable energy operational documents do not contain sufficient verified evidence to answer this query. In accordance with operational safety standards, unverified guidance is withheld."*
   - Explains that the combined retrieval score (`0.187`) was below the `0.25` safety cutoff.
   - Emphasize that in power plants, knowing when NOT to answer is as critical as answering correctly.

### Demo Action 3: Knowledge Base Explorer
1. Click the **Knowledge Base Explorer** tab.
2. Show the indexed SOPs across Wind, Solar, Hydro, Grid, BESS, and Substation.
3. Click "Query this SOP" on any card to demonstrate that any asset procedure can be queried seamlessly.

### Demo Action 4: Quantitative Benchmark Audit
1. Click the **Evaluation Benchmark** tab.
2. Click **Run Full Benchmark**.
3. Show the metric scorecards (Recall@1: 100%, MRR: 1.000, Hallucination: 0.0%).
4. Scroll through the live table showing all test cases from `data/questions.csv`.

---

## 3. Command-Line API Testing (Curl)

If the interviewer asks to see the raw API responses:

### Test In-Domain Query:
```bash
curl -X POST "http://localhost:8000/api/query" \
     -H "Content-Type: application/json" \
     -d "{\"text\": \"What should be checked for inverter derating?\", \"top_k\": 3}"
```

### Test Out-of-Domain Query:
```bash
curl -X POST "http://localhost:8000/api/query" \
     -H "Content-Type: application/json" \
     -d "{\"text\": \"How do you fix a Boeing 737 jet engine?\", \"top_k\": 3}"
```

### Test Benchmark API:
```bash
curl -X POST "http://localhost:8000/api/evaluate"
```

---

## 4. Railway Deployment Steps

To complete the required Railway deployment:

1. Create a repository on GitHub (e.g. `greenko-energy-ops-rag`).
2. Push the local commits:
   ```bash
   git remote add origin https://github.com/<your-username>/greenko-energy-ops-rag.git
   git branch -M main
   git push -u origin main
   ```
3. On [Railway.app](https://railway.app/), click **"New Project"** $\to$ **"Deploy from GitHub repo"**.
4. Select the repository.
5. Railway reads `railway.json` and `Procfile` automatically and deploys the service on an external URL (e.g., `https://greenko-energy-ops-rag.up.railway.app`).


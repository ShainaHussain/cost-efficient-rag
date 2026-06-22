import os
import json
import time
import math
from dotenv import load_dotenv
from groq import Groq
from rag import retrieve, generate

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3-8b-8192")
TOP_K = int(os.getenv("TOP_K", 5))

client = Groq(api_key=GROQ_API_KEY)

# ----------------------------------------------------------------
# RETRIEVAL METRICS
# ----------------------------------------------------------------

def hit_rate(retrieved_sources: list, relevant_sources: list) -> float:
    """Did at least one relevant chunk appear in top-k?"""
    return 1.0 if any(s in retrieved_sources for s in relevant_sources) else 0.0

def mrr(retrieved_sources: list, relevant_sources: list) -> float:
    """Mean Reciprocal Rank — where did the first relevant chunk appear?"""
    for i, source in enumerate(retrieved_sources):
        if source in relevant_sources:
            return 1.0 / (i + 1)
    return 0.0

def ndcg(retrieved_sources: list, relevant_sources: list, k: int) -> float:
    """Normalized Discounted Cumulative Gain@k"""
    def dcg(sources):
        score = 0.0
        for i, s in enumerate(sources[:k]):
            relevance = 1.0 if s in relevant_sources else 0.0
            score += relevance / math.log2(i + 2)
        return score

    actual_dcg = dcg(retrieved_sources)
    # ideal: all relevant docs at the top
    ideal_sources = [s for s in retrieved_sources if s in relevant_sources]
    ideal_sources += [s for s in retrieved_sources if s not in relevant_sources]
    ideal_dcg = dcg(ideal_sources)

    return round(actual_dcg / ideal_dcg, 4) if ideal_dcg > 0 else 0.0

def context_precision(retrieved_sources: list, relevant_sources: list) -> float:
    """What fraction of retrieved chunks are actually relevant?"""
    if not retrieved_sources:
        return 0.0
    relevant_count = sum(1 for s in retrieved_sources if s in relevant_sources)
    return round(relevant_count / len(retrieved_sources), 4)

# ----------------------------------------------------------------
# ANSWER METRICS (LLM-as-judge)
# ----------------------------------------------------------------

def judge_answer(question: str, answer: str, context: str) -> dict:
    """
    Use LLM to judge faithfulness and relevance.
    Returns scores 1-5 for each with rationale.
    """
    prompt = f"""You are an evaluation judge. Score the answer below on two criteria.
Respond ONLY with valid JSON, no extra text, no markdown.

Question: {question}
Context provided to the model: {context}
Answer given: {answer}

Return exactly this JSON structure:
{{
  "faithfulness": {{
    "score": <integer 1-5>,
    "rationale": "<one sentence>"
  }},
  "relevance": {{
    "score": <integer 1-5>,
    "rationale": "<one sentence>"
  }}
}}

Scoring guide:
Faithfulness (is the answer grounded in the context, no hallucinations?):
  5 = fully grounded, every claim traceable to context
  3 = mostly grounded, minor unsupported detail
  1 = makes claims not present in context at all

Relevance (does the answer actually address the question?):
  5 = directly and completely answers the question
  3 = partially answers, misses some aspect
  1 = does not answer the question"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300
        )
        raw = response.choices[0].message.content.strip()
        # strip markdown fences if model adds them
        raw = raw.replace("```json", "").replace("```", "").strip()
        scores = json.loads(raw)
        return {
            "faithfulness": scores["faithfulness"]["score"],
            "faithfulness_rationale": scores["faithfulness"]["rationale"],
            "relevance": scores["relevance"]["score"],
            "relevance_rationale": scores["relevance"]["rationale"],
            "judge_tokens": response.usage.total_tokens
        }
    except Exception as e:
        return {
            "faithfulness": 0,
            "faithfulness_rationale": f"Judge failed: {e}",
            "relevance": 0,
            "relevance_rationale": f"Judge failed: {e}",
            "judge_tokens": 0
        }

# ----------------------------------------------------------------
# MAIN EVAL LOOP
# ----------------------------------------------------------------

def evaluate(eval_path: str = "./eval_dataset.json", k: int = TOP_K):
    with open(eval_path, "r") as f:
        dataset = json.load(f)

    if not dataset:
        print("eval_dataset.json is empty. Add QA pairs first.")
        return

    results = []
    hit_rates, mrrs, ndcgs, precisions = [], [], [], []
    faithfulness_scores, relevance_scores = [], []
    total_latency = []

    print(f"Running evaluation on {len(dataset)} questions...\n")

    for i, item in enumerate(dataset):
        question = item["question"]
        relevant_sources = item["relevant_sources"]  # list of filenames

        print(f"[{i+1}/{len(dataset)}] {question[:60]}...")

        # retrieve
        chunks, retrieval_latency = retrieve(question, k=k)
        retrieved_sources = [c["source"] for c in chunks]

        # retrieval metrics
        hr = hit_rate(retrieved_sources, relevant_sources)
        mrr_score = mrr(retrieved_sources, relevant_sources)
        ndcg_score = ndcg(retrieved_sources, relevant_sources, k)
        cp = context_precision(retrieved_sources, relevant_sources)

        hit_rates.append(hr)
        mrrs.append(mrr_score)
        ndcgs.append(ndcg_score)
        precisions.append(cp)

        # generate answer
        result = generate(question, k=k)
        answer = result["answer"]
        total_latency.append(result["total_latency_ms"])

        # build context string for judge
        context = "\n".join([c["text"][:300] for c in chunks])

        # answer metrics
        judge = judge_answer(question, answer, context)
        faithfulness_scores.append(judge["faithfulness"])
        relevance_scores.append(judge["relevance"])

        results.append({
            "question": question,
            "answer": answer,
            "relevant_sources": relevant_sources,
            "retrieved_sources": retrieved_sources,
            "hit_rate": hr,
            "mrr": mrr_score,
            "ndcg": ndcg_score,
            "context_precision": cp,
            "faithfulness": judge["faithfulness"],
            "faithfulness_rationale": judge["faithfulness_rationale"],
            "relevance": judge["relevance"],
            "relevance_rationale": judge["relevance_rationale"],
            "total_latency_ms": result["total_latency_ms"],
            "token_usage": result["token_usage"]
        })

        time.sleep(0.5)  # avoid rate limiting

    # aggregate
    n = len(results)
    summary = {
        "total_questions": n,
        "retrieval": {
            "hit_rate": round(sum(hit_rates) / n, 4),
            "mrr": round(sum(mrrs) / n, 4),
            "ndcg_at_k": round(sum(ndcgs) / n, 4),
            "context_precision": round(sum(precisions) / n, 4)
        },
        "answer": {
            "avg_faithfulness": round(sum(faithfulness_scores) / n, 4),
            "avg_relevance": round(sum(relevance_scores) / n, 4)
        },
        "latency": {
            "p50_ms": round(sorted(total_latency)[n // 2], 2),
            "p95_ms": round(sorted(total_latency)[int(n * 0.95)], 2),
            "avg_ms": round(sum(total_latency) / n, 2)
        },
        "per_case": results
    }

    with open("./results/eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n========== EVALUATION SUMMARY ==========")
    print(f"Hit Rate:          {summary['retrieval']['hit_rate']}")
    print(f"MRR:               {summary['retrieval']['mrr']}")
    print(f"nDCG@{k}:           {summary['retrieval']['ndcg_at_k']}")
    print(f"Context Precision: {summary['retrieval']['context_precision']}")
    print(f"Avg Faithfulness:  {summary['answer']['avg_faithfulness']} / 5")
    print(f"Avg Relevance:     {summary['answer']['avg_relevance']} / 5")
    print(f"p50 Latency:       {summary['latency']['p50_ms']}ms")
    print(f"p95 Latency:       {summary['latency']['p95_ms']}ms")
    print(f"\nFull results saved to ./results/eval_results.json")

if __name__ == "__main__":
    evaluate()
"""
Evaluation Script — Email Generation Assistant
Runs 10 scenarios × 2 models and scores each on 3 custom metrics.

Metrics:
  1. Fact Recall       — fraction of key facts present in the generated email
  2. Tone Alignment    — LLM-as-judge score (0.0 to 1.0)
  3. Conciseness       — penalises emails that are too short or too long
"""

import json
import re
import csv
import os
from dotenv import load_dotenv
from llm import generate_email, MODEL_A, MODEL_B
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ── Judge LLM (used only for Tone Alignment metric) ───────────────────
_judge = ChatGroq(
    model=MODEL_A,
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

_tone_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert communication evaluator. "
     "Score how well the email matches the requested tone. "
     "Reply with ONLY a decimal number between 0.0 and 1.0. Nothing else."),
    ("human",
     "Requested tone: {tone}\n\nEmail:\n{email}\n\nScore (0.0 = completely wrong tone, 1.0 = perfect match):"),
])

# ── Metric 1: Fact Recall ─────────────────────────────────────────────
def fact_recall(generated: str, facts: str) -> float:
    """
    Logic: Split facts into individual keywords (ignoring stopwords).
    A fact is 'recalled' if the majority of its keywords appear in the email.
    This handles paraphrasing — e.g. 'Met on Monday' → checks 'monday', 'met'.
    Score = recalled_facts / total_facts.
    """
    STOPWORDS = {"a","an","the","is","are","was","were","for","on","at",
                 "to","of","in","and","or","but","with","this","that",
                 "it","be","as","by","from","has","have","had","will"}

    fact_list = [
        f.strip("•-* \t")
        for f in facts.strip().splitlines()
        if f.strip()
    ]
    if not fact_list:
        return 1.0

    generated_lower = generated.lower()
    hits = 0

    for fact in fact_list:
        # Extract meaningful keywords from the fact
        keywords = [
            w.strip(".,!?:;\"'()").lower()
            for w in fact.split()
            if w.strip(".,!?:;\"'()").lower() not in STOPWORDS
            and len(w.strip(".,!?:;\"'()")) > 2
        ]
        if not keywords:
            hits += 1
            continue
        # Fact is recalled if more than half its keywords appear in the email
        keyword_hits = sum(1 for kw in keywords if kw in generated_lower)
        if keyword_hits >= len(keywords) * 0.5:
            hits += 1

    return round(hits / len(fact_list), 3)

# ── Metric 2: Tone Alignment (LLM-as-Judge) ──────────────────────────
def tone_alignment(generated: str, tone: str) -> float:
    """
    Logic: Send the generated email + requested tone to a judge LLM.
    Ask it to score from 0.0 to 1.0. Parse the float from the response.
    Clamp to [0.0, 1.0] to handle any edge cases.
    """
    try:
        chain = _tone_prompt | _judge | StrOutputParser()
        raw = chain.invoke({"tone": tone, "email": generated}).strip()
        score = float(re.search(r"[\d.]+", raw).group())
        return round(min(max(score, 0.0), 1.0), 3)
    except Exception:
        return 0.5  # neutral fallback if judge fails

# ── Metric 3: Conciseness ─────────────────────────────────────────────
def conciseness(generated: str, facts: str) -> float:
    """
    Logic: Ideal email length scales with number of facts.
    Base target: 80–150 words. Each fact adds ~15 words of allowance.
    Below 50 words → penalise (too brief). Above ideal_max → penalise.
    Score = 1.0 inside the ideal window, linear decay outside.
    """
    words = len(generated.split())
    fact_count = len([f for f in facts.splitlines() if f.strip()])
    ideal_max = 150 + fact_count * 15

    if words < 50:
        return round(words / 50, 3)
    elif words <= ideal_max:
        return 1.0
    else:
        penalty = (words - ideal_max) / ideal_max
        return round(max(0.0, 1.0 - penalty), 3)

# ── Runner ────────────────────────────────────────────────────────────
def run_evaluation(model_id: str, model_label: str, scenarios: list) -> list:
    rows = []
    for s in scenarios:
        print(f"  [{model_label}] Scenario {s['id']}: {s['intent'][:40]}...")
        try:
            generated = generate_email(
                intent=s["intent"],
                facts=s["facts"],
                tone=s["tone"],
                model_id=model_id
            )
            fr = fact_recall(generated, s["facts"])
            ta = tone_alignment(generated, s["tone"])
            co = conciseness(generated, s["facts"])
            avg = round((fr + ta + co) / 3, 3)

            rows.append({
                "scenario_id":    s["id"],
                "model":          model_label,
                "intent":         s["intent"],
                "tone":           s["tone"],
                "fact_recall":    fr,
                "tone_alignment": ta,
                "conciseness":    co,
                "average_score":  avg,
                "generated_email": generated[:400].replace("\n", " "),
            })
        except Exception as e:
            print(f"  [ERROR] Scenario {s['id']}: {e}")
            rows.append({
                "scenario_id":    s["id"],
                "model":          model_label,
                "intent":         s["intent"],
                "tone":           s["tone"],
                "fact_recall":    0,
                "tone_alignment": 0,
                "conciseness":    0,
                "average_score":  0,
                "generated_email": f"ERROR: {str(e)}",
            })
    return rows

# ── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with open("test_data.json", encoding="utf-8") as f:
        scenarios = json.load(f)

    all_rows = []

    print("\n=== Running Model A (llama-3.3-70b-versatile) ===")
    all_rows += run_evaluation(MODEL_A, "Model_A", scenarios)

    print("\n=== Running Model B (llama-3.1-8b-instant) ===")
    all_rows += run_evaluation(MODEL_B, "Model_B", scenarios)

    # Write CSV
    fieldnames = [
        "scenario_id", "model", "intent", "tone",
        "fact_recall", "tone_alignment", "conciseness",
        "average_score", "generated_email"
    ]
    with open("results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # Print summary
    print("\n=== SUMMARY ===")
    for label in ["Model_A", "Model_B"]:
        subset = [r for r in all_rows if r["model"] == label]
        print(f"\n{label}:")
        for metric in ["fact_recall", "tone_alignment", "conciseness", "average_score"]:
            avg = sum(r[metric] for r in subset) / len(subset)
            print(f"  {metric:20s}: {avg:.3f}")

    print("\n✅ Results saved to results.csv")
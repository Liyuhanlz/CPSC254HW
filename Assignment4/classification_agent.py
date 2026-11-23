"""
CPSC 254 - Assignment 4, Part 2: LLM-based classification agent
(team version, matched to the team's Part 1 `topic_classifier.py`)

A zero-shot LangChain agent (langchain.agents.create_agent) running the "gpt-oss" model on the
NRP Nautilus Hypercluster classifies one AG News article as World, Sports, Business, or Sci/Tech
and returns validated JSON: {"category": ..., "justification": ...}.

`--evaluate` compares the agent with the best Part 1 classifier on the SAME 200 test articles
that Part 1 saved in `sample_idx_200.npy` (stratified: 50 per category, seed 42). Part 1 chose its
best configuration using accuracy on the test set EXCLUDING these 200 articles, so the
comparison sample was not used to tune the Part 1 model.

Files this script needs (created by running topic_classifier.py first):
    best_topic_classifier.joblib   {"vectorizer", "model" (sklearn MLPClassifier), "class_names"}
    sample_idx_200.npy             the 200 shared test-set indices
Both are looked for in the current folder, this script's folder, and its parent folder.
A Nautilus API key must be in a .env file:  OPENAI_API_KEY=...

Usage
    python classification_agent.py --text "Wall St. Bears Claw Back Into the Black ..."
    python classification_agent.py --evaluate              # all 200 shared examples
    python classification_agent.py --evaluate --limit 20   # quick test on the first 20
"""

import argparse
import csv
import json
import os
import platform
import re
import sys
import time
import urllib.request

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

load_dotenv()

NAUTILUS_BASE_URL = "https://ellm.nrp-nautilus.io/v1"
MODEL_NAME = "gpt-oss"
CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]  # same order as Part 1 (labels 0-3)
REQUIRED_FIELDS = {"category", "justification"}
SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# Zero-shot prompt (category descriptions only; no labelled examples)
# ----------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a news topic classification agent for the AG News dataset.
Classify the news article given by the user into exactly ONE of these categories:
- World: international news, politics, government, conflicts, elections, diplomacy.
- Sports: sports events, teams, athletes, games, leagues, tournaments.
- Business: companies, markets, economy, finance, earnings, trade, industry.
- Sci/Tech: science, technology, computing, internet, software, space, health research.

Respond with ONLY a valid JSON object with exactly two fields and nothing else:
{"category": "<one of: World, Sports, Business, Sci/Tech>", "justification": "<one short sentence>"}
The justification must be brief (at most 25 words) and cite the key evidence in the article.
Do not include your reasoning steps, markdown, code fences, or any extra fields."""

USER_TEMPLATE = "Article:\n{article}"

RETRY_MESSAGE = (
    "Your previous reply was invalid: {error}. Reply again with ONLY the JSON object "
    '{{"category": ..., "justification": ...}} where category is exactly one of '
    "World, Sports, Business, Sci/Tech."
)


class ValidationError(ValueError):
    pass


# ----------------------------------------------------------------------------
# Part 2 (1): the agent
# ----------------------------------------------------------------------------
def build_agent(reasoning_effort="low"):
    """Create the LangChain agent wrapping gpt-oss on Nautilus."""
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set. Put your Nautilus token in a .env file.")
    llm = ChatOpenAI(
        model=MODEL_NAME,
        base_url=NAUTILUS_BASE_URL,
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0,
        timeout=120,
        max_retries=2,
        reasoning_effort=reasoning_effort,  # keep gpt-oss's hidden reasoning short
    )
    # A single classification decision needs no external tools.
    return create_agent(model=llm, tools=[], system_prompt=SYSTEM_PROMPT)


def parse_and_validate(raw):
    """Return {'category', 'justification'} or raise ValidationError."""
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()  # tolerate code fences
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValidationError("no JSON object found")
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid JSON ({exc.msg})") from exc
    if not isinstance(obj, dict) or set(obj.keys()) != REQUIRED_FIELDS:
        got = sorted(obj) if isinstance(obj, dict) else type(obj).__name__
        raise ValidationError(f"fields must be exactly {sorted(REQUIRED_FIELDS)}, got {got}")
    canon = {c.lower(): c for c in CLASS_NAMES}
    cat = str(obj["category"]).strip()
    if cat.lower() not in canon:
        raise ValidationError(f"category '{cat}' is not one of {CLASS_NAMES}")
    just = str(obj["justification"]).strip()
    if not just:
        raise ValidationError("justification is empty")
    return {"category": canon[cat.lower()], "justification": just}


def _usage(messages):
    tokens = {"input_tokens": 0, "output_tokens": 0}
    for m in messages:
        meta = getattr(m, "usage_metadata", None) or {}
        tokens["input_tokens"] += meta.get("input_tokens", 0)
        tokens["output_tokens"] += meta.get("output_tokens", 0)
    return tokens


def classify_article(article, agent=None, max_attempts=2):
    """Classify one article. Retries once on invalid output, then returns a clear
    validation error instead of raising."""
    agent = agent or build_agent()
    messages = [("user", USER_TEMPLATE.format(article=article))]
    usage = {"input_tokens": 0, "output_tokens": 0}
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            result = agent.invoke({"messages": messages})
        except Exception as exc:  # network / server error
            last_error = f"LLM call failed: {type(exc).__name__}: {exc}"
            continue
        out_msgs = result["messages"]
        for k, v in _usage(out_msgs[len(messages):]).items():  # count only new messages
            usage[k] += v
        raw = out_msgs[-1].content
        if isinstance(raw, list):  # some providers return content blocks
            raw = "".join(b.get("text", "") for b in raw if isinstance(b, dict))
        try:
            parsed = parse_and_validate(raw)
            parsed.update({"valid": True, "attempts": attempt, "usage": usage})
            return parsed
        except ValidationError as exc:
            last_error = str(exc)
            # feed the bad answer back and ask for a corrected one (retry once)
            messages = list(out_msgs) + [("user", RETRY_MESSAGE.format(error=last_error))]
    return {"category": None, "justification": None, "valid": False,
            "attempts": max_attempts, "usage": usage,
            "error": f"Validation failed after {max_attempts} attempts: {last_error}"}


# ----------------------------------------------------------------------------
# Part 2 (2): comparison with the team's best Part 1 classifier
# ----------------------------------------------------------------------------
def find_file(name):
    for folder in (os.getcwd(), HERE, os.path.dirname(HERE)):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    return None


def load_test_split():
    """Return (labels 0-3, texts) for the official 7,600-article test split, in the SAME
    order and text format as Part 1 (torchtext.datasets.AG_NEWS)."""
    try:
        import warnings
        warnings.filterwarnings("ignore")
        import torchtext
        if hasattr(torchtext, "disable_torchtext_deprecation_warning"):
            torchtext.disable_torchtext_deprecation_warning()
        from torchtext.datasets import AG_NEWS
        rows = list(AG_NEWS(split="test"))
        print("Loaded the AG News test split with torchtext (same as Part 1).")
        return [label - 1 for label, _ in rows], [text for _, text in rows]
    except Exception as exc:
        print(f"torchtext unavailable ({type(exc).__name__}); reading the same official CSV "
              "that torchtext uses, with the identical text format.")
    url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
    path = os.path.join(os.getcwd(), "data", "ag_news_test.csv")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path)
    labels, texts = [], []
    with open(path, encoding="utf-8") as fh:
        for row in csv.reader(fh):
            labels.append(int(row[0]) - 1)
            texts.append(" ".join(row[1:]))  # exactly what torchtext's AG_NEWS returns
    return labels, texts


def load_sample_indices(labels):
    import numpy as np

    path = find_file("sample_idx_200.npy")
    if path:
        idx = np.load(path)
        print(f"Using the shared Part 1 sample: {path}")
    else:  # same rule as topic_classifier.py
        print("sample_idx_200.npy not found; regenerating it with Part 1's rule "
              "(RandomState(42), 50 per class).")
        y = np.array(labels)
        rng = np.random.RandomState(SEED)
        idx = np.concatenate([rng.choice(np.where(y == c)[0], 50, replace=False) for c in range(4)])
    return [int(i) for i in idx]


def load_part1_model():
    import joblib

    path = find_file("best_topic_classifier.joblib")
    if not path:
        sys.exit("best_topic_classifier.joblib not found. Run topic_classifier.py (Part 1) first, "
                 "then run this script from the same folder.")
    bundle = joblib.load(path)
    print(f"Loaded the Part 1 model: {path}")
    return bundle["vectorizer"], bundle["model"], bundle.get("class_names", CLASS_NAMES)


def evaluate(limit=None):
    import numpy as np
    from sklearn.metrics import accuracy_score

    labels, texts = load_test_split()
    sample_idx = load_sample_indices(labels)
    if limit:
        sample_idx = sample_idx[:limit]
        print(f"*** Quick test: only the first {limit} of the 200 shared examples ***")
    y_true = [CLASS_NAMES[labels[i]] for i in sample_idx]
    sample_texts = [texts[i] for i in sample_idx]
    print(f"Sample size: {len(sample_idx)} (per class: "
          f"{ {c: y_true.count(c) for c in CLASS_NAMES} })")

    # --- Part 1 model: no retraining, no tuning on these examples ---
    vec, mlp, names = load_part1_model()
    t0 = time.time()
    mlp_pred = [names[int(p)] for p in mlp.predict(vec.transform(sample_texts))]
    mlp_time = time.time() - t0
    print(f"Part 1 model: {type(mlp).__name__}, hidden layers {getattr(mlp, 'hidden_layer_sizes', '?')}, "
          f"{len(vec.vocabulary_):,} features ({type(vec).__name__})")

    # --- zero-shot agent ---
    agent = build_agent()
    rows, latencies = [], []
    in_tok = out_tok = 0
    eval_start = time.time()
    for k, (i, text, truth) in enumerate(zip(sample_idx, sample_texts, y_true), 1):
        t = time.time()
        res = classify_article(text, agent)
        lat = time.time() - t
        latencies.append(lat)
        in_tok += res["usage"]["input_tokens"]
        out_tok += res["usage"]["output_tokens"]
        rows.append({"test_index": i, "true": truth, "mlp_pred": mlp_pred[k - 1],
                     "agent_pred": res["category"] or "INVALID", "valid": res["valid"],
                     "attempts": res["attempts"], "latency_s": round(lat, 3),
                     "justification": res.get("justification") or res.get("error"),
                     "text": text})
        if k % 20 == 0 or k == len(sample_idx):
            acc = np.mean([r["true"] == r["agent_pred"] for r in rows])
            print(f"  {k}/{len(sample_idx)} done, running agent acc={acc:.3f}, "
                  f"avg latency={np.mean(latencies):.2f}s", flush=True)
    total_time = time.time() - eval_start

    agent_acc = accuracy_score(y_true, [r["agent_pred"] for r in rows])
    mlp_acc = accuracy_score(y_true, mlp_pred)
    n_invalid = sum(not r["valid"] for r in rows)
    n_retry = sum(r["attempts"] > 1 for r in rows)
    both = sum(r["agent_pred"] != r["true"] and r["mlp_pred"] != r["true"] for r in rows)
    agent_only = sum(r["agent_pred"] != r["true"] and r["mlp_pred"] == r["true"] for r in rows)
    mlp_only = sum(r["agent_pred"] == r["true"] and r["mlp_pred"] != r["true"] for r in rows)

    with open("part2_predictions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {
        "sample_size": len(sample_idx), "sample_file": "sample_idx_200.npy", "seed": SEED,
        "mlp_model": f"{type(mlp).__name__} {getattr(mlp, 'hidden_layer_sizes', '')}",
        "mlp_vectorizer": type(vec).__name__, "mlp_accuracy": mlp_acc,
        "mlp_total_inference_s": mlp_time,
        "agent_model": MODEL_NAME, "agent_accuracy": agent_acc,
        "agent_avg_latency_s": float(np.mean(latencies)),
        "agent_median_latency_s": float(np.median(latencies)),
        "agent_max_latency_s": float(np.max(latencies)),
        "agent_total_eval_s": total_time,
        "agent_invalid_outputs": n_invalid, "agent_retries_used": n_retry,
        "agent_input_tokens": in_tok, "agent_output_tokens": out_tok,
        "errors_both_wrong": both, "errors_agent_only": agent_only, "errors_mlp_only": mlp_only,
        "client": f"{platform.system()} {platform.release()}, {os.cpu_count()} CPU threads, "
                  f"Python {platform.python_version()}",
    }
    with open("part2_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "=" * 78)
    print("PART 2: Zero-shot agent vs. best Part 1 classifier (same examples)")
    print("=" * 78)
    mlp_label = f"Part 1 MLP ({type(vec).__name__})"
    print(f"{'Metric':<32}{'gpt-oss agent (zero-shot)':<28}{mlp_label}")
    print(f"{'Sample size':<32}{len(sample_idx):<28}{len(sample_idx)}")
    print(f"{'Accuracy':<32}{agent_acc:<28.4f}{mlp_acc:.4f}")
    print(f"{'Avg time per example':<32}{f'{np.mean(latencies):.3f} s':<28}"
          f"{mlp_time / len(sample_idx) * 1000:.3f} ms")
    print(f"{'Total evaluation time':<32}{f'{total_time:.1f} s':<28}{mlp_time:.3f} s")
    print(f"{'Invalid outputs / retries used':<32}{f'{n_invalid} / {n_retry}':<28}n/a")
    print(f"{'Tokens (input / output)':<32}{f'{in_tok:,} / {out_tok:,}':<28}n/a")
    print(f"{'Labelled training examples':<32}{'0 (zero-shot)':<28}120,000")
    print(f"{'Runs on':<32}{'Nautilus GPUs (remote)':<28}local CPU")
    print(f"\nErrors: both wrong={both}, only agent wrong={agent_only}, only MLP wrong={mlp_only}")
    print("Saved part2_predictions.csv and part2_summary.json")

    print("\nSample agent errors with justifications:")
    for r in [r for r in rows if r["agent_pred"] != r["true"]][:5]:
        print(f"  true={r['true']:<9} agent={r['agent_pred']:<9} mlp={r['mlp_pred']:<9} "
              f"| {r['justification']}\n      {r['text'][:150]}")


def main():
    ap = argparse.ArgumentParser(description="Zero-shot AG News classification agent (gpt-oss)")
    ap.add_argument("--text", help="article text (title + description) to classify")
    ap.add_argument("--evaluate", action="store_true", help="run the Part 2 comparison")
    ap.add_argument("--limit", type=int, default=None,
                    help="only use the first N shared examples (for a quick test)")
    args = ap.parse_args()

    if args.evaluate:
        evaluate(args.limit)
        return
    article = args.text or input("Paste an AG News article (title + description): ")
    print("System prompt (zero-shot):\n" + SYSTEM_PROMPT + "\n")
    t = time.time()
    res = classify_article(article)
    print(f"Agent output ({time.time() - t:.2f}s, attempts={res['attempts']}):")
    if res["valid"]:
        print(json.dumps({"category": res["category"], "justification": res["justification"]}, indent=2))
    else:
        print(json.dumps({"error": res["error"]}, indent=2))


if __name__ == "__main__":
    main()

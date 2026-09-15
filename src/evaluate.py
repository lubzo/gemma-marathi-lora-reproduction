"""
evaluate.py — score a base model vs. its LoRA-fine-tuned version on a benchmark.
Runs on the same GPU notebook used for training. 
Adapt run_benchmark's field
names per dataset schema (sentiment/COPA/XNLI/ARC all differ).
"""
import argparse, csv, os
import torch
from datasets import load_dataset
from peft import PeftModel
from sklearn.metrics import f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer

BENCHMARK_DATASETS = {
    "indic_sentiment": ("ai4bharat/IndicSentiment", "mr"),
    "indic_copa": ("ai4bharat/IndicCOPA", "mr"),
    "indic_xnli": ("Divyanshu/indicxnli", "mr"),
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy"),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge"),
}


def load_model(model_id, adapter_dir=None):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer


def predict_label(model, tokenizer, prompt, label_options):
    best_label, best_score = None, float("-inf")
    for label in label_options:
        inputs = tokenizer(prompt + label, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        score = -outputs.loss.item()
        if score > best_score:
            best_score, best_label = score, label
    return best_label


def run_benchmark(model, tokenizer, benchmark_name):
    repo_id, config = BENCHMARK_DATASETS[benchmark_name]
    ds = load_dataset(repo_id, config, split="validation")
    preds, golds = [], []
    for ex in ds:
        prompt = ex.get("input") or ex.get("premise") or ex.get("question")
        label_options = ex.get("label_options") or ["0", "1"]
        preds.append(predict_label(model, tokenizer, prompt, label_options))
        golds.append(str(ex.get("label")))
    return f1_score(golds, preds, average="macro")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--adapter_dir", default=None)
    parser.add_argument("--benchmark", required=True, choices=BENCHMARK_DATASETS.keys())
    parser.add_argument("--output", default="results/automated_scores.csv")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_id, args.adapter_dir)
    f1 = run_benchmark(model, tokenizer, args.benchmark)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    file_exists = os.path.isfile(args.output)
    with open(args.output, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["model_id", "adapter_dir", "benchmark", "f1"])
        writer.writerow([args.model_id, args.adapter_dir or "base", args.benchmark, f1])
    print(f"{args.model_id} ({'fine-tuned' if args.adapter_dir else 'base'}) on {args.benchmark}: F1 = {f1:.4f}")


if __name__ == "__main__":
    main()
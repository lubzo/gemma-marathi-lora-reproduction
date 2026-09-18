"""
evaluate.py — Evaluate base Gemma models vs. fine-tuned LoRA adapters on South Asian
and Commonsense Reasoning benchmarks.

Reproduces the evaluation methodology from:
  "Challenges in Adapting Multilingual LLMs to Low-Resource Languages using LoRA PEFT Tuning"
  (Khade et al., CHiPSAL 2025, https://aclanthology.org/2025.chipsal-1.22.pdf)
  using established AI4Bharat benchmarks (Gala et al., 2024 / IndicInstruct).

Supported Benchmarks:
  1. indic_sentiment : ai4bharat/IndicSentiment (translation-mr)
  2. indic_copa      : ai4bharat/IndicCOPA (translation-mr)
  3. indic_xnli      : Divyanshu/indicxnli (mr)
  4. arc_easy        : allenai/ai2_arc (ARC-Easy)
  5. arc_challenge   : allenai/ai2_arc (ARC-Challenge)
  6. all             : Evaluates across all 5 benchmarks in a single run
"""

import argparse
import csv
import os
import random
import sys
import torch
from datasets import load_dataset
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer


BENCHMARKS = [
    "indic_sentiment",
    "indic_copa",
    "indic_xnli",
    "arc_easy",
    "arc_challenge",
]


def get_hardware_config():
    """Dynamically determine optimal precision and attention implementation."""
    if torch.cuda.is_available():
        use_bf16 = torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        attn_impl = "sdpa" if use_bf16 else "eager"
        device = "cuda"
    else:
        dtype = torch.float32
        attn_impl = "eager"
        device = "cpu"
    return device, dtype, attn_impl


def load_model(model_id, adapter_dir=None, dtype=None, attn_impl=None):
    """Load base model and optionally attach a LoRA adapter."""
    if dtype is None or attn_impl is None:
        _, dtype, attn_impl = get_hardware_config()

    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    print(f"Loading base model: {model_id} (dtype={dtype}, attn={attn_impl})")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
    )

    if adapter_dir:
        print(f"Attaching LoRA adapter from: {adapter_dir}")
        model = PeftModel.from_pretrained(model, adapter_dir)

    model.eval()
    return model, tokenizer


def apply_chat_template_if_needed(prompt, tokenizer, use_chat_format=False):
    """Format prompt with Gemma turn tokens if evaluating an instruction-tuned model."""
    if not use_chat_format:
        return prompt

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Append answer trigger token
        if chat_prompt[-1] in ["\n", " "]:
            chat_prompt += "The answer is: "
        else:
            chat_prompt += " The answer is: "
        return chat_prompt
    else:
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\nThe answer is: "


def get_candidate_token_ids(tokenizer, candidate_strings):
    """
    Safely extract the single-token ID for each choice candidate.
    Accounts for leading space tokenization across SentencePiece and BPE tokenizers.
    """
    candidate_ids = []
    for cand in candidate_strings:
        # Check tokenization with leading space first (standard next-token behavior)
        tok_with_space = tokenizer.encode(" " + cand.strip(), add_special_tokens=False)
        tok_raw = tokenizer.encode(cand.strip(), add_special_tokens=False)

        if len(tok_with_space) == 1:
            candidate_ids.append(tok_with_space[0])
        elif len(tok_raw) == 1:
            candidate_ids.append(tok_raw[0])
        else:
            # Fall back to the last token of the spaced string
            candidate_ids.append(tok_with_space[-1])
    return candidate_ids


@torch.no_grad()
def get_next_word_predictions(model, tokenizer, prompts, candidate_token_ids, batch_size=8):
    """
    Logit-based candidate scoring.
    Computes logits at the last position of each left-padded prompt and takes
    argmax over the candidate token probabilities.
    """
    predictions = []
    is_per_prompt = isinstance(candidate_token_ids[0], (list, tuple))

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        encoded = tokenizer(
            batch_prompts,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=2048,
        )
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)

        for j in range(len(batch_prompts)):
            c_ids = candidate_token_ids[i + j] if is_per_prompt else candidate_token_ids
            cand_probs = probs[j, c_ids]
            pred_idx = torch.argmax(cand_probs).item()
            predictions.append(pred_idx)

    return predictions


# ==============================================================================
# Benchmark 1: IndicSentiment (ai4bharat/IndicSentiment)
# ==============================================================================
def evaluate_indic_sentiment(model, tokenizer, ntrain=5, limit=None, batch_size=8, use_chat_format=False):
    print("Loading ai4bharat/IndicSentiment (translation-mr)...")
    try:
        ds = load_dataset(
            "json",
            data_files={
                "validation": "https://huggingface.co/datasets/ai4bharat/IndicSentiment/resolve/main/data/validation/mr.json",
                "test": "https://huggingface.co/datasets/ai4bharat/IndicSentiment/resolve/main/data/test/mr.json",
            },
        )
    except Exception:
        ds = load_dataset("ai4bharat/IndicSentiment", "translation-mr", trust_remote_code=True)
    dev_data = ds["validation"]
    test_data = ds["test"]

    if limit:
        test_data = test_data.select(range(min(limit, len(test_data))))

    choices = ["positive", "negative"]

    def parse_sentiment_label(lbl):
        if lbl is None:
            return 0
        s = str(lbl).strip().lower()
        if s in ["positive", "pos", "1"]:
            return 0
        elif s in ["negative", "neg", "0"]:
            return 1
        return 0

    def format_example(text, label=None):
        p = f"Review: {text.strip()}\nSentiment:"
        if label is not None:
            lbl_str = "positive" if parse_sentiment_label(label) == 0 else "negative"
            p += f" {lbl_str}\n\n"
        return p

    few_shot_prefix = "Predict the sentiment of the review. The possible choices for the sentiment are: 'positive' and 'negative'.\n\n"
    if ntrain > 0 and len(dev_data) > 0:
        for ex in dev_data.select(range(min(ntrain, len(dev_data)))):
            few_shot_prefix += format_example(ex.get("INDIC REVIEW", ""), ex.get("LABEL"))

    prompts, golds = [], []
    for ex in test_data:
        prompt_text = few_shot_prefix + format_example(ex.get("INDIC REVIEW", ""))
        prompts.append(apply_chat_template_if_needed(prompt_text, tokenizer, use_chat_format))
        golds.append(parse_sentiment_label(ex.get("LABEL")))

    candidate_ids = get_candidate_token_ids(tokenizer, choices)
    preds = get_next_word_predictions(model, tokenizer, prompts, candidate_ids, batch_size=batch_size)

    acc = accuracy_score(golds, preds)
    binary_f1 = f1_score(golds, preds)
    macro_f1 = f1_score(golds, preds, average="macro")
    return acc, binary_f1, macro_f1


# ==============================================================================
# Benchmark 2: IndicCOPA (ai4bharat/IndicCOPA)
# ==============================================================================
def evaluate_indic_copa(model, tokenizer, ntrain=5, limit=None, batch_size=8, use_chat_format=False):
    print("Loading ai4bharat/IndicCOPA (translation-mr)...")
    try:
        ds = load_dataset(
            "json",
            data_files={
                "test": "https://huggingface.co/datasets/ai4bharat/IndicXCOPA/resolve/main/data/test.mr.jsonl",
            },
        )
    except Exception:
        ds = load_dataset("ai4bharat/IndicCOPA", "translation-mr", trust_remote_code=True)
    # IndicCOPA only provides the 'test' split
    full_data = ds["test"]
    choices = ["A", "B"]

    def parse_copa_label(lbl):
        if lbl is None:
            return 0
        s = str(lbl).strip().upper()
        if s in ["0", "A"]:
            return 0
        elif s in ["1", "B"]:
            return 1
        try:
            return int(s)
        except ValueError:
            return 0

    def format_example(premise, choice1, choice2, question, label_idx=None):
        connector = {"cause": "because", "effect": "therefore"}.get(question, "therefore")
        p = f"{premise.strip()} {connector}\nA. {choice1.strip()}\nB. {choice2.strip()}\nAnswer:"
        if label_idx is not None:
            idx = parse_copa_label(label_idx)
            p += f" {choices[idx]}\n\n"
        return p

    # Use first few samples as exemplars if ntrain > 0, evaluate on remainder
    exemplars = [full_data[k] for k in range(min(ntrain, len(full_data)))] if ntrain > 0 else []
    eval_slice = full_data.select(range(len(exemplars), len(full_data)))

    if limit:
        eval_slice = eval_slice.select(range(min(limit, len(eval_slice))))

    few_shot_prefix = "I am hesitating between two options. Help me choose the more likely cause or effect.\n\n"
    for ex in exemplars:
        few_shot_prefix += format_example(ex["premise"], ex["choice1"], ex["choice2"], ex["question"], ex["label"])

    prompts, golds = [], []
    for ex in eval_slice:
        prompt_text = few_shot_prefix + format_example(ex["premise"], ex["choice1"], ex["choice2"], ex["question"])
        prompts.append(apply_chat_template_if_needed(prompt_text, tokenizer, use_chat_format))
        golds.append(parse_copa_label(ex["label"]))

    candidate_ids = get_candidate_token_ids(tokenizer, choices)
    preds = get_next_word_predictions(model, tokenizer, prompts, candidate_ids, batch_size=batch_size)

    acc = accuracy_score(golds, preds)
    binary_f1 = f1_score(golds, preds)
    macro_f1 = f1_score(golds, preds, average="macro")
    return acc, binary_f1, macro_f1


# ==============================================================================
# Benchmark 3: IndicXNLI (Divyanshu/indicxnli)
# ==============================================================================
def evaluate_indic_xnli(model, tokenizer, ntrain=5, limit=None, batch_size=8, use_chat_format=False):
    print("Loading Divyanshu/indicxnli (mr)...")
    try:
        test_ds = load_dataset(
            "json",
            data_files={"test": "https://huggingface.co/datasets/Divyanshu/indicxnli/resolve/main/forward/test/xnli_mr.json"},
            field="test",
        )["test"]
        dev_ds = load_dataset(
            "json",
            data_files={"validation": "https://huggingface.co/datasets/Divyanshu/indicxnli/resolve/main/forward/dev/xnli_mr.json"},
            field="validation",
        )["validation"]
    except Exception:
        ds = load_dataset("Divyanshu/indicxnli", "mr", trust_remote_code=True)
        dev_ds = ds["validation"]
        test_ds = ds["test"]
    dev_data = dev_ds
    test_data = test_ds

    if limit:
        test_data = test_data.select(range(min(limit, len(test_data))))

    choices = ["true", "unknown", "false"]

    def parse_xnli_label(lbl):
        if lbl is None:
            return 0
        s = str(lbl).strip().lower()
        label_map = {
            "0": 0, "entailment": 0, "true": 0,
            "1": 1, "neutral": 1, "unknown": 1,
            "2": 2, "contradiction": 2, "false": 2,
        }
        if s in label_map:
            return label_map[s]
        try:
            return int(s)
        except ValueError:
            return 0

    def format_example(premise, hypothesis, label_idx=None):
        p = f"Premise: {premise.strip()}\nHypothesis: {hypothesis.strip()}\nAnswer:"
        if label_idx is not None:
            idx = parse_xnli_label(label_idx)
            p += f" {choices[idx]}\n\n"
        return p

    few_shot_prefix = (
        "Answer whether the hypothesis is more likely to be true (entailment), "
        "false (contradiction), or unknown (neutral) based on the given premise.\n\n"
    )
    if ntrain > 0 and len(dev_data) > 0:
        for ex in dev_data.select(range(min(ntrain, len(dev_data)))):
            few_shot_prefix += format_example(ex["premise"], ex["hypothesis"], ex["label"])

    prompts, golds = [], []
    for ex in test_data:
        prompt_text = few_shot_prefix + format_example(ex["premise"], ex["hypothesis"])
        prompts.append(apply_chat_template_if_needed(prompt_text, tokenizer, use_chat_format))
        golds.append(parse_xnli_label(ex["label"]))

    candidate_ids = get_candidate_token_ids(tokenizer, choices)
    preds = get_next_word_predictions(model, tokenizer, prompts, candidate_ids, batch_size=batch_size)

    acc = accuracy_score(golds, preds)
    macro_f1 = f1_score(golds, preds, average="macro")
    return acc, None, macro_f1


# ==============================================================================
# Benchmark 4 & 5: AI2 ARC (allenai/ai2_arc: ARC-Easy & ARC-Challenge)
# ==============================================================================
def evaluate_arc(model, tokenizer, subset="ARC-Easy", ntrain=5, limit=None, batch_size=8, use_chat_format=False):
    print(f"Loading allenai/ai2_arc ({subset})...")
    ds = load_dataset("allenai/ai2_arc", subset)
    dev_data = ds["validation"]
    test_data = ds["test"]

    if limit:
        test_data = test_data.select(range(min(limit, len(test_data))))

    num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}

    def format_example(question, choices_text, choices_label, answer_key=None):
        p = f"{question.strip()}\n"
        std_labels = [num_to_letter.get(str(lbl), str(lbl)) for lbl in choices_label]
        for lbl, text in zip(std_labels, choices_text):
            p += f"{lbl}. {text.strip()}\n"
        p += "\nAnswer:"
        if answer_key is not None:
            ans_letter = num_to_letter.get(str(answer_key), str(answer_key))
            p += f" {ans_letter}\n\n"
        return p

    few_shot_prefix = "The following are multiple choice questions (with answers) about science.\n\n"
    if ntrain > 0 and len(dev_data) > 0:
        for ex in dev_data.select(range(min(ntrain, len(dev_data)))):
            few_shot_prefix += format_example(
                ex["question"], ex["choices"]["text"], ex["choices"]["label"], ex["answerKey"]
            )

    prompts, golds, per_prompt_candidates = [], [], []
    for ex in test_data:
        std_labels = [num_to_letter.get(str(lbl), str(lbl)) for lbl in ex["choices"]["label"]]
        ans_letter = num_to_letter.get(str(ex["answerKey"]), str(ex["answerKey"]))

        if ans_letter not in std_labels:
            continue

        prompt_text = few_shot_prefix + format_example(ex["question"], ex["choices"]["text"], ex["choices"]["label"])
        prompts.append(apply_chat_template_if_needed(prompt_text, tokenizer, use_chat_format))
        golds.append(std_labels.index(ans_letter))
        per_prompt_candidates.append(get_candidate_token_ids(tokenizer, std_labels))

    preds = get_next_word_predictions(model, tokenizer, prompts, per_prompt_candidates, batch_size=batch_size)

    acc = accuracy_score(golds, preds)
    macro_f1 = f1_score(golds, preds, average="macro")
    return acc, None, macro_f1


# ==============================================================================
# Benchmark Runner Dispatcher
# ==============================================================================
def run_single_benchmark(model, tokenizer, benchmark_name, ntrain=5, limit=None, batch_size=8, use_chat_format=False):
    if benchmark_name == "indic_sentiment":
        return evaluate_indic_sentiment(model, tokenizer, ntrain, limit, batch_size, use_chat_format)
    elif benchmark_name == "indic_copa":
        return evaluate_indic_copa(model, tokenizer, ntrain, limit, batch_size, use_chat_format)
    elif benchmark_name == "indic_xnli":
        return evaluate_indic_xnli(model, tokenizer, ntrain, limit, batch_size, use_chat_format)
    elif benchmark_name == "arc_easy":
        return evaluate_arc(model, tokenizer, "ARC-Easy", ntrain, limit, batch_size, use_chat_format)
    elif benchmark_name == "arc_challenge":
        return evaluate_arc(model, tokenizer, "ARC-Challenge", ntrain, limit, batch_size, use_chat_format)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")


def save_score_to_csv(csv_path, model_id, adapter_dir, benchmark, acc, binary_f1, macro_f1):
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    bin_f1_str = f"{binary_f1:.4f}" if binary_f1 is not None else "N/A"
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["model_id", "adapter_dir", "benchmark", "accuracy", "binary_f1", "macro_f1"])
        writer.writerow([model_id, adapter_dir or "base", benchmark, f"{acc:.4f}", bin_f1_str, f"{macro_f1:.4f}"])


def evaluate_suite(model, tokenizer, benchmarks, model_id, adapter_dir, ntrain, limit, batch_size, use_chat_format, output_csv):
    """Run a suite of benchmarks and print a summary table."""
    results = {}
    label_name = f"{model_id} + {adapter_dir}" if adapter_dir else f"{model_id} (base)"
    print(f"\n{'='*75}\nRunning evaluation for: {label_name}\n{'='*75}")

    for b_name in benchmarks:
        print(f"\n---> Evaluating on {b_name}...")
        acc, binary_f1, macro_f1 = run_single_benchmark(
            model,
            tokenizer,
            b_name,
            ntrain=ntrain,
            limit=limit,
            batch_size=batch_size,
            use_chat_format=use_chat_format,
        )
        results[b_name] = (acc, binary_f1, macro_f1)
        save_score_to_csv(output_csv, model_id, adapter_dir, b_name, acc, binary_f1, macro_f1)
        bin_str = f"{binary_f1:.4f}" if binary_f1 is not None else "N/A"
        print(f"[{b_name}] Accuracy: {acc:.4f} | Binary F1: {bin_str} | Macro F1: {macro_f1:.4f}")

    print(f"\n{'-'*75}")
    print(f"{'Benchmark':<20} | {'Accuracy':<10} | {'Binary F1 (Cls 1)':<18} | {'Macro F1':<10}")
    print(f"{'-'*75}")
    for b_name, (acc, bin_f1, mac_f1) in results.items():
        bin_str = f"{bin_f1:<18.4f}" if bin_f1 is not None else f"{'N/A':<18}"
        print(f"{b_name:<20} | {acc:<10.4f} | {bin_str} | {mac_f1:<10.4f}")
    print(f"{'-'*75}\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Base & LoRA Gemma models on Marathi/Commonsense benchmarks.")
    parser.add_argument("--model_id", required=True, help="Base model ID (e.g. google/gemma-2b).")
    parser.add_argument("--adapter_dir", default=None, help="LoRA adapter path (local folder or Hugging Face Hub ID).")
    parser.add_argument(
        "--benchmark",
        default="all",
        choices=BENCHMARKS + ["all"],
        help="Benchmark name or 'all' to evaluate on all 5.",
    )
    parser.add_argument(
        "--compare_base_and_adapter",
        action="store_true",
        help="Evaluate base model first, attach adapter in memory, and evaluate adapter.",
    )
    parser.add_argument("--ntrain", type=int, default=5, help="Number of few-shot exemplars (default: 5).")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test samples (for fast smoke test).")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Batch size for model inference.")
    parser.add_argument("--use_chat_format", action="store_true", help="Force chat template formatting.")
    parser.add_argument("--output", default="results/automated_scores_final.csv", help="CSV path to append results.")
    args = parser.parse_args()

    # Single-GPU isolation guard
    if "LOCAL_RANK" not in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    random.seed(42)
    torch.manual_seed(42)

    # Determine chat formatting
    is_chat_model = args.use_chat_format or ("-it" in args.model_id.lower())
    benchmarks_to_run = BENCHMARKS if args.benchmark == "all" else [args.benchmark]

    if args.compare_base_and_adapter:
        if not args.adapter_dir:
            print("Error: --compare_base_and_adapter requires --adapter_dir.")
            sys.exit(1)

        # 1. Evaluate Base Model
        base_model, tokenizer = load_model(args.model_id, adapter_dir=None)
        evaluate_suite(
            base_model,
            tokenizer,
            benchmarks_to_run,
            args.model_id,
            adapter_dir=None,
            ntrain=args.ntrain,
            limit=args.limit,
            batch_size=args.eval_batch_size,
            use_chat_format=is_chat_model,
            output_csv=args.output,
        )

        # 2. Attach adapter directly in VRAM and evaluate Fine-Tuned Model
        print(f"\nAttaching adapter {args.adapter_dir} to the base model in memory...")
        peft_model = PeftModel.from_pretrained(base_model, args.adapter_dir)
        peft_model.eval()
        evaluate_suite(
            peft_model,
            tokenizer,
            benchmarks_to_run,
            args.model_id,
            adapter_dir=args.adapter_dir,
            ntrain=args.ntrain,
            limit=args.limit,
            batch_size=args.eval_batch_size,
            use_chat_format=is_chat_model,
            output_csv=args.output,
        )
    else:
        # Standard evaluation (Base alone OR Adapter alone)
        model, tokenizer = load_model(args.model_id, adapter_dir=args.adapter_dir)
        evaluate_suite(
            model,
            tokenizer,
            benchmarks_to_run,
            args.model_id,
            adapter_dir=args.adapter_dir,
            ntrain=args.ntrain,
            limit=args.limit,
            batch_size=args.eval_batch_size,
            use_chat_format=is_chat_model,
            output_csv=args.output,
        )

    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
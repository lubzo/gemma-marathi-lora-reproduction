"""
translate.py — My own translated Marathi Alpaca dataset.
"""
import argparse, json
import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "facebook/nllb-200-distilled-600M"


def build_translator():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

    def translate(text):
        if not text:
            return ""
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        out = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids("mar_Deva"), max_new_tokens=256, max_length=None)
        return tokenizer.decode(out[0], skip_special_tokens=True)

    return translate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume_from", type=int, default=0)
    args = parser.parse_args()

    ds = load_dataset("unsloth/alpaca-cleaned")["train"]
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    translate = build_translator()
    mode = "a" if args.resume_from > 0 else "w"
    with open(args.output, mode, encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            if i < args.resume_from:
                continue
            row = {"instruction": translate(ex["instruction"]), "input": translate(ex["input"]), "output": translate(ex["output"])}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 50 == 0:
                print(f"{i}/{len(ds)} rows translated")


if __name__ == "__main__":
    main()
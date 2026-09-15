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
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    def translate(texts, max_new_tokens=256):
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        if not any(texts):
            return "" if single else ["" for _ in texts]

        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids("mar_Deva"),
                max_new_tokens=max_new_tokens,
                max_length=None,
            )
        results = tokenizer.batch_decode(out, skip_special_tokens=True)
        results = [res if orig else "" for orig, res in zip(texts, results)]
        return results[0] if single else results

    return translate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume_from", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    ds = load_dataset("unsloth/alpaca-cleaned")["train"]
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    translate = build_translator()
    mode = "a" if args.resume_from > 0 else "w"
    with open(args.output, mode, encoding="utf-8") as f:
        translated_rows = 0
        total_len = len(ds)

        for i in range(args.resume_from, total_len, args.batch_size):
            batch = ds[i : i + args.batch_size]
            try:
                trans_instructions = translate(batch["instruction"], max_new_tokens=256)
                trans_inputs = translate(batch["input"], max_new_tokens=256)
                trans_outputs = translate(batch["output"], max_new_tokens=512)

                for inst, inp, out in zip(trans_instructions, trans_inputs, trans_outputs):
                    row = {
                        "instruction": inst,
                        "input": inp,
                        "output": out,
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()

                batch_count = len(batch["instruction"])
                translated_rows += batch_count
                current_progress = min(i + args.batch_size, total_len)
                if (i // args.batch_size) % 5 == 0 or current_progress == total_len:
                    print(f"{current_progress}/{total_len} rows translated")
            except Exception as error:
                print(f"Stopped at row {i}: {error}")
                print(f"Resume with --resume_from {i}")
                break
    print(f"Done. Translated {translated_rows} rows.")

if __name__ == "__main__":
    main()
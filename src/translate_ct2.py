"""
translate_ct2.py — High-throughput Marathi Alpaca translation using CTranslate2.
"""
import argparse
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import ctranslate2

TOKENIZER_NAME = "facebook/nllb-200-distilled-600M"


def build_translator(model_dir: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, src_lang="eng_Latn")
    translator = ctranslate2.Translator(model_dir, device=device, compute_type=compute_type)

    def translate(texts, max_decoding_length=256):
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        if not any(texts):
            return "" if single else ["" for _ in texts]

        # Tokenize source texts
        source_tokens = [
            tokenizer.convert_ids_to_tokens(tokenizer.encode(t, truncation=True, max_length=512))
            if t else []
            for t in texts
        ]

        # Prepare target prefix token for Marathi
        target_prefix = [["mar_Deva"] if toks else [] for toks in source_tokens]

        # Translate in parallel with CTranslate2 C++ engine
        results = translator.translate_batch(
            source_tokens,
            target_prefix=target_prefix,
            max_decoding_length=max_decoding_length,
            beam_size=1,  # Greedy search: maximum speed and high quality
        )

        translations = []
        for orig, res in zip(texts, results):
            if not orig or not res.hypotheses:
                translations.append("")
                continue
            # Hypothesis starts with target prefix token ("mar_Deva"), slice it off: [1:]
            hyp_tokens = res.hypotheses[0][1:]
            decoded = tokenizer.decode(tokenizer.convert_tokens_to_ids(hyp_tokens), skip_special_tokens=True)
            translations.append(decoded)

        return translations[0] if single else translations

    return translate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_dir", default="nllb-600m-ct2-fp16")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume_from", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    ds = load_dataset("unsloth/alpaca-cleaned")["train"]
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    print(f"Loading CTranslate2 model from {args.model_dir}...")
    translate = build_translator(args.model_dir)

    mode = "a" if args.resume_from > 0 else "w"
    with open(args.output, mode, encoding="utf-8") as f:
        translated_rows = 0
        total_len = len(ds)

        for i in range(args.resume_from, total_len, args.batch_size):
            batch = ds[i : i + args.batch_size]
            try:
                trans_instructions = translate(batch["instruction"], max_decoding_length=256)
                trans_inputs = translate(batch["input"], max_decoding_length=256)
                trans_outputs = translate(batch["output"], max_decoding_length=512)

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

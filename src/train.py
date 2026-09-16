"""
train.py — LoRA fine-tuning for one Gemma variant on Marathi Alpaca data.
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset



def format_example(ex):
    return {"text": f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{ex.get('input','')}\n\n### Response:\n{ex['output']}"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--dataset", default="lubzo/marathi-alpaca-cleaned-translated")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of training samples for testing")
    args = parser.parse_args()

    # Configure the model for 4-bit quantization using BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, quantization_config=bnb_config, device_map="auto")

    lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                              target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")

    ds = load_dataset(args.dataset)["train"]
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    ds = ds.map(format_example)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        dataset_text_field="text",
        max_length=args.max_seq_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=25,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Done. Adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
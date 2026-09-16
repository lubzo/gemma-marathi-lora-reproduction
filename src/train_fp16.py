"""
train_fp16.py — FP16 + LoRA fine-tuning for Gemma on 2x T4 GPUs.
No 4-bit quantization, no dequantization latency, native Tensor Core acceleration.
"""
import argparse, os

# If running single-process on multi-GPU, isolate to GPU 0 to prevent DataParallel conflicts
if "LOCAL_RANK" not in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of training samples for testing")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_map = {"": local_rank} if torch.cuda.is_available() else None

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Load base model directly in native FP16 (takes ~5 GB VRAM on each 16 GB T4)
    print(f"[Rank {local_rank}] Loading base model in native FP16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.float16,
        device_map=device_map,
        attn_implementation="eager",
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

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
        fp16=True,
        bf16=False,
        optim="adamw_torch",
        ddp_find_unused_parameters=False,
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

    print(f"[Rank {local_rank}] Starting training...")
    trainer.train()

    if local_rank == 0:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Done. Adapter saved to {args.output_dir}")

    # Clean up distributed process group if running under torchrun/DDP
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

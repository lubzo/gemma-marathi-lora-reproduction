"""
push_to_hub.py — upload the trained LoRA adapter folder to HuggingFace.
Run this on your local computer, after downloading the trained adapter folder from Colab
"""
import argparse, shutil
from huggingface_hub import HfApi, create_repo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--model_card", default=None)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    if args.model_card:
        shutil.copy(args.model_card, f"{args.local_dir}/README.md")

    print(f"Uploading {args.local_dir} -> https://huggingface.co/{args.repo_id} ...")
    api.upload_folder(folder_path=args.local_dir, repo_id=args.repo_id, repo_type="model")
    print(f"Done: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
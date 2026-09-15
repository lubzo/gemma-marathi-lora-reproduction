# Gemma + LoRA PEFT for Marathi — Reproduction

Reproduction of ["Challenges in Adapting Multilingual LLMs to Low-Resource Languages using LoRA PEFT Tuning"](https://aclanthology.org/2025.chipsal-1.22.pdf) (Khade et al., CHiPSAL 2025), which LoRA-fine-tunes Gemma models on a Marathi-translated Alpaca-52k dataset and finds that automated benchmarks and human judgment often disagree about whether fine-tuning helped.

This repo contains the fine-tuning code, evaluation scripts, and results for reproducing that study on `gemma-2b`, `gemma-2-2b`, and `gemma-2-2b-it`.

## Results

| Model | IndicSentiment F1 | ARC-Easy F1 | ARC-Challenge F1 | IndicCOPA F1 | IndicXNLI F1 |
|---|---|---|---|---|---|
| gemma-2b (base) | — | — | — | — | — |
| gemma-2b (Mr, LoRA) | — | — | — | — | — |
| gemma-2b-it (base) | — | — | — | — | — |
| gemma-2-2b (base) | — | — | — | — | — |
| gemma-2-2b (Mr, LoRA) | — | — | — | — | — |
| gemma-2-2b-it (base) | — | — | — | — | — |
| gemma-2-2b-it (Mr, LoRA) | — | — | — | — | — |

*see `results/automated_scores.csv`*

**Manual evaluation (150 questions, win rate):** see `results/manual_eval.csv`.

## Links

- 🤗 Fine-tuned models: `https://huggingface.co/lubzo/gemma-2b-marathi-lora` (and the other 3 variants)
- 🤗 Dataset used: `https://huggingface.co/datasets/lubzo/marathi-alpaca-translated`
- 🤗 Live demo Space: `https://huggingface.co/spaces/lubzo/gemma-marathi-demo`
- 📄 Original paper: `https://aclanthology.org/2025.chipsal-1.22.pdf`

## Repo structure

```
src/
  train.py                        # standalone training script (mirrors the notebooks)
  evaluate.py                     # benchmark scoring
  translate.py                    # optional: build your own translated dataset
results/
  automated_scores.csv
  manual_eval.csv
```

## What I found vs. the paper


## Limitations


## Citation

```bibtex
@inproceedings{khade2025challenges,
  title={Challenges in Adapting Multilingual LLMs to Low-Resource Languages using LoRA PEFT Tuning},
  author={Khade, Omkar and Jagdale, Shruti and Phaltankar, Abhishek and Takalikar, Gauri and Joshi, Raviraj},
  booktitle={Proceedings of the First Workshop on Challenges in Processing South Asian Languages (CHiPSAL 2025)},
  pages={217--222},
  year={2025}
}
```

## License

MIT for the code in this repo (see `LICENSE`). The dataset and base models retain their original licenses (Alpaca's non-commercial research license, Gemma's terms of use, Google Translate ToS if you re-translate).
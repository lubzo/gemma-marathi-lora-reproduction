# Gemma + LoRA PEFT for Marathi — Reproduction & Re-evaluation

A reproduction and re-evaluation of [*"Challenges in Adapting Multilingual LLMs to Low-Resource Languages using LoRA PEFT Tuning"*](https://aclanthology.org/2025.chipsal-1.22.pdf) (Khade et al., CHiPSAL 2025).

---

## 🎯 Motivation: Why Reproduce with a Cleaned Dataset?

The original study fine-tuned Gemma models on a Marathi-translated version of the original **Stanford Alpaca 52k** dataset. However, the original Stanford Alpaca dataset contains documented quality issues:
* **Hallucinations & Factually Contradictory Answers**: Fabricated citations, incorrect factual knowledge, and flawed maths.
* **Malformed Code & Syntax Errors**: Truncated code blocks and invalid syntax that confuse language model completion.
* **Empty or Degenerate Outputs**: Hundreds of samples where the teacher model produced blank responses or repetitive phrases.
* **Low-Quality Prompts**: Ambiguous, repetitive, or ungrounded instructions.

When noisy data is translated into a low-resource language like Marathi, errors compound: hallucinations translate into ungrammatical Marathi phrasing, and syntax errors produce malformed token structures.

### My Research Question
> **Does fine-tuning on a cleaned dataset ([`unsloth/alpaca-cleaned`](https://huggingface.co/datasets/unsloth/alpaca-cleaned)) translated to Marathi prevent capability degradation on reasoning benchmarks or improve performance compared to the original paper?**

Additionally, because the original paper reported single "F1" values without clarifying whether they used **Binary F1** (class 1) or **Macro F1** (unweighted class average), I report **Accuracy**, **Binary F1 (Cls 1)**, and **Macro F1** across all benchmarks to ensure complete transparency.

---

## 📊 Results Summary

The benchmark suite consists of **3 Marathi tasks** (`IndicSentiment`, `IndicCOPA`, `IndicXNLI`) and **2 English science reasoning tasks** (`ARC-Easy`, `ARC-Challenge`) to evaluate both Marathi acquisition and English capability retention ("catastrophic forgetting").

### High-Level Macro F1 Comparison

| Model | IndicSentiment | ARC-Easy | ARC-Challenge | IndicCOPA | IndicXNLI |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **gemma-2b (base)** | 0.9074 | 0.4902 | 0.4201 | 0.4792 | 0.1869 |
| **gemma-2b (Mr, LoRA)** | 0.8441 | 0.2076 | 0.1339 | 0.3594 | 0.2906 |
| **gemma-2b-it (base)** | 0.3556 | 0.1893 | 0.2507 | 0.3343 | 0.1696 |
| **gemma-2b-it (Mr, LoRA)** | 0.8710 | 0.0895 | 0.2052 | 0.4269 | 0.1919 |
| **gemma-2-2b (base)** | 0.9172 | 0.6463 | 0.5247 | 0.5671 | 0.3075 |
| **gemma-2-2b (Mr, LoRA)** | 0.9519 | 0.7924 | 0.3043 | 0.3846 | 0.3551 |
| **gemma-2-2b-it (base)** | 0.9560 | 0.8799 | 0.5404 | 0.3443 | 0.2977 |
| **gemma-2-2b-it (Mr, LoRA)** | 0.9660 | 0.6135 | 0.1782 | 0.3343 | 0.2727 |

*Note: Table reports Macro F1 across all benchmarks.*

---

## 🔬 Detailed Benchmark Breakdown (Accuracy, Binary F1, Macro F1 vs. Paper)

Because prior work did not document which F1 formulation was reported, the tables below report all three metrics side-by-side with the paper's published numbers.

### 1. Gemma-2-2B (`google/gemma-2-2b`)

| Model | Metric | IndicSentiment | ARC-Easy | ARC-Challenge | IndicCOPA | IndicXNLI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **google/gemma-2-2b (Base)** | Accuracy | 0.9180 | 0.8085 | 0.6570 | 0.5698 | 0.3782 |
| | Binary F1 (Cls 1) | 0.9093 | N/A | N/A | 0.6013 | N/A |
| | Macro F1 | 0.9172 | 0.6463 | 0.5247 | 0.5671 | 0.3075 |
| | *Paper Reported F1* | *0.9206* | *0.6384* | *0.6463* | *0.6577* | *0.2191* |
| **gemma-2-2b-marathi-lora** | Accuracy | 0.9520 | 0.7424 | 0.4300 | 0.5158 | 0.4439 |
| | Binary F1 (Cls 1) | 0.9502 | N/A | N/A | 0.6687 | N/A |
| | Macro F1 | 0.9519 | 0.7924 | 0.3043 | 0.3846 | 0.3551 |
| | *Paper Reported F1* | *0.8411* | *0.6135* | *0.5271* | *0.5764* | *0.2753* |

### 2. Gemma-2-2B-IT (`google/gemma-2-2b-it`)

| Model | Metric | IndicSentiment | ARC-Easy | ARC-Challenge | IndicCOPA | IndicXNLI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **google/gemma-2-2b-it (Base)** | Accuracy | 0.9560 | 0.8497 | 0.5410 | 0.5068 | 0.3956 |
| | Binary F1 (Cls 1) | 0.9560 | N/A | N/A | 0.0179 | N/A |
| | Macro F1 | 0.9560 | 0.8799 | 0.5404 | 0.3443 | 0.2977 |
| | *Paper Reported F1* | *0.9749* | *0.6851* | *0.7210* | *0.7210* | *0.2814* |
| **gemma-2-2b-it-marathi-lora** | Accuracy | 0.9660 | 0.6237 | 0.2961 | 0.5023 | 0.3597 |
| | Binary F1 (Cls 1) | 0.9657 | N/A | N/A | 0.0000 | N/A |
| | Macro F1 | 0.9660 | 0.6135 | 0.1782 | 0.3343 | 0.2727 |
| | *Paper Reported F1* | *0.9589* | *0.6343* | *0.6374* | *0.5835* | *0.1667* |

### 3. Gemma-2B (`google/gemma-2b`)

| Model | Metric | IndicSentiment | ARC-Easy | ARC-Challenge | IndicCOPA | IndicXNLI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **gemma-2b (Base)** | Accuracy | 0.9080 | 0.6216 | 0.4377 | 0.4932 | 0.3323 |
| | Binary F1 (Cls 1) | 0.9000 | N/A | N/A | 0.5648 | N/A |
| | Macro F1 | 0.9074 | 0.4902 | 0.4201 | 0.4792 | 0.1869 |
| | *Paper Reported F1* | *0.7772* | *0.4435* | *0.4240* | *0.6547* | *0.3582* |
| **gemma-2b-marathi-lora** | Accuracy | 0.8480 | 0.3519 | 0.2756 | 0.5000 | 0.3451 |
| | Binary F1 (Cls 1) | 0.8195 | N/A | N/A | 0.6595 | N/A |
| | Macro F1 | 0.8441 | 0.2076 | 0.1339 | 0.3594 | 0.2906 |
| | *Paper Reported F1* | *0.9397* | *0.6048* | *0.3848* | *0.4219* | *0.1675* |

### 4. Gemma-2B-IT (`google/gemma-2b-it`)

| Model | Metric | IndicSentiment | ARC-Easy | ARC-Challenge | IndicCOPA | IndicXNLI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **gemma-2b-it (Base)** | Accuracy | 0.5010 | 0.3215 | 0.2927 | 0.5023 | 0.3329 |
| | Binary F1 (Cls 1) | 0.6617 | N/A | N/A | 0.0000 | N/A |
| | Macro F1 | 0.3556 | 0.1893 | 0.2507 | 0.3343 | 0.1696 |
| | *Paper Reported F1* | *0.7444* | *0.4651* | *0.4043* | *0.2963* | *0.3066* |
| **gemma-2b-it-marathi-lora** | Accuracy | 0.8710 | 0.2500 | 0.2765 | 0.5023 | 0.3335 |
| | Binary F1 (Cls 1) | 0.8724 | N/A | N/A | 0.2191 | N/A |
| | Macro F1 | 0.8710 | 0.0895 | 0.2052 | 0.4269 | 0.1919 |
| | *Paper Reported F1* | *—* | *—* | *—* | *—* | *—* |

---

## 🔍 What I Found vs. the Paper

1. **Directional Trajectory on Gemma-2-2B Marathi Sentiment**:
   * **In the paper**: The authors observed an overall **performance drop** in their reported sentiment F1 metric when adapting Gemma-2-2B with uncleaned Alpaca dataset.
   * **In my reproduction**: With Cleaned Alpaca, I observed **consistent performance gains** across all evaluated metrics: Accuracy gained from `0.9180` to `0.9520`, Macro F1 gained from `0.9172` to `0.9519`, and Binary F1 gained from `0.9093` to `0.9502`.
   * *Metric comparability note*: Because the original paper did not specify which F1 variant was reported, I do not claim point-for-point decimal parity. Instead, the critical takeaway is that the sentiment degradation reported in the original paper did not occur when training on cleaned data.
2. **Forgetting on English Reasoning Persists**:
   * Both studies observe a sharp **capability loss** on English science reasoning (ARC-Easy and ARC-Challenge) after Marathi LoRA adaptation. In my runs, ARC-Challenge dropped significantly on both Gemma-2-2B (Accuracy: `0.6570` $\rightarrow$ `0.4300`, Macro F1: `0.5247` $\rightarrow$ `0.3043`) and Gemma-2-2B-IT (Accuracy: `0.5410` $\rightarrow$ `0.2961`, Macro F1: `0.5404` $\rightarrow$ `0.1782`).
   * **Takeaway:** While training on cleaned data improved Marathi sentiment on Gemma-2-2B and yielded solid IndicXNLI scores, it did not prevent the decline on English science reasoning (ARC). This indicates that dataset cleaning alone is insufficient to preserve source-language reasoning capabilities, at least under my fixed LoRA configuration. 
3. **IndicXNLI Retention Relative to Paper**:
   * On IndicXNLI, fine-tuning on Cleaned Alpaca yielded scores that remained consistently higher than the paper's reported numbers across models:
     * `gemma-2-2b`: Base (`0.3782` Acc, `0.3075` Macro F1) $\rightarrow$ Marathi LoRA (`0.4439` Acc, `0.3551` Macro F1, vs. paper's `0.2753`).
     * `gemma-2b`: Base (`0.3323` Acc, `0.1869` Macro F1) $\rightarrow$ Marathi LoRA (`0.3451` Acc, `0.2906` Macro F1, vs. paper's `0.1675`).
     * `gemma-2-2b-it`: Base (`0.3956` Acc, `0.2977` Macro F1) $\rightarrow$ Marathi LoRA (`0.3597` Acc, `0.2727` Macro F1, vs. paper's `0.1667`).
   * Cleaned training data generally preserved higher cross-lingual inference performance relative to the original paper's reported results.
4. **IndicCOPA Results (Causal Reasoning)**:
   * Across all models, accuracy remained near **50%** (ranging between 49.3% and 57.0%), which is close to the 50% random-choice baseline for a two-choice task.
   * Fine-tuning did not produce any significant improvements on this benchmark.
   * *(Note: The original paper reported only F1 scores between 0.29 and 0.72, and did not publish accuracy).*

---

## ⚠️ Limitations

1. **Underspecified Experimental Hyperparameters in Original Paper**:
   * The original paper did not report critical training hyperparameters, including LoRA rank $r$, scaling factor $\alpha$, target modules, dropout, learning rate, or epoch counts. Based on common PEFT defaults for instruction tuning, I assumed:
     * LoRA rank $r = 16$
     * Scaling factor $\alpha = 32$
     * LoRA dropout $= 0.05$
     * Target modules: Attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
     * Learning rate $= 2\times 10^{-4}$
     * Epoch count $= 3$
   * Any divergence between my hyperparameter assumptions and whatever unstated setup the original authors ran could contribute to numerical variance.
2. **Translation Engine & Cost Constraints (NLLB vs. Google Translate)**:
   * The paper authors translated 52k samples using the proprietary Google Translate API. Due to budget and cost implications of cloud translation APIs at scale, I translated the cleaned Alpaca dataset using open-source **Meta NLLB-200** via `CTranslate2` (`nllb-200-distilled-600M` / `1.3B`). While NLLB produces high-quality Marathi translations, subtle syntactic and vocabulary differences between translation engines can influence fine-tuning dynamics.
3. **Evaluation Metric Ambiguity**:
   * The original paper only published a single "F1" column without defining whether it represented Binary F1 (class 1) or Macro F1. In AI4Bharat's evaluation harness, binary tasks default to binary F1, whereas multiclass tasks default to macro F1. By reporting Accuracy, Binary F1, and Macro F1, my results demonstrate how metric selection drastically affects perceived model success.
4. **Hardware & Quantization Constraints**:
   * Fine-tuning was conducted on single cloud GPU instances (RTX A4500/A5000) using 16-bit precision (`bfloat16`/`float16`) with sequence lengths capped at 512 tokens.

---

## 🔗 Links

- 🤗 Fine-tuned models: `https://huggingface.co/lubzo` (and variant adapters)
- 🤗 Dataset used: `https://huggingface.co/datasets/lubzo/marathi-alpaca-cleaned-translated`
- 🤗 Live demo Space: `https://huggingface.co/spaces/lubzo/gemma-marathi-demo`
- 📄 Original paper: [CHiPSAL 2025 Paper](https://aclanthology.org/2025.chipsal-1.22.pdf)

---

## 📁 Repo Structure

```
src/
  train.py                        # Standalone training script with LoRA PEFT
  evaluate.py                     # 5-benchmark evaluation suite (Logit scoring + batched)
  translate_ct2.py                # High-speed NLLB CTranslate2 translation pipeline
results/
  automated_scores.csv            # Raw log of all benchmark runs across models
```

---

## 📜 Citation

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
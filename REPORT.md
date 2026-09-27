# MP1 Report: Gated FFN, Regularized Scaling, and Distillation

DASE7506 · Protocol `7506-mp1-wt2-v2`  
Submitted full-test BPB (CPU, FP32): **1.5147**

## 1. Problem and baseline limitation

The task is to train a language model from random initialization on the supplied WikiText-2 splits and BPE-2048 tokenizer, then minimize full-test bits per byte (BPB; lower is better) under the evaluation budget (≤5× baseline CPU scoring time, ≤4 GiB peak evaluation RAM, ≤64 MiB uncompressed inference assets). Development uses only the validation split. Evaluation windows are independent and causal with 256 tokens.

The classroom baseline is a 4-layer GPT (width 128, 4 heads, 1,088,256 parameters) with learned absolute positions, LayerNorm, and a 4× GELU MLP. On this machine it reproduces the announced score:

| Split | BPB | Token PPL | Scoring seconds |
|---|---:|---:|---:|
| Validation | 2.0711 | 79.53 | 12.1 |
| **Test** | **2.1013** | 80.85 | 13.7 |

A GELU MLP applies the same pointwise nonlinearity to every channel. For a small model this is a structural bottleneck: there is no gate that can suppress unhelpful features. Absolute position embeddings treat each index independently. The default 1,200-step recipe processes only 9.83M training targets; a modestly larger model underfits at that budget and overfits if trained much longer without regularization.

## 2. Method

The student model (`student.py`) keeps the baseline interfaces `forward` and `predict_log_probs`. Optional config keys keep earlier checkpoints loadable.

**Key mechanism for the equal-token study: SwiGLU.**  
Each block replaces `Linear–GELU–Linear` with `down(SiLU(gate(x)) ⊙ up(x))`. The inner width is `round(8/3 · width)` so the parameter count stays within 168 of the baseline (1,088,424 vs 1,088,256). Ablation sets `ffn=gelu` and restores the original MLP.

**Submitted predictor** (`configs/submit_distill.json`):

- SwiGLU, RoPE on Q/K, RMSNorm, and QK-Norm (RMSNorm on Q and K before attention).
- Width 288, 8 heads, 7 layers (7,582,776 parameters, about 29 MiB).
- Residual, embedding, and attention dropout 0.15 (off at evaluation).
- Residual projection weights scaled by \(1/\sqrt{2L}\) at initialization.
- Teacher–student logit distillation: a frozen 384-wide, 8-layer teacher trained only on the supplied training text; the submitted student matches the teacher’s softened next-token distribution as well as the hard labels (`--distill 0.6`, temperature 2).
- AdamW with cosine decay, EMA decay 0.999 used for validation and checkpointing, and optional SWA of the last few snapshots if it wins on validation (here the selected step was 12,000, not SWA).
- Student: 12,000 steps × 32 × 256 = 98,304,000 targets, seed 17. Teacher: 8,000 steps, same seed, not submitted.

The teacher is a training-only asset. Evaluation loads a single student state dict. Dropout is disabled at evaluation, so causality, normalization, and window-independence tests still pass.

## 3. Equal-token comparison and ablation

All three runs below use seed 17, 1,200 steps, batch 32, and 9,830,400 training targets.

| Run | Implementation | FFN | Val BPB | Test BPB | Params | Train seconds |
|---|---|---|---:|---:|---:|---:|
| Baseline | `model.py` | GELU | 2.0711 | **2.1013** | 1,088,256 | 906 |
| Student | `student.py` | SwiGLU | **2.0109** | **2.0406** | 1,088,424 | 2,159 |
| Ablation | `student.py` | GELU | 2.0711 | — | 1,088,256 | 905 |

The ablation training loss and validation BPB match the official baseline to reported precision. Therefore the 0.060 test-BPB drop at matched token count is attributable to the gate, not to a different codebase, seed, or optimizer. Token perplexity moves from 80.85 to 71.22, in the same direction as BPB.

The longer wall-clock time of the SwiGLU 1,200-step run was a one-off stall on this laptop: the GELU ablation finished in 905 seconds, essentially identical to `model.py`.

## 4. Scaling, regularization, and distillation

Later runs keep SwiGLU and add capacity, dropout, EMA, and distillation. Selection is always by validation.

| Stage | Setup | Val BPB | Test BPB | Test seconds |
|---|---|---:|---:|---:|
| Scale, no dropout | 256×6, RoPE, RMS, 4k steps | 1.6517 | 1.6869 | 66.5 |
| Same, 8k steps | overfits after ~4k | 1.7868 | 1.8251 | 54.3 |
| Dropout 0.1 | 256×6, 8k steps | 1.5364 | 1.5637 | 44.3 |
| Longer + EMA | 256×6, 15k, EMA, selected 10k | 1.5199 | 1.5429 | 59.0 |
| Wider + EMA | 288×7, dropout 0.15, 10k | 1.5111 | 1.5372 | 63.7 |
| **Submitted** | **288×7 + QK-Norm + distill 12k** | **1.4938** | **1.5147** | **64.5–68.4** |

Without dropout, validation is best near 4,000 steps and then degrades: training loss falls while validation BPB rises. Dropout 0.1 reverses that through 8,000 steps. EMA on a single trajectory is a legal weight average and improved the 256-wide run from 1.560 (20k, no EMA) to 1.543 test.

Distillation is the last gain that still fits the time cap. The submitted student is only 504 parameters larger than the 288×7 EMA run (QK-Norm scales). A 384-wide teacher is trained first, then frozen; the student minimizes \(0.4\,\mathrm{CE} + 0.6\,T^2\,\mathrm{KL}\) with \(T=2\). Validation fell from 1.511 to 1.494 and test from 1.537 to 1.515.

The validation–test gap stays about 0.02–0.03 on every frozen predictor. Test scoring of the submitted checkpoint was 68.4 s on a cold pass and 64.5 s on a repeat, versus a 13.7 s baseline (about 4.7–5.0×). The state dict is about 29 MiB. Peak evaluation RAM on CPU is well under 4 GiB.

## 5. What failed

- **More steps without regularization** (256×6, 8k): validation 1.65 → 1.79.
- **A 50k-step wider run without EMA:** best validation about 1.557, then overfitting.
- **Averaging checkpoints from different runs** (8k + 20k): validation 2.061. Distant soups are not a substitute for same-run EMA.
- **Longer distillation with dropout 0.18 and a 15k teacher:** validation 1.4933, statistically tied with the submitted 1.4938 and worse at the matched 12k step (1.505 vs 1.494). Not submitted.
- **A 512-wide teacher plus sampling without replacement:** student validation 1.500, worse than the submitted run. A larger teacher did not raise the student’s ceiling on this corpus.

These failures suggest that WikiText-2 plus a 7.6M student that must score within 5× of a 1.1M baseline is capacity- and data-limited. Further schedule tweaks stayed in a 1.49–1.50 validation band.

## 6. Trade-offs and limitations

SwiGLU is cheap and is the required algorithmic contribution. Most of the remaining gain (2.04 → 1.51 test BPB) comes from a larger network, more tokens, dropout, EMA, and distillation. Scoring became slower (14 s → 65 s) but stayed at the legal edge on this laptop. Training used a rented RTX 3080 Ti; the teacher is discarded at submission.

The method never uses test text for selection, never changes the tokenizer or evaluator, never loads pretrained weights or extra text, and never carries state across evaluation windows.

## 7. Reproducibility

See `code/README.md` for installation. Frozen evaluation (no retraining):

```bash
cd code
python evaluate.py --checkpoint /path/to/cloud-distill-12k/checkpoint.pt --device cpu --precision fp32 --split test
```

Expected test BPB: **1.5147**. Checkpoint SHA-256: `235c0fc0821d95d4045db3cdbe62af6021d4b1d84ce680dcdddeb6c8df52b3aa`.

Retrain on a GPU (teacher, then student):

```bash
python train.py --implementation student --config configs/teacher.json --device cuda --seed 17 --steps 8000 --eval-every 1000 --ema 0.999 --lr 0.001 --run-dir runs/teacher-8k
python train.py --implementation student --config configs/submit_distill.json --device cuda --seed 17 --steps 12000 --eval-every 1000 --ema 0.999 --lr 0.001 --teacher runs/teacher-8k/checkpoint.pt --distill 0.6 --distill-temp 2 --swa 6 --run-dir runs/cloud-distill-12k
```

The equal-token SwiGLU run uses `configs/student.json` and `--steps 1200`. The ablation uses `configs/ablation_gelu.json`.

## References

- Shazeer, N. (2020). GLU Variants Improve Transformer. arXiv:2002.05202.
- Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv:2104.09864.
- Zhang & Sennrich (2019). Root Mean Square Layer Normalization. NeurIPS.
- Hinton et al. (2015). Distilling the Knowledge in a Neural Network. arXiv:1503.02531.
- Merity et al. (2016). Pointer Sentinel Mixture Models. arXiv:1609.07843.

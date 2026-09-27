# DASE7506 MP1 — Small Language Model

Submitted full-test CPU FP32 BPB: **1.5147** (protocol `7506-mp1-wt2-v2`).

- Assignment: [GUIDE.md](GUIDE.md)
- Report (≤10 pages): [REPORT.md](REPORT.md)
- Install, train, and evaluate: [code/README.md](code/README.md)

The matching checkpoint is **not** in this repository (`runs/` is gitignored). Download `checkpoint.pt` from the release or netdisk link on the course form, then:

```bash
cd code
python evaluate.py --checkpoint /path/to/checkpoint.pt --device cpu --precision fp32 --split test
```

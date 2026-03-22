# PRISM: Progressive Refinement with calIbrated Simulation for Multimodal Recommendation

PRISM is a multimodal recommendation model that addresses two core problems in graph-based recommendation:

1. **Modality noise** — item-item graphs mix useful attributes with spurious noise (backgrounds, redundant text), polluting item representations.
2. **Interaction imbalance** — observed interactions are incomplete due to partial item exposure, leaving tail users/items under-served.

PRISM introduces three contributions: **(1)** selective modality decoupling via a gating mechanism, **(2)** calibrated preference graph generation using user-wise normalized simulation scores, and **(3)** progressive item-item graph refinement that self-improves alongside learned embeddings.

---

## Requirements

```bash
pip install torch numpy scipy scikit-learn matplotlib
```

---

## How to Run

```bash
cd src

# Train on Baby dataset
python main.py --model PRISM --dataset baby

# Train on Sports dataset
python main.py --model PRISM --dataset sports

# Train on Clothing dataset
python main.py --model PRISM --dataset clothing
```

Hyperparameters are configured in `src/configs/model/PRISM.yaml`. Set list values for automatic grid search across seeds.

---

## Results (Recall@10 / NDCG@10, mean ± std over 3 seeds)

| Model    | Baby R@10 | Baby N@10 | Sports R@10 | Sports N@10 | Clothing R@10 | Clothing N@10 |
|----------|-----------|-----------|-------------|-------------|---------------|---------------|
| MGCN     | 0.0620    | 0.0339    | 0.0729      | 0.0397      | 0.0641        | 0.0347        |
| **PRISM**| **0.0641±0.0007** | **0.0346±0.0005** | **0.0754±0.0008** | **0.0416±0.0006** | **0.0672±0.0009** | **0.0368±0.0007** |

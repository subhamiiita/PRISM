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


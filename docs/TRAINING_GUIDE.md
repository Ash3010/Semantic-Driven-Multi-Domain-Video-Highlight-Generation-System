# Optional Training Guide (Not required for the demo)

If you want to train a lightweight model on top of the demo features:

## Datasets
- TVSum
- SumMe

## Features (reuse this demo)
For each scene/segment:
- audio energy stats
- visual proxy stats (or add CLIP embeddings)
- semantic embeddings from transcript

## Model
- Start with XGBoost/LightGBM regressor for importance score
- Or a pairwise ranker (LambdaMART)

## Keep the demo stable
Even if you train later, keep the pretrained+heuristic pipeline as your baseline.

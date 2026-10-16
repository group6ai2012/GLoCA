# CDC + GLoCA Streamlit App

The local Streamlit app runs deterministic single-image inference with the main
seed-42 `CDC + gloca_gated` checkpoints. It supports the reportable PlantSeg and
PlantVillage datasets.

## Run

From the repository root:

```powershell
conda run -n XDC python -m streamlit run streamlit_app.py
```

The first inference can take longer while DINOv2 and the selected checkpoint are
loaded. CUDA is used when available; otherwise inference runs on CPU.

## Required local files

For each selectable dataset, the app expects:

```text
outputs/main_experiments/gloca_gated/cdc/<dataset>/seed_42/
  config.yaml
  assignments.json
  embeddings.pt
  checkpoints/latest.ckpt
```

It also expects the original folder dataset at:

```text
data/raw/plantseg_folder
data/raw/plantvillage
```

These large artifacts and datasets remain local and are not bundled with the app.

## Interpretation

The app ranks anonymous CDC clusters using the calibration head's softmax output.
For each selected cluster, it shows the five original images with the highest
calibration probability for that cluster. These examples are computed from the
saved final GLoCA embeddings, so the app does not re-embed the full dataset.

The output is an unsupervised research result, not a disease diagnosis. A high
cluster probability indicates affinity to a learned cluster and does not establish
the plant's disease class or the correctness of the cluster.

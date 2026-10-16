# Reproducing GLoCA experiments

Run commands from the repository root. The certified target is Windows x64,
the pinned Python/PyTorch/CUDA stack, and the same hardware and NVIDIA driver.
Deterministic execution does not promise identical results on other platforms or
GPU generations. Strict runs fail when an operation has no deterministic implementation.

## Environment and data

```powershell
conda env create -f environment.windows.yml
conda activate gloca-repro
python -m pip check
python -m pytest -q
```

The environment pins the Conda builds from XDC. `requirements-windows.lock.txt`
pins the Windows CPython 3.12 wheels, including their SHA-256 hashes and the CUDA
13.2 wheel index. A compatible NVIDIA driver must already be installed. The first
backbone load downloads the pinned DINOv2 revision. Configuration and weight-file
SHA-256 hashes are checked against the pinned reference and included in provenance.
Keep the dataset and backbone cache unchanged during an invocation. For a repository-local cache:

```powershell
$env:HF_HOME = "$PWD/.cache/huggingface"
```

Obtain PlantVillage and PlantSeg from the sources documented in `dataset.txt`.
Use class folders under `data/raw/plantvillage` and `data/raw/plantseg_folder`.
The existing PlantSeg conversion utility is:

```powershell
python scripts/prepare_plantseg_folder_dataset.py --source-root <raw_plantseg_root>
```

The checked-in manifests contain relative paths, class labels, and image-byte
hashes for 54,305 PlantVillage and 11,458 PlantSeg images. Main recipes verify these
inputs before running. Different dataset exports, image encodings, missing images,
and additional selected images are intentionally rejected. Do not overwrite a
reference manifest just to bypass a mismatch. For an intentional custom dataset,
set `dataset.root` and point `dataset.manifest` at a separately generated manifest:

```powershell
python scripts/build_dataset_manifest.py plantseg --root <folder> --output <new_manifest.json>
```

Custom runs without a reference manifest still record their actual content
fingerprint. Selected classes and per-class limits are verified against the
corresponding subset of the reference manifest. Dataset files must remain unchanged
throughout an invocation: sweeps share one content verification across seeds and
heads to avoid repeatedly hashing the same images.

## Seeds and baseline sweeps

Only seed-42 variants are retained. Adjacent YAML comments document the other
seeds previously used for each recipe. Override the seed explicitly:

```powershell
python scripts/run_cdc.py configs/cdc/cdc_plantseg_cls_seed42.yaml --seed 67
python scripts/run_cdc.py configs/cdc/cdc_plantseg_gloca_gated_seed42.yaml --seed 67
python scripts/run_propos.py configs/propos/propos_plantseg_cls_full_seed42.yaml --seed 69
python scripts/run_propos.py configs/propos/propos_plantseg_gloca_gated_full_seed42.yaml --seed 69
python scripts/run_propos_gloca_diagnostics.py configs/diagnostics/propos_gloca_diagnostics_plantwild_seed42.yaml --seed 67
```

`--seed N` replaces the YAML seed, including every diagnostic run after merging.
The original YAML remains unchanged. Seeds must be integers from 0 to 4294967295.
The effective seed appears in paths, saved config, assignments, metrics, and
checkpoint identity; the command is recorded in provenance. Experiment settings
other than these explicit seed overrides remain YAML-defined.

The sole baseline entry point accepts every existing baseline recipe family:

```powershell
python scripts/run_baselines.py configs/baselines/baselines_unified.yaml
python scripts/run_baselines.py configs/baselines/baselines_unified.yaml --seed 67
python scripts/run_baselines.py configs/baselines/baselines_unified.yaml --seeds 42 67 69
```

`--seed` and `--seeds` are mutually exclusive. `--seeds` replaces the entire YAML
seed list, rejects duplicates, and runs all requested seeds in one sweep. All
retained recipes default to 42. Each invocation replaces its own
`baseline_summary.csv` and `baseline_summary_agg.csv`; it does not discover seeds
from older invocations. Use one `--seeds` invocation for a multi-seed aggregate.
Failed runs remain in the per-run summary and make the process exit nonzero.

The combined recipe includes nine baseline variants on PlantVillage and PlantSeg.
It preserves each method's existing schedule and evaluation semantics. In
particular, supervised MLP test metrics are a supervised reference, and fixed
CLS-plus-mean-patch features are distinct from GLoCA. Historical PlantWild recipes
remain smoke/debug context, not independent main evidence.

## Strict execution, resume, and provenance

`trainer.deterministic` defaults to `true`. Strict execution uses deterministic
algorithms, one numerical thread, independently seeded loaders, and sample/epoch
augmentation seeds. Parallel loaders remain available, with persistent workers
disabled so epoch-boundary resume recreates their state. Evaluation and diagnostic
logging do not change training randomness. Null/omitted matmul precision resolves
to `highest`; explicit `high`/`medium` settings remain effective. HDBSCAN uses one job.

Clustering-head initialization is matched across CLS and GLoCA paths. Training
objectives, default image preprocessing, adapter math, and the frozen-backbone
contract are preserved. This controlled setup changes random-number consumption
relative to legacy runs; historical numerical results are not retroactively certified.

`reproducibility.json` records resolved config/input identity, source fingerprint,
Git revision/dirty status, Python/package versions, CUDA/cuDNN, hardware/driver,
and invocation. `completed.json` is written last with artifact hashes, including
conditional attention, checkpoints, and CDC exports. A lone metrics file cannot
cause a run to be skipped. Corrupt or missing artifacts do not count as complete.

Strict execution refuses an existing directory with incompatible or missing
provenance. For new strict experiments, set a fresh `experiment.output_dir` in the
YAML rather than reusing historical result directories. Output roots, dataset
locations, and resume paths are excluded from scientific identity; input bytes,
source code, seed, hardware/software, and scientific settings remain part of it.

Keep the full intended schedule in the config when interrupting a run. Resume with
the same config and seed, using `trainer.resume_from_checkpoint: auto` or an
explicit existing checkpoint path. CDC/ProPos resume at the next epoch; DEC/IDEC
resume their saved phase. MLP restores its splits, optimizer, random state, and
best validation weights. A changed schedule or seed requires a new experiment.
Use a periodic file under `checkpoints/` for resume, rather than the canonical
final `checkpoint.ckpt` export.
To load an old checkpoint lacking reproducibility metadata, explicitly select
`trainer.deterministic: false`; that path has no exact-resume guarantee.

## Short validation runs

```powershell
python scripts/run_baselines.py configs/baselines/baselines_repro_smoke.yaml --seeds 42 67
python scripts/run_cdc.py configs/cdc/cdc_plantseg_cls_repro_smoke.yaml --seed 67
python scripts/run_cdc.py configs/cdc/cdc_plantseg_gloca_gated_repro_smoke.yaml --seed 67
python scripts/run_propos.py configs/propos/propos_plantseg_cls_repro_smoke.yaml --seed 67
python scripts/run_propos.py configs/propos/propos_plantseg_gloca_gated_repro_smoke.yaml --seed 67
```

These use six real PlantSeg images and short schedules. They check pipeline
behavior and artifact compatibility, not disease-discovery quality. For a fresh
repeat, change only the output root; compare tensors, assignments, and scientific
metrics, excluding timing/resource fields and serialization bytes. Repeating an
unchanged baseline command should verify and skip completed runs.

## Validation performed

On 2026-10-16, all 161 tests passed from an empty project copy using the rebuilt
Windows environment. Coverage includes exact interrupted-versus-uninterrupted
training for CDC/ProPos with and without GLoCA, DEC, IDEC, and MLP; a full Windows
multiworker CDC/GLoCA resume comparison; seed overrides; content verification;
completion checks; and canonical output compatibility.

Real DINOv2 CUDA smoke runs on an NVIDIA GeForce RTX 4070 Laptop GPU verified:

- 18 baseline fresh-repeat pairs across seeds 42 and 67, with identical scientific outputs.
- Nine baseline restored/fresh comparisons and 18 validated skips.
- Four live-image fresh-repeat pairs and four restored/fresh comparisons, covering
  CDC/ProPos with CLS and GLoCA, with identical tensors, assignments, and scientific metrics.
- All five ProPos/GLoCA diagnostic variants with the CLI seed override.

All 72 locked wheels passed download/hash verification, and `pip check` passed in
the rebuilt environment. These checks use short debug schedules and small image
subsets; full research-length experiments were not rerun.

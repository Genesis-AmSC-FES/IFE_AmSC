
# RHINO Surrogate Training

This layer trains and evaluates a multi-output neural-network surrogate using
the CSV produced by `AI_ready_workflow/3_feature_extraction`. It preserves the
feature-table provenance needed to reproduce the test split and writes model,
metric, prediction, and plotting artifacts.

## Relationship to the Feature Layer

The current feature table begins with campaign identity and simulation-time
metadata, followed by declared model inputs and outputs:

```text
archive, datasetid, run_id,
simulation_date, simulation_time, simulation_datetime,
<model inputs>, <model outputs>
```

Identity and datetime columns are not passed to the neural network or included
in normalization. By default, `trainSurrogate.py` reads the `inputs` and
`outputs` keys from the feature layer's `feature_spec.json`; `metadata` entries
are intentionally excluded. The available metadata columns are retained in
`split_indices.json` and copied into `test_predictions.csv`, allowing every
prediction to be traced back to its campaign run.

The trainer automatically loads the feature layer's adjacent
`dataset_manifest.json` (or `<features>.manifest.json` for an explicit export),
verifies its CSV digest and row count, and carries that provenance into the
training artifacts. The trainer rejects missing columns, nonnumeric model
columns, NaN or infinite model values, and datasets too small to create
non-empty training, validation, and test splits. Missing feature values should
therefore be resolved in the feature/campaign layers rather than silently
dropped during training.

## Requirements

- Python 3.10 or newer
- NumPy and pandas
- PyTorch
- Matplotlib for evaluation plots
- MLflow when remote tracking or model registration is enabled
- A completed `rhino_features.csv` and, preferably, its dataset manifest

The RHINO package and its I/O dependencies are required to rebuild the
upstream BP5 and feature products, but training itself reads the resulting CSV.
Install the training dependencies from the RHINO package root with:

```bash
python -m pip install -e '.[ml]'
```

Use `'.[io,ml]'` when the same environment must also rebuild BP5 products.

## Training Workflow

```text
feature_spec.json + rhino_features.csv
                  |
                  v
Validate selected model columns and finite values
                  |
                  v
Seeded 80% / 15% / 5% train-validation-test split
                  |
                  v
Normalize from training statistics only
                  |
                  v
Train MLP and retain the best validation state
                  |
                  v
Model, normalization, dataset fingerprint, and split metadata
```

The default network has three hidden layers, 100 units per hidden layer, ReLU
activations, and one output node per declared target. It uses Adam and
mean-squared error on normalized targets. Training always runs for the requested
number of epochs, but the saved checkpoint contains the weights from the epoch
with the lowest validation loss rather than necessarily the final epoch.

Each feature-table row represents one unique campaign run, so the default
seeded row split does not divide a single run between subsets. It estimates
interpolation performance for runs drawn from the same sampled campaign
distribution. It does not, by itself, establish accuracy for extrapolation
beyond the training input ranges or for a materially different future
campaign.

## Files

### `trainSurrogate.py`

Loads and validates the feature CSV, selects input/output columns, creates
seeded data splits, normalizes values, trains the model, and saves:

```text
artifacts/
├── rhino_surrogate.pt
├── training_history.csv
├── split_indices.json
├── input_example.csv
├── run_manifest.json
└── provenance/
    ├── feature_spec.json
    └── dataset_manifest.json
```

`rhino_surrogate.pt` contains model weights, architecture, column names,
normalization statistics, and best validation loss. `split_indices.json`
contains source row indexes, the resolved CSV, dataset-manifest, and
feature-spec paths, their SHA-256 fingerprints, row count, metadata columns,
random seed, and—when enabled—the MLflow dataset records, run, model URI,
registered-model name, and version.

`run_manifest.json` makes the directory a portable deferred-upload bundle. It
records all training controls, package versions, Git state, selected Slurm and
NERSC metadata, artifact checksums, and dataset provenance. `input_example.csv`
allows the serving signature to be recreated even when the uploader cannot
access the original feature CSV. Available feature and dataset specifications
are copied into `provenance/`.

### `testSurrogate.py`

Reconstructs the held-out split, verifies the CSV fingerprint and model-column
contract, generates predictions, and saves:

```text
artifacts/
├── metrics.json
├── test_predictions.csv
└── parity_plots.png
```

Metrics include aggregate and per-output MSE, RMSE, MAE, and maximum absolute
error. Per-output metrics are the meaningful comparison when targets have
different units. The prediction CSV includes campaign identity, simulation
datetime, model inputs, original source-row number, true values, predictions,
and absolute errors.

### `mlflow_model.py`

Defines the serving interface registered with MLflow. It accepts a pandas
table with raw, named RHINO input columns, applies the saved input
normalization, evaluates the PyTorch network, reverses output normalization,
and returns a table with named outputs in physical units. This prevents API
clients from having to reproduce training normalization.

## Usage

The commands can be run from any working directory. Pass the immutable CSV
emitted by the feature layer:

```text
/shared/rhino/ml-datasets/<12-char-sha256>/rhino_features.csv
```

Train using the feature JSON's declared inputs and outputs:

```bash
python RHINO/ML/surrogate_training/trainSurrogate.py \
  --features /shared/rhino/ml-datasets/<12-char-sha256>/rhino_features.csv
```

Use a different feature CSV or explicitly select model columns:

```bash
python RHINO/ML/surrogate_training/trainSurrogate.py \
  --features /path/to/rhino_features.csv \
  --inputs tritium_burning_rate burn_fraction \
  --outputs plant_doubling_time_days \
            minimum_startup_inventory_g \
            tritium_in_isotope_separation
```

The adjacent manifest is discovered automatically. Use `--dataset-manifest`
only when it is stored elsewhere. Other useful training controls include
`--epochs`, `--batch-size`, `--hidden-dim`, `--hidden-layers`, `--lr`, `--seed`,
`--feature-spec`, and `--outdir`. Run the training command with `--help` for the
complete interface.

Evaluate the saved model on its held-out rows. When training used a custom
output directory, pass that same directory explicitly; shell variables such as
`RUN_DIR` must be set again in each new shell session:

```bash
export RUN_DIR=/path/to/run-directory

python RHINO/ML/surrogate_training/testSurrogate.py \
  --model "$RUN_DIR/rhino_surrogate.pt" \
  --splits "$RUN_DIR/split_indices.json" \
  --outdir "$RUN_DIR"
```

If the exact training CSV was moved without being changed, provide its new
location. The SHA-256 fingerprint must still match:

```bash
python RHINO/ML/surrogate_training/testSurrogate.py \
  --features /new/location/rhino_features.csv
```

Existing artifacts created by the earlier scripts do not contain the new CSV
fingerprint or metadata-column declarations. Retrain after rebuilding the
feature CSV to obtain the complete provenance and prediction schema.

## Assessing model quality

A completed command and a saved checkpoint show that training operated
correctly; they do not by themselves show that the surrogate is scientifically
adequate. Review the held-out results before registering or using a model:

```bash
python -m json.tool "$RUN_DIR/metrics.json"
tail -n 20 "$RUN_DIR/training_history.csv"
```

Also inspect `parity_plots.png` and the largest errors in
`test_predictions.csv`. In particular:

- Use the per-output RMSE, MAE, and maximum absolute error in physical units.
  The aggregate values combine targets with different units and scales and are
  not suitable as the primary scientific acceptance criterion.
- Compare errors with a domain-relevant tolerance and a simple baseline, such
  as predicting each training-set target mean. A neural network should improve
  materially on that baseline.
- Confirm that both training and validation losses settle rather than diverge.
  A falling training loss with a rising validation loss indicates overfitting;
  uniformly high losses indicate underfitting, unsuitable inputs, or an
  optimization problem.
- Treat the default three-layer, 100-unit network as a baseline rather than an
  established optimum. Compare a smaller network and more than one seed. Use
  validation results for those choices and reserve the test split for the final
  unbiased comparison; repeatedly choosing settings from test results leaks
  test information into model selection.
- Check the recorded split counts in `run_manifest.json`. The default test
  fraction is 5%; for a small table this can yield too few test cases for a
  stable estimate. Repeated seeds or cross-validation should be used for a
  stronger small-data assessment, although this trainer currently implements
  one fixed split per run.
- Treat predictions outside the ranges represented by the training inputs as
  extrapolation. The current model does not calculate uncertainty or reject
  out-of-distribution requests.

The seed makes the CPU split, initialization, and data-loader order repeatable
for a fixed software environment. GPU numerical kernels and changes in PyTorch
or hardware are not guaranteed to be bit-for-bit deterministic; the manifest
records the environment needed to interpret such differences.

## MLflow Tracking and Model Registry

MLflow is optional: commands without `--mlflow` retain the local-only behavior
described above. When enabled, one training execution creates one MLflow run
and logs:

- Network, optimizer, split, column, and seed parameters
- Training and validation loss at every epoch
- Best validation loss
- Full source, training, and validation datasets as MLflow run inputs
- Dataset-manifest, CSV, and feature-spec SHA-256 provenance
- Native checkpoint, training history, split metadata, feature spec, and
  dataset manifest
- A signature-bearing serving model with normalization included

The MLflow dataset entries are references and fingerprints, not uploads of the
full CSV. The durable shared dataset path remains the source of truth. The
small manifest is copied into the run's `provenance/` artifacts so the run
retains the full extraction record even if paths are reorganized later.

By default, the serving model is registered as `rhino-surrogate`. Pass
`--skip-model-registration` to log it only as a run model, or use
`--registered-model-name` to choose another registry name. Model Registry
requires a database-backed MLflow store.

### Start a small tracking server

For personal testing, use a persistent SQLite database and artifact directory:

```bash
mlflow server \
  --backend-store-uri sqlite:////path/to/mlflow/mlflow.db \
  --artifacts-destination file:///path/to/mlflow/artifacts \
  --host 127.0.0.1 \
  --port 5000
```

Use PostgreSQL and managed/shared artifact storage for a multi-user production
service. Put remotely accessible servers behind HTTPS and authentication; do
not expose an unauthenticated tracking server publicly.

### Train and log

Point the client at the server and enable logging explicitly:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

python RHINO/ML/surrogate_training/trainSurrogate.py \
  --mlflow \
  --mlflow-experiment rhino-surrogate \
  --mlflow-run-name baseline-2026-04-29
```

`--mlflow-tracking-uri` may be used instead of the environment variable. If
MLflow logging fails, the MLflow run is marked failed by its run context.

### Evaluate and attach results

Evaluation reads the originating run ID from `split_indices.json`, reopens that
run, logs the held-out rows as a `testing` dataset input, and logs
aggregate/per-output test metrics and evaluation artifacts:

```bash
python RHINO/ML/surrogate_training/testSurrogate.py --mlflow
```

After reviewing the metrics, an alias can be assigned explicitly:

```bash
python RHINO/ML/surrogate_training/testSurrogate.py \
  --mlflow \
  --model-alias candidate
```

Use an alias such as `champion` only after the version satisfies the project's
validation policy. The alias and registered version are not created when
training uses `--skip-model-registration`.

## NERSC execution and deferred MLflow upload

For the current small MLP and a modest feature table, a short CPU run is usually
sufficient. NERSC permits short, lightweight Python work on login nodes. Use a
compute allocation if the run becomes sustained, CPU- or memory-intensive. A
direct run still produces the complete portable bundle; Slurm metadata in the
manifest is simply absent.

```bash
export RUN_DIR="$PSCRATCH/rhino-runs/manual-test"
mkdir -p "$RUN_DIR"

python RHINO/ML/surrogate_training/trainSurrogate.py \
  --features /global/cfs/cdirs/m3239/2026_FES-AmSC/data/rhino/ml-datasets/<dataset-id>/rhino_features.csv \
  --outdir "$RUN_DIR"

python RHINO/ML/surrogate_training/testSurrogate.py \
  --model "$RUN_DIR/rhino_surrogate.pt" \
  --splits "$RUN_DIR/split_indices.json" \
  --outdir "$RUN_DIR"
```

For a longer or larger run, submit the included single-GPU template from the
repository root. Confirm that the account, QOS, time limit, and environment
name are appropriate for the NERSC allocation first:

```bash
sbatch --export=ALL,FEATURES=/global/cfs/cdirs/<project>/path/rhino_features.csv \
  RHINO/ML/surrogate_training/train_nersc.slurm
```

The job writes a unique bundle under `$PSCRATCH/rhino-runs/$SLURM_JOB_ID` by
default. Scratch is temporary and purgeable; copy completed bundles to CFS,
HPSS, or another durable system.

Choose the MLflow destination only when uploading. The command-line option
takes precedence over the environment variable:

```bash
python RHINO/ML/surrogate_training/uploadRunToMlflow.py \
  --run-dir /path/to/rhino-runs/<job-id> \
  --mlflow-tracking-uri https://mlflow.example.org \
  --experiment rhino-surrogate \
  --run-name perlmutter-<job-id>
```

Equivalently, set the destination once in the upload environment:

```bash
export MLFLOW_TRACKING_URI=https://mlflow.example.org
python RHINO/ML/surrogate_training/uploadRunToMlflow.py \
  --run-dir /path/to/rhino-runs/<job-id>
```

The uploader deliberately requires one of those settings so a deferred upload
cannot silently land in MLflow's default local store. It verifies the bundle,
creates a new run, replays per-epoch metrics, logs the complete bundle, creates
the signature-bearing PyFunc model, and registers it as `rhino-surrogate`.
Use `--skip-model-registration` to omit registry creation.

The original feature CSV is not required to upload or serve the model. If it is
available on the upload machine, pass `--features /current/path/features.csv`;
after checking its SHA-256 digest, the uploader also recreates the source,
training, and validation MLflow dataset inputs. A successful upload writes an
`mlflow_upload_<run-id>.json` receipt into the bundle. Credentials should be
provided through the MLflow deployment's normal environment or credential
provider, not embedded in the tracking URI or saved bundle.

### Serve a registered version

Registration stores and versions the model but does not create a continuously
running inference endpoint. For local validation, serve a selected alias with:

```bash
mlflow models serve \
  --model-uri 'models:/rhino-surrogate@candidate' \
  --port 8080
```

The endpoint accepts raw physical inputs by name:

```bash
curl http://127.0.0.1:8080/invocations \
  -H 'Content-Type: application/json' \
  -d '{
    "dataframe_split": {
      "columns": ["tritium_burning_rate", "burn_fraction"],
      "data": [[82.4, 0.035]]
    }
  }'
```


## DagsHub MLflow Tracking on NERSC

RHINO jobs can stream training progress, provenance, models, and evaluation
artifacts to the shared DagsHub MLflow service while NERSC remains the source
of truth for the feature CSV and job-local run bundle. The CSV is referenced by
its path, manifest, and SHA-256 digest; it is not uploaded on every run.

### Configure a Jupyter terminal

Create a DagsHub access token, then configure the terminal that submits Slurm
jobs:

```bash
export MLFLOW_TRACKING_URI="https://dagshub.com/cbhowmic/rhino-mlflow.mlflow"
export MLFLOW_TRACKING_USERNAME="<your-dagshub-username>"
export MLFLOW_TRACKING_PASSWORD="<your-dagshub-token>"
export MLFLOW_EXPERIMENT="rhino-surrogate"
```

Never commit the token or place it in a Slurm script. These values apply to the
current shell. `sbatch --export=ALL` passes them to the compute allocation
without writing credentials into the repository.

### Submit a live-tracked job

When `MLFLOW_TRACKING_URI` is set, `train_nersc.slurm` enables tracking,
requires the adjacent `dataset_manifest.json`, and gives the run the default
name `perlmutter-<Slurm job id>`.

```bash
cd /global/homes/b/bhowmic/Projects/IFE_AmSC
sbatch --export=ALL,FEATURES=/global/homes/b/bhowmic/Projects/IFE_AmSC/RHINO/AI_ready_workflow/3_feature_extraction/outputs/rhino_features.csv \
  RHINO/ML/surrogate_training/train_nersc.slurm
```

For a short end-to-end check, override the epoch count:

```bash
sbatch --export=ALL,EPOCHS=10,FEATURES=/global/homes/b/bhowmic/Projects/IFE_AmSC/RHINO/AI_ready_workflow/3_feature_extraction/outputs/rhino_features.csv \
  RHINO/ML/surrogate_training/train_nersc.slurm
```

Runs appear in [DagsHub MLflow](https://dagshub.com/cbhowmic/rhino-mlflow/experiments) after they start. Refresh the experiment page or clear filters if
a new run is not immediately visible.

### Live metrics and artifacts

`trainSurrogate.py` logs the experiment configuration, Git and dataset
provenance, model signature, model artifact, checkpoint, and per-epoch metrics:

- `train_loss` and `val_loss`: normalized-target mean squared error.
- `train_r2` and `val_r2`: variance-weighted R-squared across surrogate outputs.

This is multi-output regression, so classification-style percentage accuracy is
not meaningful. R-squared is the appropriate accuracy-like measure: `1.0` is
perfect, `0.0` equals a mean-value baseline, and negative values are worse
than that baseline. A domain-specific "within tolerance" percentage can be
added later once physical tolerances are chosen for each output.

`testSurrogate.py` reopens the same MLflow run after training and logs final
test metrics, predictions, `metrics.json`, and parity plots. The matching
NERSC bundle is stored at `$PSCRATCH/rhino-runs/<Slurm job id>`.

### Data split interpretation

The seeded split is 80% training, 15% validation, and 5% held-out test data.
Training loss updates the model weights. Validation loss/R-squared are measured
each epoch on unseen validation data and the checkpoint with lowest validation
loss is retained. Test metrics are computed once after model selection on the
separate held-out samples, making them the final unbiased estimate. Validation
loss is therefore not the same as test loss.

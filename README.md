# IFE_AmSC

IFE_AmSC provides an end-to-end path for turning heterogeneous experimental and
simulation outputs into AI-ready data and using those data in machine-learning
applications. We developed a modular workflow that gets the raw data AI-ready,
by making it self-describing, accessible, discoverable, and ready for downstream ML applications.

## Workflow Overview

```text
Raw experimental or simulation data
        |
        v
+------------------------------------------+
| AI-ready data workflow                   |
| 1. Standardization (shim)                |
| 2. Campaign organization and discovery  |
| 3. Feature extraction                    |
+------------------------------------------+
        |
        v
ML-ready dataset
        |
        v
Machine-learning applications
        |
        v
Models, predictions, and analysis products
        |
        v
Inference, visualization, and decision support
```

The workflow is organized into three layers:

1. **Shim layer:** The shim converts application-specific raw outputs into
   a consistent openPMD/ADIOS2 representation. This establishes common data and
   metadata conventions while preserving scientific context and provenance.
2. **Campaign layer:** HPC Campaign packages standardized datasets in
   checksummed TAR replicas, groups them into campaign archives, and builds
   searchable indexes. This makes runs easier to locate, compare, query, and
   share without manually inspecting individual files, while enabling remote
   access to federated data.
3. **Feature extraction layer:** Query-driven extraction transforms campaign data into
   curated tables of features and targets suitable for ML and scientific
   analysis. This creates a reproducible boundary between data preparation and
   model development, focusing on the QoI suitable for the scientific data and ML application.


**ML applications:** AI-ready datasets can support surrogate modeling, ML-driven control, anomaly detection etc as the downstream ML applications.

## Project Status

RHINO is the first implemented application workflow. It currently supports:

1. Converting raw RHINO outputs to openPMD/ADIOS2 BP5 datasets
2. Organizing and indexing those datasets with HPC Campaign
3. Extracting ML-ready features from campaign data
4. Training and evaluating surrogate models
5. Running a local surrogate inference and visualization demo

LASER support is planned and will be added as the next application workflow.

The Python packaging under `RHINO/src/` and `RHINO/pyproject.toml` is under
active development. The package structure, installation process, and
command-line interfaces may change as more of the workflow is incorporated into
the package.

## Repository Structure

```text
IFE_AmSC/
|-- environment.yml
|-- schema.md
|-- RHINO/
|   |-- AI_ready_workflow/
|   |   |-- 1_shim/
|   |   |-- 2_campaign/
|   |   `-- 3_feature_extraction/
|   |-- ML/
|   |   `-- surrogate_training/
|   |-- demo/
|   |   `-- surrogate/
|   |-- src/
|   |   `-- rhino/
|   |-- tests/
|   `-- pyproject.toml
`-- LASER/                         # Planned
```

The `AI_ready_workflow` directory contains the three ordered stages that turn
raw application output into AI-ready data. Machine-learning training is kept
separate under `ML`, while runnable inference and visualization applications
belong under `demo`.

## Environment Setup

The root `environment.yml` defines the shared environment for the RHINO shim,
campaign tools, feature extraction, surrogate training, notebooks, and demo.

Create and activate the environment from the repository root:

```bash
conda env create -f environment.yml
conda activate IFE_AmSC
```

To update an existing environment after the dependency file changes:

```bash
conda activate IFE_AmSC
conda env update --prefix "$CONDA_PREFIX" --file environment.yml
```

The surrogate visualization uses Node.js, which is included in the Conda
environment. Its frontend dependencies are installed separately:

```bash
npm --prefix RHINO/demo/surrogate/react_flow_viz ci
```

## Installation on NERSC

1. Load Python:

   ```bash
   module load python
   ```

   Loading the module activates NERSC's base environment, shown as
   `(nersc-python)`. Activate the project environment after loading the module.

2. Clone the repository:

   ```bash
   git clone git@github.com:cbhowmic/IFE_AmSC.git
   cd IFE_AmSC
   ```

3. Create and activate the project environment:

   ```bash
   conda env create --file environment.yml
   conda activate IFE_AmSC
   ```

   For an environment that already exists, update the active environment
   explicitly:

   ```bash
   conda activate IFE_AmSC
   conda env update --prefix "$CONDA_PREFIX" --file environment.yml
   ```

4. Verify the environment:

   ```bash
   which python
   python -m pip check
   python -c "import torch, rhino, openpmd_api, adios2; print('Environment ready')"
   python -m pytest RHINO/tests -q
   ```

   `which python` should resolve inside the `IFE_AmSC` Conda environment.

5. Optional: register the environment as a Jupyter kernel:

   ```bash
   python -m ipykernel install --user \
       --name ife_amsc \
       --display-name "IFE AmSC"
   ```

   Open [NERSC Jupyter](https://jupyter.nersc.gov/), select a login node, and
   choose the `IFE AmSC` kernel. Terminal workflows do not require this step;
   activate `IFE_AmSC` in each new terminal or batch job instead.

Login nodes are appropriate for setup, tests, and lightweight conversions.
Run long, resource-intensive, or GPU workloads in a Slurm compute-node job.

## Data Products

The workflow produces several kinds of data and model artifacts:

- `.bp5`: standardized openPMD/ADIOS2 simulation datasets
- `.tar`, `.sha256`, and `.tar.idx`: verified, indexed archival replicas
- `.aca`: campaign archives
- `.acx`: searchable campaign indexes
- `.csv` or `.parquet`: ML-ready feature tables
- `.pt`: trained PyTorch model artifacts

Raw simulation data and large generated artifacts should generally be stored in
the appropriate project data location rather than committed to Git.


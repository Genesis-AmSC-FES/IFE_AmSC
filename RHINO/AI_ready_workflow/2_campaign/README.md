# RHINO Campaign Management Layer

This layer organizes the openPMD/ADIOS BP5 datasets produced by the RHINO shim
layer into date-based HPC Campaign archives and a searchable campaign index. It
preserves the original datasets as live replicas, creates and records a portable
TAR copy for each day, and builds a SQLite-backed `.acx` index over the generated
`.aca` archives. The index provides one entry point for discovering archives and
their datasets, inspecting campaign metadata, and selecting data for downstream
queries, feature extraction, and analysis.

The Python entry point, `create_archives.py`, is the canonical implementation for the
current NERSC workflow.

## Artifact model

A `.bp5` dataset is an ADIOS dataset directory containing the scientific data.
A `.aca` file is a small HPC Campaign metadata archive: it identifies datasets,
run IDs, and the storage replicas from which those datasets may be accessed. The
scientific payload is not embedded in the `.aca` file.

The workflow creates one campaign archive and one TAR for each configured date:

```text
Input date directory              Campaign archive          Archival copy
2026-04-29/*.bp5             ->   rhino-2026-04-29.aca  ->  rhino-2026-04-29.tar
2026-04-30/*.bp5             ->   rhino-2026-04-30.aca  ->  rhino-2026-04-30.tar
2026-05-01/*.bp5             ->   rhino-2026-05-01.aca  ->  rhino-2026-05-01.tar
```

The TAR is not produced by feeding the serialized `.aca` file to `tar`.
Instead, the script first constructs the exact dataset list for the date-based
ACA and then uses that same list as the TAR membership list. Consequently, the
ACA and TAR describe the same set of BP5 datasets. Dataset paths inside the TAR
remain relative to `RHINO_DATA_ROOT`, allowing `hpc_campaign taridx` to match
TAR members to the dataset paths recorded in the ACA.

## Archive workflow

```text
Discover top-level *.bp5 datasets in each configured date directory
    |
    v
Create one date-based .aca and add its live BP5 dataset references
    |
    v
Create one uncompressed TAR from exactly that ACA's dataset list
    |
    v
Write a SHA-256 checksum and create the TAR index
    |
    v
Register the NERSC TAR location in the corresponding .aca
    |
    v
Publish completed .aca files to the shared project campaign store
    |
    v
Build the searchable campaign index (.acx) from the published copies
```

For each date, `create_archives.py` performs these operations:

1. Loads and validates `campaign_spec.json`.
2. Discovers top-level `*.bp5` datasets in the configured date directory.
3. Creates `<ARCHIVE_PREFIX>-<date>.aca` and assigns each dataset its RHINO
   run ID.
4. Creates `<TAR_PREFIX>-<date>.tar` from exactly those datasets.
5. Writes `<tar-name>.sha256` and `<tar-name>.idx`.
6. Registers the TAR location with that ACA using
   `hpc_campaign manager ... add-archival-storage`.

HPC Campaign records the location of a replica; it does not copy or upload the
TAR. File transfer is a separate operational step.

## Current NERSC configuration

The current configuration records one TAR copy per ACA. That copy is stored on
the NERSC project filesystem under:

```text
/global/cfs/cdirs/m3239/2026_FES-AmSC/data/rhino/
```

The generated products have names such as:

```text
rhino-2026-04-29.tar
rhino-2026-04-29.tar.sha256
rhino-2026-04-29.tar.idx
```

The campaign store containing the ACA metadata is configured as:

```text
/global/homes/b/bhowmic/campaign-store/IFE/
```

It contains date-based archives such as:

```text
rhino-2026-04-29.aca
rhino-2026-04-30.aca
rhino-2026-05-01.aca
```

The ACA files are first created in the private working store. The publication
stage copies completed ACA files to the shared project store:

```text
/global/cfs/cdirs/m3239/2026_FES-AmSC/campaigns/IFE/
```

The authoritative `.acx` index is built in this shared directory after
publication. The current archive stage still records only the NERSC TAR copy; a
second facility TAR replica remains part of the later facility handoff.

## Requirements

- Python 3.10 or newer
- `hpc-campaign` 0.7 with `hpc_campaign` available on `PATH`
- `tar` and `rsync`
- RHINO BP5 datasets in the configured input directories
- `sqlite3` for running the supplied SQL queries directly

Run commands from this directory:

```bash
cd RHINO/AI_ready_workflow/2_campaign
```

## Configuration

The Python workflow reads `campaign_spec.json`.

| Setting | Purpose |
| --- | --- |
| `RHINO_DATA_ROOT` | Root directory used to resolve and record BP5 dataset paths |
| `CAMPAIGN_STORE` | Private working directory containing generated ACA files |
| `PUBLISH_CAMPAIGN_STORE` | Shared project directory receiving completed ACA files and the authoritative index |
| `CAMPAIGN_NAMESPACE` | Logical campaign name shared by the archive and index configuration |
| `ARCHIVE_PREFIX` | Prefix for date-based ACA names, such as `rhino` |
| `CAMPAIGN_INDEX` | Index path; a relative path is resolved from `PUBLISH_CAMPAIGN_STORE` |
| `TAR_OUTPUT_DIR` | Directory in which TAR products are created |
| `TAR_PREFIX` | Prefix for date-based TAR names, such as `rhino` |
| `TAR_STORAGE_SYSTEM` | HPC Campaign storage type, currently `fs` |
| `TAR_STORAGE_HOST` | Host label recorded for the TAR replica, currently `NERSC` |
| `INPUT_DIRS` | Date directories to scan under `RHINO_DATA_ROOT` |

Each `INPUT_DIRS` entry must be relative to `RHINO_DATA_ROOT`. Only
top-level `*.bp5` entries are discovered; the search is not recursive.

## Running archive creation

Preview discovery, date grouping, and generated commands without modifying
files:

```bash
python create_archives.py --dry-run
```

Create the ACAs, TAR products, and NERSC replica registrations:

```bash
python create_archives.py
```

The first dataset added to each ACA uses `--truncate`; therefore, an existing ACA
with the same date-based name is replaced during a real archive build.

Existing TARs are reused only if their SHA-256 checksum succeeds and their
members match the associated ACA dataset list. If an older TAR contains the
whole date directory or otherwise differs from the ACA, the script stops and
requests an explicit rebuild:

```bash
python create_archives.py --rebuild-tars
```

This option replaces the TAR, checksum, and TAR index. Use it deliberately
because rebuilding a large daily TAR may be expensive.

A different specification may be supplied with:

```bash
python create_archives.py --spec /path/to/campaign_spec.json
```

## Publishing campaign archives

`publish_campaign.py` is the required stage between archive creation and index
creation. It finds completed `<ARCHIVE_PREFIX>-*.aca` files in
`CAMPAIGN_STORE` and copies only those ACA files to
`PUBLISH_CAMPAIGN_STORE` with `rsync -a --checksum`. It does not copy a
personal `.acx` index or delete older published archives.

Preview the source archives, destination, and rsync command without changing
files:

```bash
python publish_campaign.py --dry-run
```

Publish the completed ACA files:

```bash
python publish_campaign.py
```

After rsync completes, the script calculates SHA-256 digests of the source and
published ACA files and stops if any copy is missing or differs. Index creation
must run only after this stage succeeds.

The complete Python workflow order is:

```bash
python create_archives.py
python publish_campaign.py
python create_index.py
```

## Using the workflow at the RHINO facility

The same organization can manage new data where it is generated. At the
facility, scientists should configure:

- `RHINO_DATA_ROOT` for the facility's RHINO output root.
- `INPUT_DIRS` for the completed date directories.
- `CAMPAIGN_STORE` for the facility's ACA metadata store.
- `TAR_OUTPUT_DIR` for durable facility archive storage.
- `TAR_STORAGE_HOST` with a short, stable facility host or site label.

Running the workflow there will create one ACA and one local TAR per date and
register the facility TAR as the first archival replica. The facility can keep
this TAR as its retained copy.

To maintain a second copy at NERSC, the operational handoff is:

1. Transfer the TAR, `.sha256`, and `.idx` files to the designated NERSC
   project-data directory using the facility's approved transfer mechanism,
   such as Globus.
2. Verify the transferred TAR against its SHA-256 checksum at NERSC.
3. Register the verified NERSC TAR location in the same ACA as an additional
   archival-storage replica.
4. Publish or copy the final updated ACA to the location from which the
   campaign index will be built.

One ACA can describe both physical copies; a second ACA is not required. The
facility and NERSC copies should use the same TAR filename and content so the
same TAR index describes both. A NERSC location must not be registered before
the upload and checksum verification succeed, because registration records a
location but does not prove that the file exists there.

The current script automates the single local/NERSC registration used during
development. Automating the facility-to-NERSC transfer and adding the second
replica are handoff extensions and are intentionally not performed by the
current NERSC run.

## Campaign index

After publication, `create_index.py` builds the authoritative searchable
`.acx` index directly in `PUBLISH_CAMPAIGN_STORE`:

```bash
python create_index.py --dry-run
python create_index.py
```

It discovers archives matching `ARCHIVE_PREFIX*.aca` in the published store,
recreates `CAMPAIGN_INDEX` there, registers the published ACA paths, and lists
the resulting index. If ACA files are republished later, rebuild the index.

## Bash entry points

Shell entry points remain available for compatibility and read
`config_campaign.sh`:

```bash
bash create_archives.sh --dry-run
bash create_archives.sh
bash create_archives.sh --rebuild-tars
bash create_index.sh
```

Use either the Python or Bash archive entry point for a given build. The Python
entry point documents and enforces the current ACA-driven TAR membership and is
recommended for the NERSC workflow.

## SQL queries

The `queries/` directory contains reusable SQL for campaign inspection,
metadata discovery, scientific analysis, feature extraction, and surrogate
model dataset generation. For an index at
`<PUBLISH_CAMPAIGN_STORE>/rhino.acx`, a query can be run with:

```bash
sqlite3 /global/cfs/cdirs/m3239/2026_FES-AmSC/campaigns/IFE/rhino.acx < queries/list_archives.sql
```

See `queries/README.md` for available queries and their expected outputs.


## Model-training campaign

The trained model itself remains in MLflow; it is not copied into an ACA. After
`uploadRunToMlflow.py` successfully logs the model (and, unless disabled,
creates a registry version), it writes one immutable JSON provenance record to
`ML/surrogate_training/artifacts/mlflow_registry/<mlflow-run-id>.json`.

Each record contains:

- the clickable MLflow run URL;
- the MLflow model URI, registered-model name, and version;
- evaluation metrics;
- the original training manifest, including dataset and feature hashes,
  train/validation/test sizes, hyperparameters, source Git state, runtime, and
  artifact checksums.

`create_model_campaign.py` validates every completed record and rebuilds the
separate `rhino-model-training.aca`. Each ACA dataset is named
`mlflow-run-<run-id>` and points to its JSON provenance record. Rebuilding from
all records keeps the result deterministic and avoids duplicate ACA entries.

Preview the operation first:

```bash
python create_model_campaign.py --dry-run
```

After at least one MLflow upload has produced a record, create the ACA:

```bash
python create_model_campaign.py
```

Then run the normal publication sequence:

```bash
python publish_campaign.py --dry-run
python publish_campaign.py
python create_index.py
```

The existing publication glob includes `rhino-model-training.aca`, so it is
copied to the shared NERSC project campaign store with the date-based data ACAs.
The index also includes it, allowing the data archives and model-training
provenance to be discovered through the same `rhino.acx` index while remaining
separate ACA files.

The related settings in `campaign_spec.json` are
`MODEL_CAMPAIGN_ARCHIVE`, `MODEL_PROVENANCE_ROOT`, and
`MODEL_RUN_RECORD_DIR`.

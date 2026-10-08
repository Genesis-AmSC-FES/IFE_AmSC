# Laser Campaign Archiving, Indexing, and Publication

This component turns converted Phoenix BP5 run Series into searchable HPC Campaign artifacts.

The active workflow creates date-windowed ACA archives, builds one combined laser.acx index, and publishes ACA/ACX products to a shared project directory. TAR creation, TAR checksums, TAR member manifests, TAR indexes, and archival-replica registration are currently disabled.

## Inputs and outputs

Default BP5 input:

    /global/cfs/cdirs/m3239/2026_FES-AmSC/data/laser/bp_output

Private campaign build store:

    /global/homes/b/bhowmic/campaign-store/IFE

Shared project store:

    /global/cfs/cdirs/m3239/2026_FES-AmSC/campaigns/IFE

Combined index name:

    laser.acx

Configuration defaults live in config_campaign.sh and publication settings live in campaign_spec.json. Every setting can be overridden before execution.

## Campaign grouping

CAMPAIGN_WINDOW_DAYS defaults to 10. The first BP5 dataset starts a window; subsequent datasets dated before the start plus ten calendar days remain in that campaign. The next dataset starts a new window. All runs from a date stay together unless MAX_DATASETS_PER_ARCHIVE is set to a positive limit.

MAX_DATASETS_PER_ARCHIVE defaults to 0, meaning unlimited. When a positive maximum splits a busy date window, deterministic part suffixes are used.

ACA names record the actual covered dates:

    laser-2026-03-22_to_2026-03-25.aca

Input filenames must contain an ISO date. The shim produces run-<run_id>-<YYYY-MM-DD>.bp5, which satisfies this requirement.

## Recommended workflow

Activate the environment that provides hpc_campaign, then enter this directory:

    cd /global/homes/b/bhowmic/Projects/IFE_AmSC/Laser/AI_ready_workflow/2_campaign

Preview the complete plan without writing artifacts:

    bash create_campaign.sh --dry-run

Preview different grouping choices:

    bash create_campaign.sh --dry-run --window-days 7
    bash create_campaign.sh --dry-run --window-days 10 --max-datasets 50

Create ACA archives and the combined ACX index:

    bash create_campaign.sh

The wrapper runs create_archives.py followed by create_index.py. The --rebuild-tars option is accepted for compatibility but is reserved while TAR support is disabled.

## Separate execution

The two active build steps can also be run separately:

    python create_archives.py --data-root "$LASER_DATA_ROOT" --bp5-input-dir "$BP5_INPUT_DIR" --campaign-store "$CAMPAIGN_STORE" --window-days 10
    python create_index.py --campaign-store "$CAMPAIGN_STORE" --archive-prefix laser --index laser.acx

Using create_campaign.sh is preferred because it loads and validates the shared configuration consistently.

## Publication

Campaign construction is staged in the private campaign store. Publication is intentionally separate so incomplete builds are not exposed to the project.

Preview publication:

    python publish_campaign.py --dry-run

Publish and verify files:

    python publish_campaign.py

The publisher selects date-based laser-*.aca files and the exact laser.acx index, copies them to the shared CFS store, and verifies each copy by SHA-256. TAR payloads are not published.

## Queries

The queries/ directory contains demonstration SQL for the combined ACX index:

- count_shot_alignment.sql: counts or lists shot and alignment acquisitions.
- model_quality_filter_candidates.sql: finds shot data suitable for model-quality filtering.
- model_safety_envelope_candidates.sql: identifies shot candidates for safety-envelope analysis.

Run one query at a time with the SQLite or HPC Campaign query interface used for the ACX index. Inspect the SQL first and point it at the intended laser.acx file.

## Files

- config_campaign.sh: default paths, archive naming, grouping limits, and publication destination.
- campaign_spec.json: publication source and shared-store settings.
- create_campaign.sh: validated orchestration entry point for ACA and ACX creation.
- create_archives.py: BP5 discovery, date grouping, plan generation, and ACA creation.
- create_index.py: builds the combined SQLite-based ACX index from ACA archives and openPMD/ADIOS metadata.
- publish_campaign.py: dry-run capable publication with SHA-256 verification.
- queries/: reusable demonstration and model-selection queries.

## Important behavior

- Nothing runs automatically.
- A dry run does not create ACA, ACX, TAR, or publication artifacts.
- Existing BP5 data are read, not modified.
- ACA/ACX files are built privately before publication.
- Source and shared campaign stores must remain different.
- TAR-related configuration remains in place for a future reactivation, but TAR creation is disabled in the current scripts.

## Validation

Check shell and Python syntax:

    bash -n config_campaign.sh create_campaign.sh
    python -m py_compile create_archives.py create_index.py publish_campaign.py

Display command help:

    bash create_campaign.sh --help
    python create_archives.py --help
    python create_index.py --help
    python publish_campaign.py --help

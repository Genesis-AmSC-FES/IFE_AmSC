# UCLA Phoenix Laser Data Workflow

This directory contains the data-engineering workflow for UCLA Phoenix laser-facility data used by the 2026 FES-AmSC project. It converts raw facility HDF5 acquisitions into openPMD/ADIOS2 BP5 Series, organizes those Series into searchable HPC Campaign archives, and publishes the campaign metadata for project-wide use.

## Data flow

    Phoenix HDF5 files
            |
            v
    AI_ready_workflow/1_shim
    openPMD schema + ADIOS2 BP5 serialization
            |
            v
    laser/bp_output/run-<run_id>-<date>.bp5
            |
            v
    AI_ready_workflow/2_campaign
    date-windowed ACA archives + combined ACX index
            |
            v
    shared project campaign store

The workflow treats one logical Phoenix run as one openPMD Series. If the acquisition recorder rolls a run into another preallocated HDF5 file, the files are ordered by event time and stacked as continuous openPMD iterations.

## Repository layout

- AI_ready_workflow/: production workflow and its stage-level documentation.
- AI_ready_workflow/1_shim/: HDF5 discovery, normalization, openPMD schema mapping, and BP5 writing.
- AI_ready_workflow/2_campaign/: ACA creation, ACX indexing, example queries, and publication.
- demo/: exploratory or demonstration material.
- tests/: integration and demonstration helpers.

Start with [AI_ready_workflow/README.md](AI_ready_workflow/README.md) for the end-to-end procedure.

## Default NERSC locations

- Raw and converted laser data: /global/cfs/cdirs/m3239/2026_FES-AmSC/data/laser
- Converted BP5 Series: /global/cfs/cdirs/m3239/2026_FES-AmSC/data/laser/bp_output
- Private campaign build store: /global/homes/b/bhowmic/campaign-store/IFE
- Shared campaign store: /global/cfs/cdirs/m3239/2026_FES-AmSC/campaigns/IFE

Most paths can be overridden with environment variables documented in the stage READMEs.

## Quick start

Convert the raw HDF5 inventory on a compute node:

    cd /global/homes/b/bhowmic/Projects/IFE_AmSC/Laser/AI_ready_workflow/1_shim
    sbatch convert_laser.slurm

Preview, create, and publish campaigns after conversion completes:

    cd /global/homes/b/bhowmic/Projects/IFE_AmSC/Laser/AI_ready_workflow/2_campaign
    bash create_campaign.sh --dry-run
    bash create_campaign.sh
    python publish_campaign.py --dry-run
    python publish_campaign.py

Nothing in this directory runs automatically. Review the dry-run output and job logs before publishing campaign artifacts.

## Data contract

- Input unit: one or more HDF5 files belonging to a Phoenix run.
- BP5 unit: one openPMD Series per run number and date.
- Iteration unit: one recorded event.
- Event provenance: sourceFileIndex and sourceEventIndex preserve the original HDF5 location.
- Data records: available scalar, waveform, spectrum, coordinate, and image quantities are discovered per run and written as openPMD meshes.
- Campaign unit: a chronological window of BP5 run Series, ten days by default.

## Environment

The current workflow is designed for NERSC Perlmutter and the IFE_AmSC Conda environment. The shim requires h5py, NumPy, openPMD-api, and ADIOS2. Campaign creation additionally requires hpc_campaign.

## Operational notes

- Existing BP5 outputs are skipped by the Slurm workflow. Use a clean output directory when schema changes require regeneration.
- Acquisition mode, such as shot, alignment, or target, is stored as metadata rather than encoded in output filenames.
- Hash-based BP5 names are not used.
- TAR campaign replicas are currently disabled; the active campaign products are ACA archives and the ACX index.

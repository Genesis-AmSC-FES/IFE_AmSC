# AI-Ready Laser Workflow

This directory contains the two production stages that turn UCLA Phoenix laser acquisitions into discoverable, model-ready data products.

## Stages

### 1. HDF5 to openPMD/BP5

Directory: [1_shim](1_shim/README.md)

The shim discovers supported Phoenix HDF5 files, groups rollover files belonging to the same run, normalizes heterogeneous PV and image layouts, and writes one openPMD/ADIOS2 BP5 Series per logical run.

Input:

    /global/cfs/cdirs/m3239/2026_FES-AmSC/data/laser/<date>/*.h5
    /global/cfs/cdirs/m3239/2026_FES-AmSC/data/laser/target_data/*.h5

Output:

    /global/cfs/cdirs/m3239/2026_FES-AmSC/data/laser/bp_output/run-<run_id>-<YYYY-MM-DD>.bp5

### 2. Campaign archiving and indexing

Directory: [2_campaign](2_campaign/README.md)

This stage groups BP5 Series into chronological campaign windows, creates ACA campaign archives, builds the combined laser.acx index, and publishes selected ACA/ACX products to the shared project store.

Private build output:

    /global/homes/b/bhowmic/campaign-store/IFE

Shared publication output:

    /global/cfs/cdirs/m3239/2026_FES-AmSC/campaigns/IFE

## End-to-end execution

1. Submit the conversion job.

       cd /global/homes/b/bhowmic/Projects/IFE_AmSC/Laser/AI_ready_workflow/1_shim
       sbatch convert_laser.slurm

2. Inspect the Slurm output and verify BP5 counts.

       python count_outputs.py /global/cfs/cdirs/m3239/2026_FES-AmSC/data/laser/bp_output

3. Preview the campaign plan.

       cd ../2_campaign
       bash create_campaign.sh --dry-run

4. Create ACA archives and the ACX index.

       bash create_campaign.sh

5. Preview and perform publication.

       python publish_campaign.py --dry-run
       python publish_campaign.py

## Stage boundary

The contract between the two stages is the BP5 directory. Campaign code does not interpret raw HDF5 files, and the shim does not create or publish campaigns. This separation makes it possible to regenerate campaign metadata without reconverting source data, or to regenerate BP5 files after a schema change before rebuilding campaigns.

## Reruns

- Shim reruns use --skip-existing in the supplied Slurm job.
- Campaign dry runs do not write artifacts.
- Campaign creation uses the configured private store.
- Publication is a separate explicit step and verifies copied files by SHA-256.

When the BP5 schema or run-grouping logic changes, regenerate affected BP5 outputs before rebuilding ACA and ACX products. Existing outputs will otherwise be skipped.

# Phoenix HDF5 to openPMD/BP5 Shim

This component converts UCLA Phoenix laser-facility HDF5 data into run-oriented openPMD Series stored with the ADIOS2 BP5 engine.

## Responsibilities

The shim:

1. discovers production HDF5 inputs from immediate date-named directories and target_data;
2. excludes documentation, campaign, MLrun, older-data, cache, and BP5-output trees;
3. identifies files by recorded run number and run date;
4. orders rollover segments by their first valid epoch;
5. normalizes scalar PVs, traces, coordinates, images, timestamps, and source metadata;
6. discovers only variables present in each run;
7. maps quantities into the agreed Phoenix openPMD mesh namespaces; and
8. writes one BP5 Series per logical run.

## Run and iteration model

- One logical run becomes one openPMD Series.
- One recorded event becomes one openPMD iteration.
- Multiple HDF5 files with the same run number and date are treated as rollover segments and stacked in event-time order.
- eventIndex is continuous across all segments.
- sourceFileIndex identifies the contributing HDF5 file.
- sourceEventIndex preserves the original HDF5 row.
- shotNumber remains an iteration attribute and is not used as the iteration key.

The output filename is:

    run-<run_id>-<YYYY-MM-DD>.bp5

No hash or acquisition-mode suffix is used. Shot, alignment, and target classification is stored in metadata.

## Default paths

- Input root: /global/cfs/cdirs/m3239/2026_FES-AmSC/data/laser
- Output directory: /global/cfs/cdirs/m3239/2026_FES-AmSC/data/laser/bp_output
- Documentation directory: <input root>/documentation

Overrides:

- LASER_ROOT changes the source root.
- LASER_BP_OUT_DIR changes the BP5 destination.
- LASER_DOCUMENTATION_DIR changes the documentation/PV metadata source.
- LASER_SHIM_DIR tells the Slurm wrapper where this component is installed.
- LASER_CONDA_ENV changes the Conda environment used by the batch job.

## Batch conversion

From this directory:

    sbatch convert_laser.slurm

The Slurm wrapper invokes laserWrite_multi.py once on the complete source root so files belonging to the same run can be grouped. Existing outputs are skipped. Conversion failures are reported while unrelated runs continue.

To use alternate locations:

    sbatch --export=ALL,LASER_DATA_ROOT=/path/to/laser,LASER_BP_OUT_DIR=/path/to/bp_output,LASER_CONDA_ENV=IFE_AmSC convert_laser.slurm

## Interactive conversion

For a small test or controlled rerun on an appropriate node:

    python laserWrite_multi.py /path/to/laser       --out-dir /path/to/bp_output       --skip-existing

Useful options:

- --recursive: recursively search approved date and target_data roots.
- --strict: fail on optional shape or classification problems.
- --stop-on-error: stop immediately instead of continuing with other runs.
- --no-overwrite: fail if an output already exists.
- --skip-existing: leave existing outputs untouched.

## openPMD organization

Series-level attributes contain standard openPMD metadata plus colon-namespaced Phoenix metadata for provenance, facility, system settings, experiment settings, and run information.

Each iteration contains time, dt, eventIndex, sourceEventIndex, sourceFileIndex, sourceFileName, shotNumber when available, recorderEpoch, eventAcquisitionDuration, and fullSystemShot.

Event-varying data are written as openPMD meshes. The logical hierarchy is retained in each mesh schemaPath attribute, while physical record names use underscore-safe names. Supported namespaces include laser, timing, digitizer, oscilloscope, camera, thermal, environment, vacuum, vibration, target, diagnostic, scan, and unmapped.

Variables are discovered independently for each run; a run is not required to contain every known quantity.

## Provenance for rollover runs

Every contributing source file receives an indexed provenance entry with its name, URI, file role, segment index, allocated and recorded event counts, HDF5 root attributes, PV manifest, and image manifest. Aggregated run metadata includes sourceFileCount, rolloverSegmentCount, recordedEventCount, validEventCount, and unusedAllocatedEventCount.

## Files

- laserWrite_multi.py: production discovery, run grouping, segment ordering, and batch orchestration.
- laserWrite.py: loads one or more source segments into normalized models.
- metadata.py: parses heterogeneous Phoenix HDF5 structures and constructs metadata.
- variable_discovery.py: catalogs present variables and validates mesh-name collisions.
- writer.py: writes Series metadata, continuous iterations, and mesh records.
- utils.py: shared naming, HDF5, metadata, and openPMD helpers.
- convert_laser.slurm: NERSC batch entry point.
- count_outputs.py: counts BP5 run Series and recorded events.
- compression_sweep.py and compression/: compression experiments, not part of normal conversion.

## Validation and inspection

Check source syntax:

    python -m py_compile laserWrite_multi.py laserWrite.py writer.py utils.py metadata.py variable_discovery.py count_outputs.py
    bash -n convert_laser.slurm

Count completed Series and events:

    python count_outputs.py /global/cfs/cdirs/m3239/2026_FES-AmSC/data/laser/bp_output

Inspect a BP5 Series:

    bpls -la /path/to/run-<run_id>-<date>.bp5 | less

ADIOS2 StatsLevel 1 is enabled, so bpls can expose stored per-variable statistics such as minima and maxima.

## Rerun caution

The supplied Slurm job skips existing outputs. After a schema or rollover-grouping change, use a clean output directory or intentionally remove only the affected BP5 Series before resubmitting. The converter does not modify raw HDF5 inputs.

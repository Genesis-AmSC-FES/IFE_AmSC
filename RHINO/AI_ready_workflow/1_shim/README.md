# RHINO Shim Layer

This layer converts raw RHINO simulation outputs into openPMD-compliant
ADIOS2 BP5 series. The resulting files are consumed by the indexing, feature
extraction, and downstream analysis stages of the AI-ready workflow.

## Responsibilities

- Read the pickle files belonging to one RHINO run.
- Discover and validate every available registered species.
- Store time-dependent and steady-state inventories.
- Preserve subsystem definitions and directed subsystem connectivity.
- Record simulation, parameter-scan, provenance, input, and output metadata.
- Calculate the Tritium 2% steady-state time.
- Produce a BP5 file that passes strict openPMD extent validation.

## Input files

For a run identified by `PREFIX` and `INFIX`, the converter requires these
run-level files:

```text
<PREFIX>_IFE_input.pkl
<PREFIX>_IFE_meta.pkl
<PREFIX>_IFE_processed.pkl
```

For each species present in the input table, it also requires:

```text
<PREFIX>_<INFIX>_<TOKEN>_SteadyState.pkl

and one of:

<PREFIX>_<INFIX>_<TOKEN>_reduced.pkl    # preferred
<PREFIX>_<INFIX>_<TOKEN>.pkl            # fallback
```

The currently registered species are:

| Species | Input table | File token |
| --- | --- | --- |
| Tritium | `Systems_T` | `T` |
| Deuterium | `Systems_D` | `D` |

A species is omitted when its input table is absent or contains no subsystem
rows. If the input table is present, missing time-series or steady-state files
are treated as an incomplete run and stop the conversion. Additional species
can be supported by adding an entry to `SPECIES_CONFIG` in `rhinoWrite.py`.

## Convert one run

Run the AI-ready converter from the repository root:

```bash
PREFIX=14-24-04
INFIX=IFE_AmSC_500MW_FuelCycle
DATA_PATH="/global/cfs/cdirs/m3239/2026_FES-AmSC/data/rhino/Surrogate Data/2026-04-29"
OUTPUT_PATH="$HOME/my.bp5"

python AI_ready_workflow/1_shim/rhinoWrite.py \
    --data-path "$DATA_PATH" \
    --prefix "$PREFIX" \
    --infix "$INFIX" \
    --output-path "$OUTPUT_PATH"
```

`DATA_PATH` must end with a scenario date in `YYYY-MM-DD` form, and `PREFIX`
must identify the run time in `HH-MM-SS` form. The example above produces:

```text
provenance:simulationDate = 2026-04-29T14:24:04
```

The source name does not include a timezone, so this value has no UTC offset.
The openPMD-managed series attribute `date` is separate and records when the
BP5 conversion was performed.

## Output representation

Each run produces one BP5 data product. The main scientific layout is:

```text
data/<iteration>/inventory/
├── Times/
│   └── data/SCALAR                     # [Nt] time coordinates
│
└── <species>/
    ├── subsystem:*                     # [Ns] aligned subsystem definitions
    ├── steadyState:mass                # [Ns] steady-state inventory
    ├── connectivity:*                  # [Ne] directed edge list
    └── mass/SCALAR                     # [Ns, Nt] time-dependent inventory
```

For `mass[i, j]`, subsystem row `i` is identified by `subsystem:name[i]`, and
time column `j` is identified by `Times/data[j]`. Inventory values are stored
in grams with `unitSI = 1e-3`, which converts them to SI kilograms.

Subsystem connectivity is stored on each species as a directed edge list:

```text
connectivity:representation = "directed_edge_list"
connectivity:sourceId
connectivity:targetId
connectivity:fractionalInflow
connectivity:fractionalInflowDefined
```

Undefined RHINO fractional inflows are retained as `NaN`; the aligned
`fractionalInflowDefined` value is `0`. Defined values have a flag of `1`.

The only dataset record inside an inventory species is `mass`. Subsystem
definitions, steady-state inventory, and connectivity are attributes because
their lengths differ from the `[Ns, Nt]` mass extent. This layout avoids the
inconsistent particle-record extents produced by the previous
`mass_steady` and `subsystems/<name>` records.

See [`schema.md`](schema.md) for the complete emitted schema, dimensions,
units, and alignment rules.

## Metadata

Application metadata is organized with namespaced openPMD attributes:

```text
software:*       RHINO software identity and version information
provenance:*     authorship, conversion time, and source file lineage
system:*         source-system description
simulation:*     numerical and physical settings for this run
parameterScan:*  parameter-scan definition and changes
species:*        available species
input:*          RHINO simulation inputs
output:*         RHINO postprocessed outputs and steady-state time
```

The original `Sweep Changes_0` through `Sweep Changes_N` metafile rows are
stored as the ordered arrays `parameterScan:changeIndices` and
`parameterScan:changes`, rather than as hundreds of flat attributes.

## Steady-state time

The converter retains the existing Tritium 2% steady-state calculation:

1. Compare each subsystem's time-dependent inventory with its steady-state
   inventory.
2. Use relative error when the steady-state value is nonzero and absolute
   error when it is close to zero.
3. Find the first time after which the inventory remains continuously within
   the 2% tolerance.
4. Exclude subsystem names containing `box`, `storage`, or `delivery`.
5. Store the maximum time across the remaining subsystems as:

   ```text
   output:Steady state time (days)
   ```

Per-subsystem steady-state times are calculated internally but are not yet
serialized. Steady-state inventories are stored for every available species,
but the steady-state-time calculation currently applies only to Tritium.

## Convert multiple runs

`rhinoWrite_multiple.py` converts one or more scenario directories and
continues after reporting an incomplete or invalid run.

Run it from the repository root:

```bash
OUTPUT_ROOT="$HOME/rhino_bp5"
ROOT_PATH="/global/cfs/cdirs/m3239/2026_FES-AmSC/data/rhino/Surrogate Data"

python -m AI_ready_workflow.1_shim.rhinoWrite_multiple \
    --root-path "$ROOT_PATH" \
    --scenarios 2026-04-29 2026-04-30 \
    --output-root "$OUTPUT_ROOT"
```

To omit a known run, provide `--skip-run SCENARIO:PREFIX` one or more times:

```bash
python -m AI_ready_workflow.1_shim.rhinoWrite_multiple \
    --root-path "$ROOT_PATH" \
    --scenarios 2026-04-29 \
    --output-root "$OUTPUT_ROOT" \
    --skip-run 2026-04-29:14-24-04
```

The current batch converter discovers runs using `*_T_reduced.pkl`. It does
not discover Deuterium-only runs, full-Tritium-only runs, or runs without a
reduced Tritium file. Once a run has been discovered, the single-run converter
still loads every available registered species.

Review batch output for `ERROR processing run` before treating the conversion
as complete.

## Interactive example

`examples/rhinoWrite_multiple.ipynb` provides an interactive multi-run
conversion example:

```bash
jupyter lab AI_ready_workflow/1_shim/examples/rhinoWrite_multiple.ipynb
```

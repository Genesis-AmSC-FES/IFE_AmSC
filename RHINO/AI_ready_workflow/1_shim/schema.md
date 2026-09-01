# RHINO openPMD/ADIOS2 Schema

This document describes the BP5 schema currently emitted by
`AI_ready_workflow/1_shim/rhinoWrite.py`.

## Conventions

- Names containing `:` are openPMD attributes that use logical namespaces;
  they are not additional physical directories.
- `<species>` is every species present in the RHINO input and registered in
  `SPECIES_CONFIG` (currently `Tritium` and optional `Deuterium`).
- `Ns` is the number of subsystems for a species.
- `Nt` is the number of stored time points.
- `Ne` is the number of directed subsystem-connectivity edges.
- `Nc` is the number of parameter-scan changes.
- Attribute arrays on a species are aligned with `subsystem:name` unless a
  different alignment is stated explicitly.
- openPMD may return a one-element attribute array as a scalar on read-back.

## Schema outline

```text
/  openPMD Series
│
├── openPMD                              # [str] openPMD standard version
├── openPMDextension                     # [int] openPMD extension bit field
├── basePath                             # [str] "/data"
├── particlesPath                        # [str] "inventory/"
├── iterationEncoding                    # [str] "variableBased"
├── iterationFormat                      # [str] managed by openPMD
├── date                                 # [str] managed by openPMD
├── software                             # [str] "openPMD-api", managed by openPMD
├── softwareVersion                      # [str] openPMD-api version
│
├── software:softwareName                # [str] "RHINO"
├── software:softwareDescription         # [str]
├── software:softwareVersion             # [str] RHINO schema producer version
├── software:versionControlSoftware      # [str]
├── software:softwareCommit              # [str]
├── software:softwareDocumentation       # [str]
│
├── provenance:author                    # [str]
├── provenance:authorAffiliation         # [str]
├── provenance:authorEmail               # [str]
├── provenance:creationDate              # [str] ISO-8601 UTC conversion time
├── provenance:creationTimeUTC           # [str] ISO-8601 UTC conversion time
├── provenance:simulationDate            # [str] derived from scenario and run prefix
├── provenance:inputDirectory            # [str] directory read by converter
├── provenance:inputFiles                # [list[str]] pickle files consumed
├── provenance:originalDataDirectory     # [str] meta["main_path"]
├── provenance:originalDataFiles         # [list[str]] original RHINO JSON
├── provenance:originalInputDirectory    # [str] meta["main_path"]
├── provenance:originalInputFiles        # [list[str]] original RHINO JSON
├── provenance:originalInputFile         # [str] meta["input_file"]
│
├── system:systemIP                      # [str] currently empty
├── system:systemDescription             # [str] currently empty
├── system:comment                       # [str] currently empty
│
├── simulation:timeStep                  # [float] meta["dt"]
├── simulation:calculationLength         # [float] meta["calc_length"]
├── simulation:timeUnits                 # [str] meta["units"]
├── simulation:computeNodes              # [int] meta["nodes"]
├── simulation:fusionType                # [str] meta["fusion_type"]
├── simulation:multiplier                # [float] meta["multiplier"]
├── simulation:gasStreams                # [list[str]] meta["gas_streams"]
│
├── parameterScan:description            # [str] meta["runs_type"]
├── parameterScan:numberOfInputFiles     # [int] meta["num_input_files"]
├── parameterScan:variables              # [list[str]] meta["what"]
├── parameterScan:start                  # [list[number]] meta["start"]
├── parameterScan:stop                   # [number/list[number]] meta["stop"]
├── parameterScan:increment              # [number/list[number]] meta["increment"]
├── parameterScan:end                    # [number/list[number]] meta["end"]
├── parameterScan:index                  # [number/list[number]] meta["index"]
├── parameterScan:selectedSubsystems     # [str/list[str]] meta["subsystem"]
├── parameterScan:selectedSpecies        # [str/list[str]] meta["species"]
├── parameterScan:changeIndices          # [Nc integers]
├── parameterScan:changes                # [Nc strings], ordered Sweep Changes_*
├── parameterScan:numberOfChanges        # [int] Nc
│
├── species:names                        # [list[str]] available species
│
├── input:TBR:Tritium Breeding Ratio     # [float]
├── input:TBRr:Required Tritium Breeding Ratio
├── input:beta:Burn fraction             # [float]
├── input:eta:Fueling efficiency         # [float]
├── input:Ndotminus:Tritium burned per day
├── input:MW:Power output in MW          # [float]
├── input:I0_SD:Starting inventory       # [float]
│
├── output:I0 (g)                        # [float]
├── output:Imin (g)                      # [float]
├── output:I_startup (g)                 # [float]
├── output:I_subtract (g)                # [float]
├── output:reserve_time (days)           # [number]
├── output:Iops (g)                      # [float]
├── output:plant_doubling_time (days)    # [float]
├── output:Steady state time (days)      # [float/NaN], Tritium 2% calculation
│
└── data/<iteration>/
    ├── time                              # [float] 0.0
    ├── dt                                # [float] simulation timestep
    ├── timeUnitSI                        # [float] 86400 seconds/day
    │
    └── inventory/
        ├── Times/
        │   ├── description               # [str] "Times"
        │   └── data/SCALAR               # [Nt floats], time coordinates
        │       ├── unitDimension         # T^1
        │       └── unitSI                # 86400 seconds/day
        │
        └── <species>/
            ├── description               # [str]
            ├── subsystemsAxis            # [int] 0
            ├── timeAxis                  # [int] 1
            │
            ├── subsystem:count           # [int] Ns
            ├── subsystem:id              # [Ns integers], mass-axis-0 order
            ├── subsystem:name            # [Ns strings], mass-axis-0 order
            ├── subsystem:label           # [Ns strings], mass-axis-0 order
            ├── subsystem:processingTime  # [Ns floats]
            ├── subsystem:processingTimeUnit
            │                              # [str], currently "days"
            ├── subsystem:processingTimeUnitSI
            │                              # [float], currently 86400
            ├── subsystem:nonradioactiveLossFraction
            │                              # [Ns floats]
            ├── subsystem:initialMass      # [Ns floats]
            ├── subsystem:initialMassUnitSI
            │                              # [float] 1e-3 kg per stored gram
            ├── subsystem:source           # [Ns floats], RHINO source terms
            │
            ├── steadyState:mass           # [Ns floats], subsystem-aligned
            ├── steadyState:massUnitSI     # [float] 1e-3 kg per stored gram
            │
            ├── connectivity:representation
            │                              # [str] "directed_edge_list"
            ├── connectivity:sourceId      # [Ne integers]
            ├── connectivity:targetId      # [Ne integers]
            ├── connectivity:fractionalInflow
            │                              # [Ne floats], NaN if undefined
            ├── connectivity:fractionalInflowDefined
            │                              # [Ne uint8], 1 defined / 0 undefined
            │
            └── mass/SCALAR                # [Ns, Nt] time-dependent inventory
                ├── unitDimension          # M^1
                └── unitSI                 # 1e-3 kg per stored gram
```

## Alignment rules

For subsystem row `i` and time index `j`:

```text
mass[i, j]
    subsystem = subsystem:name[i]
    subsystem ID = subsystem:id[i]
    time = Times/data[j]

steadyState:mass[i]
    subsystem = subsystem:name[i]
```

Connectivity arrays are edge-aligned rather than subsystem-aligned. For edge
index `e`:

```text
source subsystem ID = connectivity:sourceId[e]
target subsystem ID = connectivity:targetId[e]
fraction = connectivity:fractionalInflow[e]
is defined = connectivity:fractionalInflowDefined[e]
```

## Steady-state time

`output:Steady state time (days)` is currently calculated for Tritium only.
For each included subsystem, the converter finds the first time after which
the inventory remains within 2% of its steady-state inventory. Relative error
is used for nonzero steady-state values and absolute error for values close to
zero. Subsystem names containing `box`, `storage`, or `delivery` are excluded.
The stored result is the maximum time across the remaining subsystems.

Per-subsystem steady-state times are calculated internally but are not yet
serialized.

## Current representation notes

- `mass` is the only dataset record within each inventory species. This keeps
  particle-record extents homogeneous for strict openPMD readers.
- Subsystem definitions, steady-state inventories, and connectivity are
  species attributes because their lengths differ from the `[Ns, Nt]` mass
  record.
- The former `mass_steady` and `subsystems/<name>` records are not part of this
  schema.

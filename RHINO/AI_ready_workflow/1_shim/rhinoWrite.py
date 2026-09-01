"""
RHINO → openPMD/ADIOS2 shim layer

This file reads RHINO data (pkl files) and writes it into an openPMD series using ADIOS2 as a backend (.bp5)
Outputs:
    - one BP5 series per RHINO run
    - input:* metadata attributes
    - output:* metadata attributes
    - subsystem inventories
    - directed subsystem connectivity for each available species
    - steady-state values
    - time-series records
"""

import ast
import sys
import pickle 
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import openpmd_api as io


SPECIES_CONFIG = {
    "Tritium": {
        "system_key": "Systems_T",
        "file_token": "T",
    },
    "Deuterium": {
        "system_key": "Systems_D",
        "file_token": "D",
    },
}


def simulation_datetime_from_source(data_path, prefix):
    """Return the run datetime encoded by its scenario directory and prefix."""
    scenario_date = Path(data_path).name
    try:
        return datetime.strptime(
            f"{scenario_date} {prefix}",
            "%Y-%m-%d %H-%M-%S",
        )
    except ValueError as exc:
        raise ValueError(
            "Cannot derive the simulation datetime: expected DATA_PATH to end "
            "in YYYY-MM-DD and PREFIX to use HH-MM-SS, but received "
            f"DATA_PATH={data_path!r} and PREFIX={prefix!r}"
        ) from exc


def _is_missing_connection_value(value):
    """Return whether a RHINO connection value represents missing information."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() == "none"
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    return False


def _connection_values_as_list(value):
    """Normalize a scalar or sequence-valued RHINO connection field to a list."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def build_connectivity_graph(subsystems):
    """Build a validated directed edge list from RHINO subsystem metadata.

    RHINO stores connectivity on each destination subsystem. ``injectors`` is
    the ordered list of upstream subsystem IDs and ``fractional inflows`` is
    the aligned list of optional edge properties. Undefined fractional inflows
    are retained as NaN and identified by ``fractional_inflow_defined``.
    """
    subsystem_names_by_id = {}
    for subsystem_name, metadata in subsystems.items():
        subsystem_id = int(metadata["id"])
        if subsystem_id in subsystem_names_by_id:
            other_name = subsystem_names_by_id[subsystem_id]
            raise ValueError(
                f"Duplicate subsystem id {subsystem_id} for "
                f"{other_name!r} and {subsystem_name!r}"
            )
        subsystem_names_by_id[subsystem_id] = subsystem_name

    source_ids = []
    target_ids = []
    fractional_inflows = []
    fractional_inflows_defined = []

    for target_name, metadata in subsystems.items():
        target_id = int(metadata["id"])
        injectors = [
            injector
            for injector in _connection_values_as_list(metadata.get("injectors"))
            if not _is_missing_connection_value(injector)
        ]
        if not injectors:
            continue

        raw_fractional_inflows = _connection_values_as_list(
            metadata.get("fractional inflows")
        )
        if all(
            _is_missing_connection_value(value)
            for value in raw_fractional_inflows
        ):
            edge_fractional_inflows = [np.nan] * len(injectors)
        elif len(raw_fractional_inflows) != len(injectors):
            raise ValueError(
                f"Subsystem {target_name!r} (id {target_id}) has "
                f"{len(injectors)} injectors but "
                f"{len(raw_fractional_inflows)} "
                "fractional inflows"
            )
        else:
            edge_fractional_inflows = raw_fractional_inflows

        for injector, fractional_inflow in zip(
            injectors, edge_fractional_inflows
        ):
            try:
                source_id = int(injector)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Subsystem {target_name!r} (id {target_id}) has invalid "
                    f"injector id {injector!r}"
                ) from exc
            if source_id not in subsystem_names_by_id:
                raise ValueError(
                    f"Subsystem {target_name!r} (id {target_id}) references "
                    f"unknown injector id {source_id}"
                )

            fractional_inflow_defined = not _is_missing_connection_value(
                fractional_inflow
            )
            try:
                numeric_fractional_inflow = (
                    float(fractional_inflow)
                    if fractional_inflow_defined
                    else np.nan
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Connection {source_id} -> {target_id} has invalid "
                    f"fractional inflow {fractional_inflow!r}"
                ) from exc

            source_ids.append(source_id)
            target_ids.append(target_id)
            fractional_inflows.append(numeric_fractional_inflow)
            fractional_inflows_defined.append(fractional_inflow_defined)

    return {
        "source_id": np.ascontiguousarray(source_ids, dtype=np.int64),
        "target_id": np.ascontiguousarray(target_ids, dtype=np.int64),
        "fractional_inflow": np.ascontiguousarray(
            fractional_inflows, dtype=np.float64
        ),
        "fractional_inflow_defined": np.ascontiguousarray(
            fractional_inflows_defined, dtype=np.uint8
        ),
    }


def write_connectivity_graph(particle_species, graph):
    """Write a directed edge list as attributes of its inventory species."""
    particle_species.set_attribute(
        "connectivity:representation", "directed_edge_list"
    )
    particle_species.set_attribute(
        "connectivity:sourceId", graph["source_id"].tolist()
    )
    particle_species.set_attribute(
        "connectivity:targetId", graph["target_id"].tolist()
    )
    particle_species.set_attribute(
        "connectivity:fractionalInflow",
        graph["fractional_inflow"].tolist(),
    )
    particle_species.set_attribute(
        "connectivity:fractionalInflowDefined",
        graph["fractional_inflow_defined"].tolist(),
    )


def write_subsystem_metadata(
    particle_species,
    subsystems,
    steady_state_mass,
    time_unit,
    time_unit_si,
):
    """Write row-aligned subsystem and steady-state values as attributes.

    Particle records within one openPMD species must have homogeneous
    extents. These arrays describe subsystem rows rather than the full
    subsystem-by-time inventory, so they are species attributes instead of
    independent particle records. Their order matches axis 0 of ``mass``.
    """
    subsystem_names = list(subsystems)
    steady_state_mass = np.asarray(steady_state_mass)
    if steady_state_mass.ndim != 1:
        raise ValueError(
            "Steady-state mass must be one-dimensional; received shape "
            f"{steady_state_mass.shape}"
        )
    if len(subsystem_names) != steady_state_mass.size:
        raise ValueError(
            f"Found {len(subsystem_names)} subsystem definitions but "
            f"{steady_state_mass.size} steady-state mass values"
        )

    particle_species.set_attribute("subsystem:count", len(subsystem_names))
    particle_species.set_attribute(
        "subsystem:id",
        [int(subsystems[name]["id"]) for name in subsystem_names],
    )
    particle_species.set_attribute("subsystem:name", subsystem_names)
    particle_species.set_attribute(
        "subsystem:label",
        [str(subsystems[name]["label"]) for name in subsystem_names],
    )
    particle_species.set_attribute(
        "subsystem:processingTime",
        [
            float(subsystems[name]["processing time"])
            for name in subsystem_names
        ],
    )
    particle_species.set_attribute(
        "subsystem:processingTimeUnit", str(time_unit)
    )
    particle_species.set_attribute(
        "subsystem:processingTimeUnitSI", float(time_unit_si)
    )
    particle_species.set_attribute(
        "subsystem:nonradioactiveLossFraction",
        [
            float(subsystems[name]["nonradioactive loss fraction"])
            for name in subsystem_names
        ],
    )
    particle_species.set_attribute(
        "subsystem:initialMass",
        [float(subsystems[name]["initial mass"]) for name in subsystem_names],
    )
    particle_species.set_attribute("subsystem:initialMassUnitSI", 1e-3)
    particle_species.set_attribute(
        "subsystem:source",
        [float(subsystems[name]["source"]) for name in subsystem_names],
    )

    particle_species.set_attribute(
        "steadyState:mass", steady_state_mass.astype(float).tolist()
    )
    particle_species.set_attribute("steadyState:massUnitSI", 1e-3)


def extract_species_subsystems(input_file, system_key):
    """Extract subsystem metadata for one species from a RHINO input table."""
    if system_key not in input_file:
        return {}

    subsystems = {}
    for subsystem_id, value in input_file[system_key].items():
        if not isinstance(value, list):
            continue
        if len(value) < 8:
            raise ValueError(
                f"{system_key} subsystem {subsystem_id!r} has {len(value)} "
                "fields; expected at least 8"
            )
        subsystem_name = value[0]
        if subsystem_name in subsystems:
            raise ValueError(
                f"{system_key} contains duplicate subsystem name "
                f"{subsystem_name!r}"
            )
        subsystems[subsystem_name] = {
            "id": int(subsystem_id),
            "processing time": value[1],
            "nonradioactive loss fraction": value[2],
            "fractional inflows": value[3],
            "initial mass": value[4],
            "source": value[5],
            "injectors": value[6],
            "label": value[7],
        }
    return subsystems


def _find_time_series_path(data_path, prefix, infix, file_token):
    """Find a species time series, preferring the reduced RHINO product."""
    candidates = [
        data_path / f"{prefix}_{infix}_{file_token}_reduced.pkl",
        data_path / f"{prefix}_{infix}_{file_token}.pkl",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    candidate_list = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Missing {file_token} time-series pickle; expected one of: "
        f"{candidate_list}"
    )


def _validate_subsystem_order(dataframe, subsystem_names, source_path):
    """Ensure inventory row order matches the subsystem metadata order."""
    inventory_names = [str(name) for name in dataframe.index]
    if inventory_names != subsystem_names:
        raise ValueError(
            f"Subsystem order in {source_path} does not match the RHINO "
            "input subsystem order"
        )


def load_available_species(
    data_path,
    prefix,
    infix,
    input_file,
    species_config=None,
):
    """Load every configured species that is present in a RHINO input table.

    A species is absent when its configured system column is missing or has no
    subsystem rows. Once subsystem rows exist, both time-series and
    steady-state files are required so incomplete species are not silently
    omitted.
    """
    if species_config is None:
        species_config = SPECIES_CONFIG

    data_path = Path(data_path)
    species_inputs = {}
    species_inventory = {}

    for species_name, config in species_config.items():
        system_key = config["system_key"]
        file_token = config["file_token"]
        subsystems = extract_species_subsystems(input_file, system_key)
        if not subsystems:
            continue

        time_series_path = _find_time_series_path(
            data_path, prefix, infix, file_token
        )
        steady_state_path = (
            data_path
            / f"{prefix}_{infix}_{file_token}_SteadyState.pkl"
        )
        if not steady_state_path.is_file():
            raise FileNotFoundError(
                f"Missing {species_name} steady-state pickle: "
                f"{steady_state_path}"
            )

        time_series_df = pd.read_pickle(time_series_path)
        steady_state_df = pd.read_pickle(steady_state_path)
        subsystem_names = list(subsystems)
        _validate_subsystem_order(
            time_series_df, subsystem_names, time_series_path
        )
        _validate_subsystem_order(
            steady_state_df, subsystem_names, steady_state_path
        )

        steady_state_column = f"{file_token}_Steady_State_Inventories"
        if steady_state_column not in steady_state_df.columns:
            raise ValueError(
                f"Missing column {steady_state_column!r} in "
                f"{steady_state_path}"
            )

        species_inputs[species_name] = subsystems
        species_inventory[species_name] = {
            "data_ts": np.ascontiguousarray(
                time_series_df.to_numpy(dtype=np.float64)
            ),
            "data_ss": steady_state_df[steady_state_column].to_numpy(
                dtype=np.float64
            ),
            "source_files": [time_series_path, steady_state_path],
        }

    if not species_inputs:
        configured_keys = ", ".join(
            config["system_key"] for config in species_config.values()
        )
        raise ValueError(
            "No configured species were found in the RHINO input table; "
            f"looked for: {configured_keys}"
        )

    timestep_counts = {
        species_name: inventory["data_ts"].shape[1]
        for species_name, inventory in species_inventory.items()
    }
    if len(set(timestep_counts.values())) != 1:
        raise ValueError(
            "Species time-series lengths do not match: "
            f"{timestep_counts}"
        )

    return species_inputs, species_inventory


def _relative_input_files(input_directory, input_paths):
    """Return unique input paths relative to their shared input directory."""
    input_directory = Path(input_directory).resolve()
    relative_paths = []
    for input_path in input_paths:
        input_path = Path(input_path).absolute()
        try:
            relative_path = input_path.relative_to(input_directory)
        except ValueError:
            relative_path = input_path
        relative_path = str(relative_path)
        if relative_path not in relative_paths:
            relative_paths.append(relative_path)
    return relative_paths


def normalize_metafile_value(value):
    """Convert a RHINO metafile value into an openPMD-friendly value.

    RHINO stores both arrays such as ``"['Ndotminus', 'beta']\n"`` and scalar
    numerics such as ``"0.0001"`` as text.  Parse those literals and otherwise
    preserve the stripped text.  ``ast.literal_eval`` avoids executing
    arbitrary contents from the metafile.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [normalize_metafile_value(item) for item in value]
    if not isinstance(value, str):
        return value

    value = value.strip()
    if len(value) >= 2 and (
        (value[0], value[-1]) in {("[", "]"), ("(", ")"), ("{", "}")}
        or value[0] == value[-1] in {"'", '"'}
    ):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value
        return normalize_metafile_value(parsed)

    # The mixed-type pandas column serializes scalar numerics as strings.
    # Accept only numeric literal results here so semantic text such as
    # "None" remains text instead of becoming an unsupported null attribute.
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value
    if isinstance(parsed, (int, float)) and not isinstance(parsed, bool):
        return parsed
    return value


def normalize_metafile(metadata):
    """Normalize all values read from a RHINO metadata pickle."""
    return {
        key: normalize_metafile_value(value)
        for key, value in metadata.items()
    }


def _write_mapped_metadata(series, metadata, namespace, mapping):
    """Write available RHINO metafile fields using explicit schema names."""
    for source_name, schema_name in mapping.items():
        if source_name not in metadata or metadata[source_name] is None:
            continue
        series.set_attribute(
            f"{namespace}:{schema_name}", metadata[source_name]
        )


def write_metafile_metadata(series, metadata):
    """Write RHINO metafile entries in simulation and parameter-scan groups."""
    _write_mapped_metadata(
        series,
        metadata,
        "simulation",
        {
            "dt": "timeStep",
            "calc_length": "calculationLength",
            "units": "timeUnits",
            "nodes": "computeNodes",
            "fusion_type": "fusionType",
            "multiplier": "multiplier",
            "gas_streams": "gasStreams",
        },
    )
    _write_mapped_metadata(
        series,
        metadata,
        "parameterScan",
        {
            "runs_type": "description",
            "num_input_files": "numberOfInputFiles",
            "what": "variables",
            "start": "start",
            "stop": "stop",
            "increment": "increment",
            "end": "end",
            "index": "index",
            "subsystem": "selectedSubsystems",
            "species": "selectedSpecies",
        },
    )

    sweep_entries = []
    for key, value in metadata.items():
        prefix = "Sweep Changes_"
        if not key.startswith(prefix):
            continue
        try:
            index = int(key[len(prefix):])
        except ValueError as exc:
            raise ValueError(
                f"Invalid RHINO sweep-change metadata key {key!r}"
            ) from exc
        sweep_entries.append((index, value))

    if sweep_entries:
        sweep_entries.sort(key=lambda entry: entry[0])
        series.set_attribute(
            "parameterScan:changeIndices",
            [entry[0] for entry in sweep_entries],
        )
        series.set_attribute(
            "parameterScan:changes",
            [entry[1] for entry in sweep_entries],
        )
        series.set_attribute(
            "parameterScan:numberOfChanges", len(sweep_entries)
        )


def write_grouped_series_metadata(
    series,
    input_directory,
    input_paths,
    original_input_path,
    simulation_datetime,
    metadata=None,
    creation_datetime=None,
):
    """Write software, provenance, and system metadata in namespaces."""
    if creation_datetime is None:
        creation_datetime = datetime.now(timezone.utc)
    if creation_datetime.tzinfo is None:
        raise ValueError("creation_datetime must include a timezone")

    creation_datetime = creation_datetime.astimezone(timezone.utc)
    creation_timestamp = creation_datetime.isoformat(timespec="seconds")
    input_directory = Path(input_directory).resolve()
    input_files = _relative_input_files(input_directory, input_paths)
    metadata = normalize_metafile(metadata or {})
    fallback_original_input_file = _relative_input_files(
        input_directory, [original_input_path]
    )[0]
    original_input_file = metadata.get(
        "input_file", fallback_original_input_file
    )
    original_data_directory = metadata.get(
        "main_path", str(input_directory)
    )

    series.set_attribute("software:softwareName", "RHINO")
    series.set_attribute(
        "software:softwareDescription",
        "RHINO: Fusion Pilot Plant fuel cycle simulation",
    )
    series.set_attribute("software:softwareVersion", "1.0")
    series.set_attribute("software:versionControlSoftware", "GitHub")
    series.set_attribute("software:softwareCommit", "")
    series.set_attribute(
        "software:softwareDocumentation",
        "https://github.com/cbhowmic/IFE_AmSC",
    )

    series.set_attribute("provenance:author", "Holly Flynn")
    series.set_attribute(
        "provenance:authorAffiliation",
        "Savannah River National Laboratory",
    )
    series.set_attribute(
        "provenance:authorEmail", "Holly.Flynn@srnl.doe.gov"
    )
    series.set_attribute("provenance:creationDate", creation_timestamp)
    series.set_attribute("provenance:creationTimeUTC", creation_timestamp)
    series.set_attribute(
        "provenance:simulationDate",
        simulation_datetime.isoformat(timespec="seconds"),
    )
    series.set_attribute(
        "provenance:inputDirectory", str(input_directory)
    )
    series.set_attribute("provenance:inputFiles", input_files)
    series.set_attribute(
        "provenance:originalDataDirectory", original_data_directory
    )
    series.set_attribute(
        "provenance:originalDataFiles", [original_input_file]
    )
    series.set_attribute(
        "provenance:originalInputDirectory", original_data_directory
    )
    series.set_attribute(
        "provenance:originalInputFiles", [original_input_file]
    )
    series.set_attribute(
        "provenance:originalInputFile", original_input_file
    )

    series.set_attribute("system:systemIP", "")
    series.set_attribute("system:systemDescription", "")
    series.set_attribute("system:comment", "")

    write_metafile_metadata(series, metadata)


def rhino_to_adios(DATA_PATH, PREFIX, INFIX, OUTPUT_PATH):
    #PREFIX="22-21-28" 
    #INFIX ="IFE_AmSC_500MW_FuelCycle"  
    
    # Paths where RHINO data is 
    #RHINO_PATH = "/global/cfs/cdirs/m3239/2026_FES-AmSC/data/rhino/Surrogate Data" 
    #DATA_PATH  = f"{RHINO_PATH}/Power&BurnFractionScan_Daily_Reduced1" 
    INPUT_PATH = f"{DATA_PATH}/{PREFIX}_IFE_input.pkl" 
    POSTPROC_PATH = f"{DATA_PATH}/{PREFIX}_IFE_processed.pkl" 
    META_PATH = f"{DATA_PATH}/{PREFIX}_IFE_meta.pkl"
    simulation_datetime = simulation_datetime_from_source(DATA_PATH, PREFIX)
    
    # Import input file 
    #sys.path.append(RHINO_PATH)
    #from makeJSON import InputFile
    
    #######################
    ### Load RHINO data ###
    #######################
    # Metafile
    meta_df = pd.read_pickle(META_PATH)
    # Input file
    InputFile = pd.read_pickle(INPUT_PATH)
    PostProcData = pd.read_pickle(POSTPROC_PATH)
    
    ########################
    ### Extract metadata ###
    ########################
    # Convert metadata to dictionary
    meta = normalize_metafile(meta_df[0].to_dict())
    
    # Get timestep
    dt = float(meta["dt"]) if "dt" in meta else None
    if dt is None:
        raise ValueError("dt is required for time-series data")   
    
    # Get simulation time 
    endtime = float(meta["calc_length"])
    #times = np.arange(0, endtime+dt, dt)
    #nt = len(times)
    
    # Useful constant 
    SECONDS_PER_DAY = 86400.0
    
    ######################
    ### Extract inputs ###
    ######################
    my_inputs, my_inventory = load_available_species(
        DATA_PATH,
        PREFIX,
        INFIX,
        InputFile,
    )
    connectivity_graphs = {
        species_name: build_connectivity_graph(subsystems)
        for species_name, subsystems in my_inputs.items()
    }
    
    #########################
    ### Create time grid ###
    #########################
    first_inventory = next(iter(my_inventory.values()))
    Nt = first_inventory["data_ts"].shape[1]
    
    # Save times
    times = np.linspace(0, endtime, Nt)

    # Time to steady state 
    # if "box" not in substystem and subs != Storage_delivery -> compute time to ss
    ss_tol = 0.02  # tolerance band for ss

    def time_to_steady_state(times, ts, ss, tol=0.01):
        if np.isclose(ss, 0.0):
            err = np.abs(ts - ss)
        else:
            err = np.abs((ts - ss) / ss)
        inside = err <= tol
        i = len(times) - 1
        while i >= 0 and inside[i]:
            i -= 1
        if i == len(times) - 1:
            return np.nan
        return float(times[i + 1])
    t_ss_by_subsystem = {}
    if "Tritium" in my_inventory:
        tritium_inventory = my_inventory["Tritium"]
        for i, name in enumerate(my_inputs["Tritium"]):
            lname = name.lower()
            if "box" in lname or "storage" in lname or "delivery" in lname:
                continue
            t_ss_by_subsystem[name] = time_to_steady_state(
                times=times,
                ts=tritium_inventory["data_ts"][i, :],
                ss=tritium_inventory["data_ss"][i],
                tol=ss_tol,
            )

    valid_t_ss = [t for t in t_ss_by_subsystem.values() if not np.isnan(t)]
    if len(valid_t_ss) > 0:
        ss_time = float(np.max(valid_t_ss))
    else:
        ss_time = np.nan
    #############################
    ### Create openPMD series ###
    #############################
    adios2_cfg = r'''
    {
      "iteration_encoding": "variable_based",
      "adios2": {
        "modifiable_attributes": false,
        "use_group_table": false,
        "engine": {
          "type": "bp5",
          "parameters": {
            "StatsLevel": "1",
            "AsyncWrite": "guided"
          }
        }
      }
    }
    '''
    series = io.Series(OUTPUT_PATH, io.Access_Type.create_linear, adios2_cfg)
    print("Converting RHINO data into openPMD/ADIOS2 format...")
    print(f"Input: {DATA_PATH}")
    
    #########################
    ### Series attributes ###
    #########################
    # openPMD automatically writes its required flat software, softwareVersion,
    # and date attributes for the serialization library. RHINO metadata is
    # written separately in the application-specific namespaces below.
    series.particles_path = "inventory"
    provenance_input_paths = [INPUT_PATH, META_PATH, POSTPROC_PATH]
    for inventory in my_inventory.values():
        provenance_input_paths.extend(inventory["source_files"])
    write_grouped_series_metadata(
        series=series,
        input_directory=DATA_PATH,
        input_paths=provenance_input_paths,
        original_input_path=INPUT_PATH,
        simulation_datetime=simulation_datetime,
        metadata=meta,
    )
    series.set_attribute("species:names", list(my_inputs))
    # General inputs shared by the available species
    series.set_attribute("input:TBR:Tritium Breeding Ratio", InputFile["System Inputs"]["TBR"])
    series.set_attribute("input:TBRr:Required Tritium Breeding Ratio", InputFile["System Inputs"]["TBRr"])
    series.set_attribute("input:beta:Burn fraction", InputFile["System Inputs"]["beta"])
    series.set_attribute("input:eta:Fueling efficiency", InputFile["System Inputs"]["eta"])
    series.set_attribute("input:Ndotminus:Tritium burned per day", InputFile["System Inputs"]["Ndotminus"])
    series.set_attribute("input:MW:Power output in MW", InputFile["System Inputs"]["MW"])
    series.set_attribute("input:I0_SD:Starting inventory", InputFile["System Inputs"]["I0_SD"])

    # Post-processes outputs computed by Holly 
    # I0 (g)	Imin (g)	I_startup (g)	I_subtract (g)	reserve_time (days)	Iops (g)	plant_doubling_time (days)
    for k,v in PostProcData.items():
        series.set_attribute(f"output:{k}", v)
    series.set_attribute("output:Steady state time (days)", ss_time)
    
    ##########################
    ### Create iteration 0 ###
    ##########################
    it = series.snapshots()[0]
    it.time = 0.0
    it.dt = float(dt)
    it.time_unit_SI = SECONDS_PER_DAY
    
    #######################
    ### Save time array ###
    #######################
    species = it.particles["Times"]  
    species.set_attribute("description", "Times")
    record = species["data"]
    record.unit_dimension =  {io.Unit_Dimension.T: 1}
    record.unit_SI = SECONDS_PER_DAY
    data = np.ascontiguousarray(times).copy()
    dataset = io.Dataset(times.dtype, times.shape)
    component = record[io.Record_Component.SCALAR]
    component = record[io.Record_Component.SCALAR]
    component.reset_dataset(dataset)
    component.store_chunk(data)
    
    #############################
    ### Save available species ###
    #############################
    def write_species(name, data_ss, data_ts):
    
        pt = it.particles[name]
        pt.set_attribute("description", "Inventory across subsystems for species " + name)
        pt.set_attribute("timeAxis", 1)  
        pt.set_attribute("subsystemsAxis", 0)

        # Connectivity is metadata on the species, so edge counts do not need
        # to match the extents of the species' inventory records.
        write_connectivity_graph(pt, connectivity_graphs[name])

        # Subsystem definitions and steady-state values align with axis 0 of
        # mass. Storing them as attributes leaves mass as the species' only
        # record and therefore keeps its particle-record extents homogeneous.
        write_subsystem_metadata(
            particle_species=pt,
            subsystems=my_inputs[name],
            steady_state_mass=data_ss,
            time_unit=meta.get("units", "days"),
            time_unit_si=SECONDS_PER_DAY,
        )
    
        # time-series inventory data
        data_arr = np.ascontiguousarray(data_ts).copy()
        inv_rec = pt["mass"][io.Record_Component.SCALAR]
        inv_rec.reset_dataset(io.Dataset(data_arr.dtype, data_arr.shape))
        inv_rec.store_chunk(data_arr)
    
        pt["mass"].unit_dimension = {io.Unit_Dimension.M: 1}
        inv_rec.unit_SI = 1e-3
    
    for species_name, inventory in my_inventory.items():
        write_species(
            species_name,
            inventory["data_ss"],
            inventory["data_ts"],
        )
    
    ######################
    ### Close and save ###
    ######################
    it.close()
    series.close()
    print("RHINO data written to ADIOS-OpenPMD in particle representation.")
    print("Output:", OUTPUT_PATH)


def main() -> None:
    """Command-line entry point for converting one RHINO run to openPMD/ADIOS2."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert one RHINO run into openPMD/ADIOS2 BP5 format."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Directory containing the RHINO pickle files for one scenario.",
    )
    parser.add_argument(
        "--prefix",
        required=True,
        help="Run prefix, for example '22-21-28'.",
    )
    parser.add_argument(
        "--infix",
        required=True,
        help="Run/file infix, for example 'IFE_AmSC_500MW_FuelCycle'.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Output BP5 path.",
    )

    args = parser.parse_args()

    rhino_to_adios(
        DATA_PATH=args.data_path,
        PREFIX=args.prefix,
        INFIX=args.infix,
        OUTPUT_PATH=args.output_path,
    )


if __name__ == "__main__":
    main()

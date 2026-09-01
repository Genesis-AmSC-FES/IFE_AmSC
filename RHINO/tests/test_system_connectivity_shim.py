import importlib.util
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _load_ai_ready_rhino_write():
    module_path = (
        Path(__file__).parents[1]
        / "AI_ready_workflow"
        / "1_shim"
        / "rhinoWrite.py"
    )
    spec = importlib.util.spec_from_file_location(
        "ai_ready_rhino_write_for_test", module_path
    )
    module = importlib.util.module_from_spec(spec)

    previous_openpmd = sys.modules.get("openpmd_api")
    sys.modules["openpmd_api"] = types.SimpleNamespace()
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_openpmd is None:
            sys.modules.pop("openpmd_api", None)
        else:
            sys.modules["openpmd_api"] = previous_openpmd
    return module


rhino_write = _load_ai_ready_rhino_write()


def _subsystem(subsystem_id, injectors, fractional_inflows):
    return {
        "id": subsystem_id,
        "injectors": injectors,
        "fractional inflows": fractional_inflows,
    }


def _raw_subsystem(name, injectors=None, fractional_inflows=None):
    if injectors is None:
        injectors = ["None"]
    if fractional_inflows is None:
        fractional_inflows = ["None"]
    return [
        name,
        0.1,
        0.0,
        fractional_inflows,
        0.0,
        0.0,
        injectors,
        name,
    ]


def _write_species_pickles(
    root,
    prefix,
    infix,
    file_token,
    subsystem_names,
    timestep_count=3,
    reduced=True,
):
    suffix = "_reduced.pkl" if reduced else ".pkl"
    time_series_path = root / f"{prefix}_{infix}_{file_token}{suffix}"
    steady_state_path = (
        root / f"{prefix}_{infix}_{file_token}_SteadyState.pkl"
    )
    pd.DataFrame(
        np.arange(len(subsystem_names) * timestep_count).reshape(
            len(subsystem_names), timestep_count
        ),
        index=subsystem_names,
    ).to_pickle(time_series_path)
    pd.DataFrame(
        {
            f"{file_token}_Steady_State_Inventories": np.arange(
                len(subsystem_names), dtype=np.float64
            )
        },
        index=subsystem_names,
    ).to_pickle(steady_state_path)


def test_build_connectivity_graph_preserves_known_and_unknown_edges():
    subsystems = {
        "Source": _subsystem(0, ["None"], ["None"]),
        "KnownTarget": _subsystem(1, [0], [0.75]),
        "UnknownTarget": _subsystem(2, [0, 1], ["None"]),
    }

    graph = rhino_write.build_connectivity_graph(subsystems)

    np.testing.assert_array_equal(graph["source_id"], [0, 0, 1])
    np.testing.assert_array_equal(graph["target_id"], [1, 2, 2])
    np.testing.assert_allclose(
        graph["fractional_inflow"], [0.75, np.nan, np.nan], equal_nan=True
    )
    np.testing.assert_array_equal(
        graph["fractional_inflow_defined"], [1, 0, 0]
    )
    assert graph["source_id"].dtype == np.int64
    assert graph["target_id"].dtype == np.int64
    assert graph["fractional_inflow"].dtype == np.float64
    assert graph["fractional_inflow_defined"].dtype == np.uint8


def test_build_connectivity_graph_rejects_unaligned_fractional_inflows():
    subsystems = {
        "SourceA": _subsystem(0, ["None"], ["None"]),
        "SourceB": _subsystem(1, ["None"], ["None"]),
        "Target": _subsystem(2, [0, 1], [0.5]),
    }

    with pytest.raises(ValueError, match="2 injectors but 1 fractional inflows"):
        rhino_write.build_connectivity_graph(subsystems)


def test_build_connectivity_graph_rejects_unknown_injector():
    subsystems = {
        "Target": _subsystem(0, [99], [1.0]),
    }

    with pytest.raises(ValueError, match="unknown injector id 99"):
        rhino_write.build_connectivity_graph(subsystems)


def test_build_connectivity_graph_rejects_duplicate_subsystem_ids():
    subsystems = {
        "First": _subsystem(0, ["None"], ["None"]),
        "Second": _subsystem(0, ["None"], ["None"]),
    }

    with pytest.raises(ValueError, match="Duplicate subsystem id 0"):
        rhino_write.build_connectivity_graph(subsystems)


def test_load_available_species_skips_absent_deuterium(tmp_path):
    prefix = "00-00-00"
    infix = "IFE_AmSC"
    input_file = {
        "Systems_T": pd.Series({0: _raw_subsystem("TritiumSystem")}),
    }
    _write_species_pickles(
        tmp_path,
        prefix,
        infix,
        "T",
        ["TritiumSystem"],
    )

    species_inputs, species_inventory = rhino_write.load_available_species(
        tmp_path, prefix, infix, input_file
    )

    assert list(species_inputs) == ["Tritium"]
    assert list(species_inventory) == ["Tritium"]
    assert species_inventory["Tritium"]["data_ts"].shape == (1, 3)
    assert [
        path.name for path in species_inventory["Tritium"]["source_files"]
    ] == [
        f"{prefix}_{infix}_T_reduced.pkl",
        f"{prefix}_{infix}_T_SteadyState.pkl",
    ]


def test_load_available_species_supports_deuterium_full_time_series(tmp_path):
    prefix = "00-00-00"
    infix = "IFE_AmSC"
    input_file = {
        "Systems_D": pd.Series({0: _raw_subsystem("DeuteriumSystem")}),
    }
    _write_species_pickles(
        tmp_path,
        prefix,
        infix,
        "D",
        ["DeuteriumSystem"],
        reduced=False,
    )

    species_inputs, species_inventory = rhino_write.load_available_species(
        tmp_path, prefix, infix, input_file
    )

    assert list(species_inputs) == ["Deuterium"]
    assert species_inventory["Deuterium"]["data_ss"].tolist() == [0.0]


def test_load_available_species_supports_registry_extension(tmp_path):
    prefix = "00-00-00"
    infix = "IFE_AmSC"
    input_file = {
        "Systems_He3": pd.Series({0: _raw_subsystem("Helium3System")}),
    }
    _write_species_pickles(
        tmp_path,
        prefix,
        infix,
        "He3",
        ["Helium3System"],
    )
    config = {
        "Helium3": {
            "system_key": "Systems_He3",
            "file_token": "He3",
        }
    }

    species_inputs, species_inventory = rhino_write.load_available_species(
        tmp_path,
        prefix,
        infix,
        input_file,
        species_config=config,
    )

    assert list(species_inputs) == ["Helium3"]
    assert species_inventory["Helium3"]["data_ts"].shape == (1, 3)


def test_load_available_species_rejects_incomplete_present_species(tmp_path):
    input_file = {
        "Systems_D": pd.Series({0: _raw_subsystem("DeuteriumSystem")}),
    }

    with pytest.raises(FileNotFoundError, match="Missing D time-series pickle"):
        rhino_write.load_available_species(
            tmp_path,
            "00-00-00",
            "IFE_AmSC",
            input_file,
        )


class _FakeParticleSpecies:
    def __init__(self):
        self.attributes = {}

    def set_attribute(self, name, value):
        self.attributes[name] = value


class _FakeSeries:
    def __init__(self):
        self.attributes = {}

    def set_attribute(self, name, value):
        self.attributes[name] = value


def test_write_grouped_series_metadata_uses_namespaces(tmp_path):
    input_path = tmp_path / "00-00-00_IFE_input.pkl"
    meta_path = tmp_path / "00-00-00_IFE_meta.pkl"
    series = _FakeSeries()

    rhino_write.write_grouped_series_metadata(
        series=series,
        input_directory=tmp_path,
        input_paths=[input_path, meta_path],
        original_input_path=input_path,
        simulation_datetime=datetime(2026, 1, 2, 3, 4, 5),
        creation_datetime=datetime(
            2026, 2, 3, 4, 5, 6, tzinfo=timezone.utc
        ),
    )

    attributes = series.attributes
    assert attributes["software:softwareName"] == "RHINO"
    assert attributes["software:softwareVersion"] == "1.0"
    assert attributes["software:versionControlSoftware"] == "GitHub"
    assert attributes["provenance:creationDate"] == "2026-02-03T04:05:06+00:00"
    assert attributes["provenance:simulationDate"] == "2026-01-02T03:04:05"
    assert attributes["provenance:inputDirectory"] == str(tmp_path.resolve())
    assert attributes["provenance:inputFiles"] == [
        input_path.name,
        meta_path.name,
    ]
    assert attributes["provenance:originalInputFiles"] == [input_path.name]
    assert "system:systemIP" in attributes
    assert "system:systemDescription" in attributes

    flat_attributes = {
        "software",
        "softwareVersion",
        "softwareDescription",
        "author",
        "authorAffiliation",
        "authorEmail",
        "date",
        "machine",
        "comment",
    }
    assert flat_attributes.isdisjoint(attributes)


def test_write_grouped_series_metadata_records_metafile_groups(tmp_path):
    input_path = tmp_path / "00-00-00_IFE_input.pkl"
    series = _FakeSeries()
    original_json = r"C:\RHINO_Runs\IFE_scan.json"
    original_directory = r"C:\RHINO_Runs"
    metadata = {
        "dt": "0.0001",
        "calc_length": "180.0",
        "units": "days",
        "input_file": original_json,
        "nodes": "8",
        "start": "[216.71, 0.076]\n",
        "stop": 1.25,
        "increment": "[1.28, 0.043]\n",
        "what": "['Ndotminus', 'beta']\n",
        "index": "[0, 0]\n",
        "end": "[648, 0.291]\n",
        "subsystem": "['Isotope_Seperation']",
        "species": "'None'",
        "fusion_type": "IFE",
        "runs_type": "Parameter scan\n",
        "num_input_files": "3012",
        "main_path": original_directory,
        "multiplier": 1.0,
        "Sweep Changes_1": "beta-0.119",
        "Sweep Changes_0": "beta-0.076",
        "gas_streams": "['Systems_T']",
    }

    rhino_write.write_grouped_series_metadata(
        series=series,
        input_directory=tmp_path,
        input_paths=[input_path],
        original_input_path=input_path,
        simulation_datetime=datetime(2026, 1, 2, 3, 4, 5),
        metadata=metadata,
        creation_datetime=datetime(
            2026, 2, 3, 4, 5, 6, tzinfo=timezone.utc
        ),
    )

    attributes = series.attributes
    assert attributes["provenance:originalInputFile"] == original_json
    assert (
        attributes["provenance:originalDataDirectory"]
        == original_directory
    )
    assert attributes["simulation:timeStep"] == 0.0001
    assert attributes["simulation:calculationLength"] == 180.0
    assert attributes["simulation:gasStreams"] == ["Systems_T"]
    assert attributes["parameterScan:variables"] == ["Ndotminus", "beta"]
    assert attributes["parameterScan:start"] == [216.71, 0.076]
    assert attributes["parameterScan:selectedSubsystems"] == [
        "Isotope_Seperation"
    ]
    assert attributes["parameterScan:selectedSpecies"] == "None"
    assert attributes["parameterScan:numberOfInputFiles"] == 3012
    assert attributes["parameterScan:changeIndices"] == [0, 1]
    assert attributes["parameterScan:changes"] == [
        "beta-0.076",
        "beta-0.119",
    ]
    assert attributes["parameterScan:numberOfChanges"] == 2
    assert not any(name.startswith("campaign:") for name in attributes)


def test_write_grouped_series_metadata_requires_creation_timezone(tmp_path):
    with pytest.raises(ValueError, match="must include a timezone"):
        rhino_write.write_grouped_series_metadata(
            series=_FakeSeries(),
            input_directory=tmp_path,
            input_paths=[],
            original_input_path=tmp_path / "input.pkl",
            simulation_datetime=datetime(2026, 1, 2, 3, 4, 5),
            creation_datetime=datetime(2026, 2, 3, 4, 5, 6),
        )


def test_write_connectivity_graph_writes_species_attributes():
    graph = {
        "source_id": np.array([0], dtype=np.int64),
        "target_id": np.array([1], dtype=np.int64),
        "fractional_inflow": np.array([0.75], dtype=np.float64),
        "fractional_inflow_defined": np.array([1], dtype=np.uint8),
    }
    particle_species = _FakeParticleSpecies()

    rhino_write.write_connectivity_graph(particle_species, graph)

    assert particle_species.attributes == {
        "connectivity:representation": "directed_edge_list",
        "connectivity:sourceId": [0],
        "connectivity:targetId": [1],
        "connectivity:fractionalInflow": [0.75],
        "connectivity:fractionalInflowDefined": [1],
    }


def test_write_subsystem_metadata_writes_aligned_species_attributes():
    particle_species = _FakeParticleSpecies()
    subsystems = {
        "First": {
            "id": 3,
            "processing time": 0.25,
            "nonradioactive loss fraction": 0.01,
            "initial mass": 2.0,
            "source": -1.0,
            "label": "F",
        },
        "Second": {
            "id": 7,
            "processing time": 0.5,
            "nonradioactive loss fraction": 0.02,
            "initial mass": 4.0,
            "source": 1.0,
            "label": "S",
        },
    }

    rhino_write.write_subsystem_metadata(
        particle_species=particle_species,
        subsystems=subsystems,
        steady_state_mass=np.array([10.0, 20.0]),
        time_unit="days",
        time_unit_si=86400.0,
    )

    assert particle_species.attributes == {
        "subsystem:count": 2,
        "subsystem:id": [3, 7],
        "subsystem:name": ["First", "Second"],
        "subsystem:label": ["F", "S"],
        "subsystem:processingTime": [0.25, 0.5],
        "subsystem:processingTimeUnit": "days",
        "subsystem:processingTimeUnitSI": 86400.0,
        "subsystem:nonradioactiveLossFraction": [0.01, 0.02],
        "subsystem:initialMass": [2.0, 4.0],
        "subsystem:initialMassUnitSI": 1e-3,
        "subsystem:source": [-1.0, 1.0],
        "steadyState:mass": [10.0, 20.0],
        "steadyState:massUnitSI": 1e-3,
    }


def test_write_subsystem_metadata_rejects_length_mismatch():
    with pytest.raises(ValueError, match="2 subsystem definitions but 1"):
        rhino_write.write_subsystem_metadata(
            particle_species=_FakeParticleSpecies(),
            subsystems={
                "First": _subsystem(0, [], []),
                "Second": _subsystem(1, [], []),
            },
            steady_state_mass=np.array([1.0]),
            time_unit="days",
            time_unit_si=86400.0,
        )

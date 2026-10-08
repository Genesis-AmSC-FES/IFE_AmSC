"""Metadata and HDF5 readers for the Laser openPMD/ADIOS2 shim.

This module turns one Laser HDF5 file plus optional spreadsheet/domain metadata into an in-memory LaserH5Model. 
The writer can then serialize that model without having to understand Excel formats, HDF5 root attributes, or PV semantics.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import pandas as pd

from utils import (
    SCHEMA_VERSION,
    count_image_frames,
    deduplicate,
    find_first_existing,
    get_default_documentation_dir,
    get_image_groups,
    get_root_datasets,
    h5_attrs_to_dict,
    image_dataset_indices,
    infer_requested_sample_count,
    infer_valid_sample_mask,
    is_dataset,
    parse_json_list,
    parse_laser_filename,
    sanitize_pv_name,
    to_jsonable,
    trim_to_valid_samples,
)


DOCUMENTATION_DIR = get_default_documentation_dir()
DEFAULT_EXCEL_CANDIDATES = [
    DOCUMENTATION_DIR / "DT peening laser PVs (updated).xlsx",
    DOCUMENTATION_DIR / "peening laser PVs.xlsx",
    Path("DT peening laser PVs (updated).xlsx"),
    Path("peening laser PVs.xlsx"),
]


# -----------------------------------------------------------------------------
# Domain metadata from laser-pv-details.docx, represented in machine-readable form
# -----------------------------------------------------------------------------

DOMAIN_MANIFEST: dict[str, Any] = {
    "important_diagnostics": [
        {
            "pv_name": "PNG-digitizer:Ch2:Energy",
            "role": "diagnostic:laser_energy:after_main_amp",
            "description": "Actual laser output energy after the main amplifier.",
        },
        {
            "pv_name": "PNG-digitizer:Ch1:Energy",
            "role": "diagnostic:laser_energy:after_pre_amp",
            "description": "Laser energy after the pre-amplifier.",
        },
        {
            "pv_name": "PNG-digitizer:Ch3:Energy",
            "role": "diagnostic:laser_energy:after_oscillator",
            "description": "Laser energy after the oscillator.",
        },
        {
            "pv_name": "Siglent:Ch1:FWHM",
            "role": "diagnostic:pulse_width:oscillator",
            "description": "Oscillator pulse width.",
        },
        {
            "pv_name": "Siglent:Ch2:FWHM",
            "role": "diagnostic:pulse_width:after_pre_amp",
            "description": "Pulse width after the pre-amplifier.",
        },
        {
            "pv_name": "Siglent:Ch4:FWHM",
            "role": "diagnostic:pulse_width:after_main_amp",
            "description": "Pulse width after the main amplifier.",
        },
        {
            "pv_name": "CAM_PNG1:Pva1:Image",
            "role": "diagnostic:image:fabry_perot_raw",
            "description": "Fabry Perot raw mode-spectrum image.",
        },
        {
            "pv_name": "PNG:FabryPerot:NumberModes",
            "role": "diagnostic:mode_spectrum:number_modes",
            "description": "Number of peaks/modes derived from the Fabry Perot image.",
        },
        {
            "pv_name": "CAM_PNG2:Pva1:Image",
            "role": "diagnostic:image:output_nearfield",
            "description": "Output nearfield beam profile image.",
        },
        {
            "pv_name": "CAM_PNG5:Pva1:Image",
            "role": "diagnostic:image:output_farfield_focus",
            "description": "Output farfield/focus image.",
        },
    ],
    "control_response": [
        {
            "control_pv": "PNG:PP:VoltageProgram",
            "control_role": "control:main_amp_gain",
            "response_pvs": [
                "PNG:PP:Circuit1:Current",
                "PNG:PP:Circuti1:Current",  # keep misspelled legacy form if present
                "PNG-digitizer:Ch2:Energy",
            ],
        },
        {
            "control_pv": "PNG:QT3:CurrentProgram",
            "control_role": "control:oscillator_gain",
            "response_pvs": [
                "PNG:QT3:Current",
                "PNG-digitizer:Ch1:Energy",
                "PNG-digitizer:Ch2:Energy",
                "PNG-digitizer:Ch3:Energy",
                "Siglent:Ch1:FWHM",
                "Siglent:Ch2:FWHM",
                "Siglent:Ch4:FWHM",
            ],
        },
        {
            "control_pv": "PNG:Etalon:TemperatureProgram",
            "control_role": "control:etalon_temperature",
            "response_pvs": [
                "PNG:Etalon:Temperature",
                "CAM_PNG1:Pva1:Image",
                "PNG:FabryPerot:NumberModes",
                "PNG-digitizer:Ch1:Energy",
                "PNG-digitizer:Ch2:Energy",
                "PNG-digitizer:Ch3:Energy",
            ],
        },
    ],
    "image_validity": {
        "timestamp_meaning": (
            "For CCD images, the timestamp records when the acquisition script "
            "attempted to read the image, not necessarily the camera exposure time."
        ),
        "preferred_validity_rule": "new frame if CAM_PNG*:cam1:ArrayCounter_RBV increments",
        "array_counter_template": "{camera_prefix}:cam1:ArrayCounter_RBV",
    },
    "full_system_criteria": {
        "description": "Samples where the main amplifier fires.",
        "criteria": [
            {"pv_name": "PNG:ModeTab:Index", "operator": "==", "value": 1},
            {"pv_name": "PNG-digitizer:Ch2:Energy", "operator": ">", "value": 0.1},
            {"pv_name": "PNG:PP:Armed", "operator": "==", "value": 1},
            {
                "any_pv": ["PNG:PP:Circuit1:Current", "PNG:PP:Circuti1:Current"],
                "operator": ">",
                "value": 0.1,
            },
        ],
        "filename_rule": {
            "png": "full-system peening run",
            "alignment": "alignment/acquisition run",
        },
    },
}


@dataclass
class PVData:
    """One scalar, trace, coordinate, or timestamp PV array."""

    name: str
    data: np.ndarray
    attrs: dict[str, Any] = field(default_factory=dict)
    excel: dict[str, Any] = field(default_factory=dict)
    data_role: str = ""
    semantic_roles: list[str] = field(default_factory=list)
    coordinate_name: str | None = None
    coordinate_role: str | None = None
    coordinate_data: np.ndarray | None = None
    coordinate_attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageData:
    """One image-producing camera/image PV group."""

    name: str
    data: np.ndarray
    frame_index: np.ndarray
    timestamp: np.ndarray
    attrs: dict[str, Any] = field(default_factory=dict)
    declared: bool = False
    array_counter_pv: str | None = None
    array_counter: np.ndarray | None = None
    new_frame_mask: np.ndarray | None = None
    semantic_roles: list[str] = field(default_factory=list)


@dataclass
class LaserH5Model:
    """In-memory representation of one Laser HDF5 file."""

    h5_path: Path
    file_info: dict[str, str]
    root_attrs: dict[str, Any]
    n_requested: int
    valid_sample_mask: np.ndarray
    source_indices: np.ndarray
    scalar_pvs_declared: list[str]
    array_pvs_declared: list[str]
    image_pvs_declared: list[str]
    scalars: dict[str, PVData]
    traces: dict[str, PVData]
    coordinates: dict[str, PVData]
    images: dict[str, ImageData]
    timestamps: dict[str, PVData]
    full_system_mask: np.ndarray | None
    pv_manifest: dict[str, Any]
    image_manifest: dict[str, Any]
    conversion_metadata: dict[str, Any]

    @property
    def num_valid_samples(self) -> int:
        return int(self.source_indices.size)


# -----------------------------------------------------------------------------
# Spreadsheet and documentation metadata
# -----------------------------------------------------------------------------

def _clean_cell(value: Any) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, float) and np.isnan(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def _looks_like_pv(value: str) -> bool:
    value = (value or "").strip()
    return bool(value and ":" in value)


def load_excel_pv_metadata(excel_path: str | os.PathLike[str]) -> dict[str, dict[str, Any]]:
    """Load PV metadata from the Laser PV spreadsheet.

    The current spreadsheet contains category heading rows such as
    "CONTROL PVs (scalars)", followed by rows of EPICS PV names. We preserve
    that heading as excel:category.
    """
    df = pd.read_excel(excel_path, header=0)
    df = df.rename(columns={column: str(column).strip() for column in df.columns})

    pv_column = "EPICS Process Variable"
    if pv_column not in df.columns:
        for column in df.columns:
            if "EPICS" in column and "Variable" in column:
                pv_column = column
                break
        else:
            raise ValueError(f"Could not find PV column in Excel file: {list(df.columns)}")

    metadata: dict[str, dict[str, Any]] = {}
    current_category = ""
    for _, row in df.iterrows():
        raw_name = _clean_cell(row.get(pv_column, ""))
        description = _clean_cell(row.get("Description", ""))
        range_text = _clean_cell(row.get("Range", ""))
        nominal = _clean_cell(row.get("Nominal", ""))
        notes = _clean_cell(row.get("Notes", ""))

        if not raw_name:
            continue
        if not _looks_like_pv(raw_name):
            current_category = raw_name
            continue

        metadata[raw_name] = {
            "excel:description": description,
            "excel:range": range_text,
            "excel:nominal": nominal,
            "excel:notes": notes,
            "excel:category": current_category,
            "excel:source": str(Path(excel_path).resolve()),
        }

    return metadata


def build_documentation_metadata(
    documentation_dir: str | os.PathLike[str] = DOCUMENTATION_DIR,
    pv_metadata_source: str = "",
    domain_notes_source: str = "",
) -> dict[str, str]:
    """Return documentation source paths used by the Laser writer."""
    documentation_dir = Path(documentation_dir)
    return {
        "readme_rtf": str(documentation_dir / "README.rtf"),
        "laser_schematic_pptx": str(documentation_dir / "Laser schematic.pptx"),
        "pv_metadata_source": pv_metadata_source,
        "domain_notes_source": domain_notes_source,
    }


def load_conversion_metadata(
    excel_candidates: list[str | os.PathLike[str]] | None = None,
    documentation_dir: str | os.PathLike[str] = DOCUMENTATION_DIR,
    domain_notes_source: str = "",
    verbose: bool = True,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load optional metadata used to enrich the BP5 output."""
    documentation_meta = build_documentation_metadata(
        documentation_dir=documentation_dir,
        domain_notes_source=domain_notes_source,
    )
    pv_meta: dict[str, dict[str, Any]] = {}

    candidates = excel_candidates or DEFAULT_EXCEL_CANDIDATES
    excel_path = find_first_existing(candidates)
    if not excel_path:
        if verbose:
            print("[meta] No PV Excel metadata file found; skipping Excel enrichment.")
        return pv_meta, documentation_meta

    try:
        pv_meta = load_excel_pv_metadata(excel_path)
    except Exception as exc:
        if verbose:
            print(f"[meta] WARNING: Failed to load Excel metadata ({excel_path}): {exc}")
        return pv_meta, documentation_meta

    documentation_meta["pv_metadata_source"] = str(Path(excel_path).resolve())
    if verbose:
        print(f"[meta] Loaded PV metadata from {excel_path} ({len(pv_meta)} PV entries)")
    return pv_meta, documentation_meta


# -----------------------------------------------------------------------------
# PV semantic helpers
# -----------------------------------------------------------------------------

def semantic_roles_for_pv(pv_name: str, domain_manifest: Mapping[str, Any] = DOMAIN_MANIFEST) -> list[str]:
    """Return domain roles for one PV."""
    roles: list[str] = []

    for entry in domain_manifest.get("important_diagnostics", []):
        if entry.get("pv_name") == pv_name:
            roles.append(str(entry.get("role", "diagnostic")))

    for relation in domain_manifest.get("control_response", []):
        if relation.get("control_pv") == pv_name:
            roles.append(str(relation.get("control_role", "control")))
        if pv_name in relation.get("response_pvs", []):
            control = relation.get("control_pv", "unknown_control")
            roles.append(f"response_to:{control}")

    if pv_name.endswith(":Time"):
        roles.append("coordinate:time")
    if pv_name.endswith(":Frequencies"):
        roles.append("coordinate:frequency")
    if pv_name.endswith(":Wavelengths"):
        roles.append("coordinate:wavelength")
    if pv_name.endswith("HistogramVoltages"):
        roles.append("coordinate:histogram_bin")

    return deduplicate(roles)


def is_coordinate_array_pv(pv_name: str) -> bool:
    """Return true if an array PV is a coordinate rather than a signal."""
    lowered = pv_name.lower()
    return (
        lowered.endswith(":time")
        or lowered.endswith(":frequencies")
        or lowered.endswith(":wavelengths")
        or lowered.endswith("histogramvoltages")
    )


def coordinate_role_for_pv(pv_name: str) -> str:
    lowered = pv_name.lower()
    if lowered.endswith(":time"):
        return "time"
    if lowered.endswith(":frequencies"):
        return "frequency"
    if lowered.endswith(":wavelengths"):
        return "wavelength"
    if lowered.endswith("histogramvoltages"):
        return "histogram_bin"
    return "coordinate"


def match_coordinate_pv(trace_pv: str, coordinates: Mapping[str, PVData]) -> str | None:
    """Match a trace/waveform PV to a coordinate PV when possible."""
    if not coordinates:
        return None

    prefix = trace_pv.rsplit(":", 1)[0] if ":" in trace_pv else trace_pv
    first_token = trace_pv.split(":", 1)[0]

    candidates = [
        f"{prefix}:Time",
        f"{first_token}:Time",
    ]

    lowered = trace_pv.lower()
    if any(token in lowered for token in [":psd", ":asd", ":magasd"]):
        candidates.extend(
            [
                f"{prefix}:Frequencies",
                f"{first_token}:Frequencies",
                "PNG:Accelerometer:Frequencies",
            ]
        )
    if "histogram" in lowered and not lowered.endswith("histogramvoltages"):
        candidates.extend([f"{prefix}Voltages", f"{first_token}:LampShotHistogramVoltages"])
    if any(token in lowered for token in ["spectrum", "intensity", "spectra"]):
        candidates.extend([f"{prefix}:Wavelengths", f"{first_token}:Wavelengths"])

    for candidate in candidates:
        if candidate in coordinates:
            return candidate
    return None


def camera_prefix_from_image_pv(image_pv: str) -> str:
    """Return CAM_PNG1 from CAM_PNG1:Pva1:Image."""
    return image_pv.split(":", 1)[0]


def array_counter_pv_for_image(image_pv: str) -> str:
    """Return the expected areaDetector array counter PV for one camera image PV."""
    return f"{camera_prefix_from_image_pv(image_pv)}:cam1:ArrayCounter_RBV"


# -----------------------------------------------------------------------------
# HDF5 readers
# -----------------------------------------------------------------------------

def _dataset_pv_name(name: str, dataset: h5py.Dataset) -> str:
    attrs = h5_attrs_to_dict(dataset.attrs)
    return str(attrs.get("pvname") or name)


def _dataset_attrs(dataset: h5py.Dataset) -> dict[str, Any]:
    attrs = h5_attrs_to_dict(dataset.attrs)
    # Normalize the original HDF5 names under epics:* keys during writing, but
    # keep raw names here so metadata/source/root_attrs remains original-like.
    return attrs


def _merge_excel(pv_name: str, excel_meta: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    return dict(excel_meta.get(pv_name, {}))


def _read_scalar_dataset(
    name: str,
    dataset: h5py.Dataset,
    valid_mask: np.ndarray,
    excel_meta: Mapping[str, dict[str, Any]],
) -> PVData:
    pv_name = _dataset_pv_name(name, dataset)
    data = trim_to_valid_samples(np.asarray(dataset[()]), valid_mask)
    return PVData(
        name=pv_name,
        data=data,
        attrs=_dataset_attrs(dataset),
        excel=_merge_excel(pv_name, excel_meta),
        data_role="scalar",
        semantic_roles=semantic_roles_for_pv(pv_name),
    )


def _read_array_dataset(
    name: str,
    dataset: h5py.Dataset,
    valid_mask: np.ndarray,
    excel_meta: Mapping[str, dict[str, Any]],
    data_role: str,
) -> PVData:
    pv_name = _dataset_pv_name(name, dataset)
    data = trim_to_valid_samples(np.asarray(dataset[()]), valid_mask)
    return PVData(
        name=pv_name,
        data=data,
        attrs=_dataset_attrs(dataset),
        excel=_merge_excel(pv_name, excel_meta),
        data_role=data_role,
        semantic_roles=semantic_roles_for_pv(pv_name),
    )


def load_image_stack(
    name: str,
    group: h5py.Group,
    declared_image_pvs: list[str],
    scalar_lookup: Mapping[str, PVData],
) -> ImageData:
    """Load one camera group into image/frame/timestamp arrays."""
    indices = image_dataset_indices(group)
    if not indices:
        raise ValueError(f"Image group contains no image frames: {name}")

    first = np.asarray(group[f"image {indices[0]}"][()])
    frames: list[np.ndarray] = []
    timestamps: list[float] = []
    for index in indices:
        dataset = group[f"image {index}"]
        image = np.asarray(dataset[()])
        if image.shape != first.shape:
            raise ValueError(
                f"Image frame shape mismatch in {name}: image {index} has {image.shape}, "
                f"expected {first.shape}"
            )
        frames.append(image)
        timestamp = h5_attrs_to_dict(dataset.attrs).get("timestamp")
        timestamps.append(float(timestamp) if timestamp not in (None, "") else np.nan)

    data = np.stack(frames, axis=0)
    frame_index = np.asarray(indices, dtype=np.int64)
    timestamp_array = np.asarray(timestamps, dtype=np.float64)
    attrs = h5_attrs_to_dict(group.attrs)

    counter_pv = array_counter_pv_for_image(name)
    counter_data: np.ndarray | None = None
    new_frame_mask: np.ndarray | None = None
    if counter_pv in scalar_lookup:
        counter = np.asarray(scalar_lookup[counter_pv].data)
        if counter.shape[0] >= frame_index.shape[0]:
            counter_data = counter[: frame_index.shape[0]]
            if counter_data.size:
                new_frame_mask = np.ones(counter_data.shape[0], dtype=np.uint8)
                if counter_data.shape[0] > 1:
                    new_frame_mask[1:] = (np.diff(counter_data.astype(float)) > 0).astype(np.uint8)

    return ImageData(
        name=name,
        data=data,
        frame_index=frame_index,
        timestamp=timestamp_array,
        attrs=attrs,
        declared=name in declared_image_pvs,
        array_counter_pv=counter_pv if counter_data is not None else None,
        array_counter=counter_data,
        new_frame_mask=new_frame_mask,
        semantic_roles=semantic_roles_for_pv(name),
    )


def _read_timestamp_datasets(
    h5_file: h5py.File,
    valid_mask: np.ndarray,
    excel_meta: Mapping[str, dict[str, Any]],
) -> dict[str, PVData]:
    timestamps: dict[str, PVData] = {}
    if "timestamps" not in h5_file or not isinstance(h5_file["timestamps"], h5py.Group):
        return timestamps

    for name, dataset in h5_file["timestamps"].items():
        if not is_dataset(dataset):
            continue
        raw = np.asarray(dataset[()])
        if raw.ndim < 1:
            continue
        data = trim_to_valid_samples(raw, valid_mask)
        pv_name = name.removesuffix(".timestamp")
        timestamps[name] = PVData(
            name=pv_name,
            data=data,
            attrs={"source_dataset": name, "source_group": "timestamps"},
            excel=_merge_excel(pv_name, excel_meta),
            data_role="timestamp",
            semantic_roles=semantic_roles_for_pv(pv_name),
        )
    return timestamps


def compute_full_system_mask(scalars: Mapping[str, PVData]) -> np.ndarray | None:
    """Compute full-system/main-amplifier firing mask when all criteria exist."""
    required = ["PNG:ModeTab:Index", "PNG-digitizer:Ch2:Energy", "PNG:PP:Armed"]
    if not all(name in scalars for name in required):
        return None

    current_name = None
    for candidate in ["PNG:PP:Circuit1:Current", "PNG:PP:Circuti1:Current"]:
        if candidate in scalars:
            current_name = candidate
            break
    if current_name is None:
        return None

    try:
        mode = np.asarray(scalars["PNG:ModeTab:Index"].data, dtype=float)
        energy = np.asarray(scalars["PNG-digitizer:Ch2:Energy"].data, dtype=float)
        armed = np.asarray(scalars["PNG:PP:Armed"].data, dtype=float)
        current = np.asarray(scalars[current_name].data, dtype=float)
    except (TypeError, ValueError):
        return None

    n = min(mode.size, energy.size, armed.size, current.size)
    if n == 0:
        return None
    mask = (mode[:n] == 1) & (energy[:n] > 0.1) & (armed[:n] == 1) & (current[:n] > 0.1)
    return mask.astype(np.uint8)


def build_pv_manifest(
    scalars: Mapping[str, PVData],
    traces: Mapping[str, PVData],
    coordinates: Mapping[str, PVData],
    timestamps: Mapping[str, PVData],
    scalar_pvs_declared: list[str],
    array_pvs_declared: list[str],
    image_pvs_declared: list[str],
) -> dict[str, Any]:
    """Build machine-readable PV classification and metadata manifest."""
    entries: dict[str, Any] = {}

    def add_entries(collection: Mapping[str, PVData]) -> None:
        for pv_name, pv in collection.items():
            entries[pv_name] = {
                "pv_name": pv.name,
                "safe_name": sanitize_pv_name(pv.name),
                "data_role": pv.data_role,
                "shape": list(pv.data.shape),
                "dtype": str(pv.data.dtype),
                "semantic_roles": pv.semantic_roles,
                "coordinate_name": pv.coordinate_name,
                "coordinate_role": pv.coordinate_role,
                "has_hdf5_metadata": bool(pv.attrs),
                "has_excel_metadata": bool(pv.excel),
                "excel_category": pv.excel.get("excel:category", ""),
                "description": pv.attrs.get("description") or pv.excel.get("excel:description", ""),
                "unitLabel": pv.attrs.get("units", ""),
            }

    add_entries(scalars)
    add_entries(traces)
    add_entries(coordinates)
    add_entries(timestamps)

    present_names = set(scalars) | set(traces) | set(coordinates) | {ts.name for ts in timestamps.values()}
    declared = set(scalar_pvs_declared) | set(array_pvs_declared)

    return {
        "schema_version": SCHEMA_VERSION,
        "entries": entries,
        "declared": {
            "scalar_pvs": scalar_pvs_declared,
            "array_pvs": array_pvs_declared,
            "image_pvs": image_pvs_declared,
        },
        "classification_rules": {
            "scalar": "PV is declared in root scalar_pvs, or is a fallback 1D non-image dataset.",
            "trace": "PV is declared in root array_pvs and is not a coordinate array.",
            "coordinate": "Array PV ending in :Time, :Frequencies, :Wavelengths, or HistogramVoltages.",
            "timestamp": "Dataset from /timestamps group.",
            "image": "Declared by root image_pvs and/or present as group with image <index> datasets.",
        },
        "missing_declared_nonimage_pvs": sorted(declared - present_names),
    }


def _get_nested_data_datasets(
    h5_file: h5py.File,
    image_groups: Mapping[str, h5py.Group],
) -> dict[str, h5py.Dataset]:
    """Return nested datasets that are neither image frames nor timestamps."""
    datasets: dict[str, h5py.Dataset] = {}

    def visit(name: str, obj: Any) -> None:
        if "/" not in name or not isinstance(obj, h5py.Dataset):
            return
        root_group = name.split("/", 1)[0]
        if root_group == "timestamps" or root_group in image_groups:
            return
        datasets[name] = obj

    h5_file.visititems(visit)
    return datasets


def build_image_manifest(
    images: Mapping[str, ImageData],
    declared_image_pvs: list[str],
) -> dict[str, Any]:
    present = sorted(images.keys())
    declared_set = set(declared_image_pvs)
    entries = {
        name: {
            "pv_name": image.name,
            "safe_name": sanitize_pv_name(image.name),
            "shape": list(image.data.shape),
            "dtype": str(image.data.dtype),
            "num_frames": int(image.frame_index.size),
            "declared_in_root_image_pvs": image.declared,
            "array_counter_pv": image.array_counter_pv,
            "has_array_counter": image.array_counter is not None,
            "semantic_roles": image.semantic_roles,
        }
        for name, image in images.items()
    }
    return {
        "declared_image_pvs": declared_image_pvs,
        "present_image_pvs": present,
        "missing_image_pvs": sorted(declared_set - set(present)),
        "extra_present_image_pvs": sorted(set(present) - declared_set),
        "entries": entries,
        "image_validity": DOMAIN_MANIFEST["image_validity"],
    }


def read_laser_h5(
    h5_path: str | os.PathLike[str],
    excel_meta: Mapping[str, dict[str, Any]] | None = None,
    strict: bool = False,
) -> LaserH5Model:
    """Read one Laser HDF5 file and classify arrays using root PV lists first."""
    h5_path = Path(h5_path)
    excel_meta = excel_meta or {}

    with h5py.File(h5_path, "r") as h5_file:
        root_attrs = h5_attrs_to_dict(h5_file.attrs)
        file_info = parse_laser_filename(h5_path)
        root_datasets = get_root_datasets(h5_file)
        image_groups = get_image_groups(h5_file)
        nested_datasets = _get_nested_data_datasets(h5_file, image_groups)
        data_datasets = {**root_datasets, **nested_datasets}

        scalar_pvs_declared = parse_json_list(root_attrs.get("scalar_pvs"))
        array_pvs_declared = parse_json_list(root_attrs.get("array_pvs"))
        image_pvs_declared = parse_json_list(root_attrs.get("image_pvs"))

        n_requested = infer_requested_sample_count(root_attrs, root_datasets)
        valid_mask = infer_valid_sample_mask(root_datasets, image_groups, n_requested)
        source_indices = np.flatnonzero(valid_mask).astype(np.int64)

        scalars: dict[str, PVData] = {}
        coordinates: dict[str, PVData] = {}
        traces: dict[str, PVData] = {}

        # Root metadata declarations are primary; nested non-image datasets use the same shape fallback.
        scalar_candidates = [name for name in scalar_pvs_declared if name in data_datasets]
        fallback_scalars = [
            name
            for name, dataset in data_datasets.items()
            if name not in scalar_candidates
            and name not in array_pvs_declared
            and name not in image_pvs_declared
            and dataset.ndim == 1
        ]
        for name in deduplicate(scalar_candidates + fallback_scalars):
            dataset = data_datasets[name]
            if dataset.ndim != 1:
                if strict:
                    raise ValueError(f"Declared scalar PV is not 1D: {name} shape={dataset.shape}")
                continue
            pv = _read_scalar_dataset(name, dataset, valid_mask, excel_meta)
            scalars[pv.name] = pv

        array_candidates = [name for name in array_pvs_declared if name in data_datasets]
        fallback_arrays = [
            name
            for name, dataset in data_datasets.items()
            if name not in scalar_pvs_declared
            and name not in array_candidates
            and name not in image_pvs_declared
            and dataset.ndim >= 2
        ]
        for name in deduplicate(array_candidates + fallback_arrays):
            dataset = data_datasets[name]
            if dataset.ndim < 2:
                if strict:
                    raise ValueError(f"Declared array PV is not at least 2D: {name} shape={dataset.shape}")
                continue
            pv_name = _dataset_pv_name(name, dataset)
            if is_coordinate_array_pv(pv_name):
                role = f"coordinate:{coordinate_role_for_pv(pv_name)}"
                pv = _read_array_dataset(name, dataset, valid_mask, excel_meta, role)
                coordinates[pv.name] = pv
            else:
                pv = _read_array_dataset(name, dataset, valid_mask, excel_meta, "trace")
                traces[pv.name] = pv

        # Attach coordinate arrays to matching trace PVs.
        for pv in traces.values():
            coordinate_name = match_coordinate_pv(pv.name, coordinates)
            if coordinate_name:
                coordinate = coordinates[coordinate_name]
                pv.coordinate_name = coordinate.name
                pv.coordinate_role = coordinate_role_for_pv(coordinate.name)
                pv.coordinate_data = coordinate.data
                pv.coordinate_attrs = coordinate.attrs

        timestamps = _read_timestamp_datasets(h5_file, valid_mask, excel_meta)

        images: dict[str, ImageData] = {}
        for name, group in image_groups.items():
            try:
                images[name] = load_image_stack(name, group, image_pvs_declared, scalars)
            except Exception:
                if strict:
                    raise
                continue

    full_system_mask = compute_full_system_mask(scalars)
    pv_manifest = build_pv_manifest(
        scalars=scalars,
        traces=traces,
        coordinates=coordinates,
        timestamps=timestamps,
        scalar_pvs_declared=scalar_pvs_declared,
        array_pvs_declared=array_pvs_declared,
        image_pvs_declared=image_pvs_declared,
    )
    image_manifest = build_image_manifest(images, image_pvs_declared)
    conversion_metadata = {
        "schema_version": SCHEMA_VERSION,
        "domain_manifest": DOMAIN_MANIFEST,
        "file_info": file_info,
        "num_requested_samples": int(n_requested),
        "num_valid_samples": int(source_indices.size),
        "valid_sample_rule": (
            "Prefer nonzero phoeniX:epoch; fallback PNG:ShotNumber/RunNumber; "
            "fallback image frame count; fallback all rows."
        ),
    }

    return LaserH5Model(
        h5_path=h5_path,
        file_info=file_info,
        root_attrs=root_attrs,
        n_requested=n_requested,
        valid_sample_mask=valid_mask,
        source_indices=source_indices,
        scalar_pvs_declared=scalar_pvs_declared,
        array_pvs_declared=array_pvs_declared,
        image_pvs_declared=image_pvs_declared,
        scalars=scalars,
        traces=traces,
        coordinates=coordinates,
        images=images,
        timestamps=timestamps,
        full_system_mask=full_system_mask,
        pv_manifest=pv_manifest,
        image_manifest=image_manifest,
        conversion_metadata=conversion_metadata,
    )


def model_summary(model: LaserH5Model) -> dict[str, Any]:
    """Return a compact JSON-safe summary useful for dry runs/tests."""
    return {
        "h5_path": str(model.h5_path),
        "file_info": model.file_info,
        "num_requested_samples": model.n_requested,
        "num_valid_samples": model.num_valid_samples,
        "num_scalars": len(model.scalars),
        "num_traces": len(model.traces),
        "num_coordinates": len(model.coordinates),
        "num_images": len(model.images),
        "num_timestamps": len(model.timestamps),
        "declared_image_pvs": model.image_pvs_declared,
        "present_image_pvs": sorted(model.images.keys()),
        "missing_image_pvs": model.image_manifest.get("missing_image_pvs", []),
        "has_full_system_mask": model.full_system_mask is not None,
        "trace_examples": list(model.traces.keys())[:5],
        "coordinate_examples": list(model.coordinates.keys())[:5],
        "scalar_examples": list(model.scalars.keys())[:5],
    }

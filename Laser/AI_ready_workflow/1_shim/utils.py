"""Shared utilities for the Laser openPMD/ADIOS2 shim.

This module intentionally contains low-level helpers only: path handling,
HDF5/string/JSON conversion, valid-sample inference, and small openPMD write
helpers. Laser-specific metadata interpretation lives in metadata.py.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np


SCHEMA_VERSION = "0.2.0"

LASER_FILENAME_RE = re.compile(
    r"^(?P<file_kind>png|alignment)-(?P<run_id>[^-]+)-"
    r"(?P<run_date>\d{4}-\d{2}-\d{2})(?P<run_suffix>.*)$",
    re.IGNORECASE,
)

LASER_FILE_ROLES = {
    "png": "full_system_peening",
    "alignment": "alignment_run",
    "target": "target_run",
}

# Default NERSC location for the Laser data campaign.
DEFAULT_NERSC_LASER_ROOT = Path("/global/cfs/cdirs/m3239/2026_FES-AmSC/data/laser")


def get_laser_root() -> Path:
    """Return the Laser data root directory.
    On NERSC this defaults to /global/cfs/cdirs/m3239/2026_FES-AmSC/data/laser.
    Override with LASER_ROOT for local testing or alternate layouts.
    """
    return Path(os.environ.get("LASER_ROOT", str(DEFAULT_NERSC_LASER_ROOT))).expanduser()


def get_laser_data_dir() -> Path:
    """Return the directory containing Laser .h5 files and documentation.
    Older local layouts used LASER_ROOT/Data. The NERSC layout places dated folders and documentation directly under LASER_ROOT. 
    Prefer LASER_ROOT/Data only when it exists; otherwise use LASER_ROOT itself.
    """
    root = get_laser_root()
    legacy_data = root / "Data"
    return legacy_data if legacy_data.exists() else root


def get_default_documentation_dir() -> Path:
    """Return the default documentation directory for PV spreadsheets/notes."""
    return Path(os.environ.get("LASER_DOCUMENTATION_DIR", str(get_laser_data_dir() / "documentation"))).expanduser()


def get_default_out_dir() -> Path:
    """Return the default BP5 output directory."""
    return Path(os.environ.get("LASER_BP_OUT_DIR", str(get_laser_data_dir() / "bp_output"))).expanduser()



# -----------------------------------------------------------------------------
# General path/string helpers
# -----------------------------------------------------------------------------


def find_first_existing(paths: Iterable[str | os.PathLike[str] | None]) -> str | None:
    """Return the first path that exists, or None if no candidate exists."""
    for path in paths:
        if path and Path(path).exists():
            return str(path)
    return None


def sanitize_pv_name(pv_name: str) -> str:
    """Convert an EPICS PV name into a safe openPMD mesh path component."""
    name = (pv_name or "").strip()
    name = name.replace("/", "_").replace(":", "__").replace(" ", "_")
    name = name.replace(".", "_")
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "pv"
    if name[0].isdigit():
        name = f"_{name}"
    return name


# Logical UCLA Phoenix namespaces used for iteration-level openPMD mesh records.
# Prefixes are ordered from most specific to least specific.
PHOENIX_NAMESPACE_PREFIXES: tuple[tuple[str, str], ...] = (
    ("PNG-Lumina-digitizer:", "digitizer/pngLumina"),
    ("PNG-PulsedPower-digitizer:", "digitizer/pngPulsedPower"),
    ("PNG-PP-digitizer:", "digitizer/pngPulsedPower"),
    ("LAPD-TS-digitizer:", "digitizer/lapdTs"),
    ("TS-digitizer:", "digitizer/ts"),
    ("PNG-digitizer:", "digitizer/png"),
    ("PNG:FE:Chiller:", "thermal/frontEndChiller"),
    ("PNG:AQ:", "environment/phoenixAirQuality"),
    ("LB:AQ:", "environment/laserBayAirQuality"),
    ("TA:AQ:", "environment/targetAreaAirQuality"),
    ("PNG:Air", "thermal/laser/air"),
    ("PNG:Breadboard", "thermal/laser/breadboard"),
    ("PNG:Accelerometer:", "vibration/laser"),
    ("Target:Accelerometer:", "vibration/target"),
    ("TGT:Accelerometer:", "vibration/target"),
    ("PNG:CoolingFans:", "thermal/coolingFans"),
    ("PNG:FabryPerot:", "diagnostic/fabryPerot"),
    ("PNG:ModeTab:", "laser/modeTable"),
    ("PNG:QT3:", "laser/qt3"),
    ("PNG:QT9:", "laser/qt9"),
    ("PNG:PP:", "laser/pulsedPower"),
    ("PNG:Etalon:", "laser/etalon"),
    ("PNG:Vacuum:", "vacuum/laserTransport"),
    ("BNC1:", "timing/bnc1"),
    ("BNC2:", "timing/bnc2"),
    ("BNC3:", "timing/bnc3"),
    ("BNC4:", "timing/bnc4"),
    ("BNC5:", "timing/bnc5"),
    ("CAEN1:", "digitizer/caen1"),
    ("CAEN2:", "digitizer/caen2"),
    ("Siglent2:", "oscilloscope/siglent2"),
    ("Siglent:", "oscilloscope/siglent"),
    ("LeCroy:", "oscilloscope/lecroy"),
    ("MDO:", "oscilloscope/mdo"),
    ("Scientech:", "diagnostic/scientech"),
    ("OceanOptics:", "diagnostic/oceanOptics"),
    ("SPEX:", "diagnostic/spex"),
    ("KSMO:", "diagnostic/ksmo"),
    ("TS:", "diagnostic/thomsonScattering"),
    ("PhoenixAirQuality:", "environment/phoenixAirQuality"),
    ("phoeniX:AirQuality:", "environment/phoenixAirQuality"),
    ("LaserBayAirQuality:", "environment/laserBayAirQuality"),
    ("TargetAreaAirQuality:", "environment/targetAreaAirQuality"),
    ("Westwood:", "environment/westwood"),
    ("Vacuum:Gas:", "vacuum/gasSystem"),
    ("Vacuum:PressureControl:", "vacuum/pressureControl"),
    ("Vacuum:", "vacuum/targetChamber"),
    ("PNG:Scan:", "scan"),
    ("Scan:", "scan"),
    ("Chiller:", "thermal/facilityChiller"),
)

EVENT_IDENTIFIER_PVS = frozenset(
    {
        "PNG:RunNumber",
        "PNG:ShotNumber",
        "phoeniX:epoch",
        "epoch",
        "dT_this_acquisition",
    }
)


def _schema_token(value: str) -> str:
    """Normalize one source-PV segment while retaining readable camel case."""
    words = [word for word in re.split(r"[^A-Za-z0-9]+", value or "") if word]
    if not words:
        return "value"
    if len(words) == 1:
        word = words[0]
        return word[:1].lower() + word[1:]
    first = words[0].lower()
    return first + "".join(word[:1].upper() + word[1:].lower() for word in words[1:])


def _schema_tail(value: str) -> str:
    components = [_schema_token(part) for part in re.split(r"[:/]", value) if part]
    return "/".join(component for component in components if component)


def phoenix_logical_mesh_path(pv_name: str, data_role: str = "") -> str:
    """Map one EPICS/HDF5 name to the logical Phoenix mesh namespace.

    Slashes express the logical schema hierarchy. Physical openPMD record
    names are generated separately with underscore delimiters.
    """
    source = (pv_name or "").strip().removesuffix(".timestamp")

    camera_match = re.match(r"^CAM_(PNG|TGT)(\d+):(.*)$", source, re.IGNORECASE)
    if camera_match:
        family = "png" if camera_match.group(1).upper() == "PNG" else "target"
        base = f"camera/{family}{int(camera_match.group(2))}"
        if data_role == "image" or camera_match.group(3).lower().endswith(":image"):
            return f"{base}:image"
        tail = _schema_tail(camera_match.group(3))
        return f"{base}/{tail}" if tail else base

    picam_match = re.match(r"^(?:13)?PICAM(\d+):(.*)$", source, re.IGNORECASE)
    if picam_match:
        base = f"camera/picam{int(picam_match.group(1))}"
        if data_role == "image" or picam_match.group(2).lower().endswith(":image"):
            return f"{base}:image"
        tail = _schema_tail(picam_match.group(2))
        return f"{base}/{tail}" if tail else base

    for prefix, namespace in (
        ("CAM_RPT5:", "camera/rpt5"),
        ("CAM_SLR:", "camera/slr"),
        ("SLR_CAM:", "camera/slr"),
    ):
        if source.lower().startswith(prefix.lower()):
            if data_role == "image" or source.lower().endswith(":image"):
                return f"{namespace}:image"
            tail = _schema_tail(source[len(prefix):])
            return f"{namespace}/{tail}" if tail else namespace

    motor_match = re.match(r"^Motor(\d+):(.*)$", source, re.IGNORECASE)
    if motor_match:
        tail = _schema_tail(motor_match.group(2))
        base = f"target/position/motor{int(motor_match.group(1))}"
        return f"{base}/{tail}" if tail else base

    target_match = re.match(r"^(?:Target|TGT):(.*)$", source, re.IGNORECASE)
    if target_match:
        raw_tail = target_match.group(1)
        lowered = raw_tail.lower()
        if lowered.startswith("delta") or any(token in lowered for token in ("move", "motion", "velocity", "speed")):
            namespace = "target/motion"
        elif any(token in lowered for token in ("position", "motor", "stage", "theta")):
            namespace = "target/position"
        else:
            namespace = "target/state"
        tail = _schema_tail(raw_tail)
        return f"{namespace}/{tail}" if tail else namespace

    for prefix, namespace in PHOENIX_NAMESPACE_PREFIXES:
        if source.lower().startswith(prefix.lower()):
            tail = _schema_tail(source[len(prefix):])
            return f"{namespace}/{tail}" if tail else namespace

    fallback = _schema_tail(source)
    return f"unmapped/{fallback or 'value'}"


def phoenix_mesh_path(pv_name: str, data_role: str = "") -> str:
    """Return a flat, openPMD-compliant record name for one Phoenix quantity."""
    return sanitize_pv_name(phoenix_logical_mesh_path(pv_name, data_role))


def is_dataset(obj: Any) -> bool:
    """Return true when obj is an HDF5 dataset."""
    return isinstance(obj, h5py.Dataset)


def is_group(obj: Any) -> bool:
    """Return true when obj is an HDF5 group."""
    return isinstance(obj, h5py.Group)


def decode_hdf5_value(value: Any) -> str:
    """Decode bytes-like HDF5 values without raising on invalid bytes."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def deduplicate(items: Iterable[str]) -> list[str]:
    """Deduplicate strings while preserving first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = (item or "").strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def parse_laser_filename(path: str | os.PathLike[str]) -> dict[str, str]:
    """Parse regular, alignment, and target Laser source filenames."""
    path_obj = Path(path)
    stem = path_obj.stem
    info = {
        "source_stem": stem,
        "file_kind": "",
        "file_role": "unknown",
        "run_id": "",
        "run_date": "",
        "run_suffix": "",
    }

    match = LASER_FILENAME_RE.match(stem)
    if match:
        file_kind = match.group("file_kind").lower()
        info.update(
            {
                "file_kind": file_kind,
                "file_role": LASER_FILE_ROLES.get(file_kind, "unknown"),
                "run_id": match.group("run_id"),
                "run_date": match.group("run_date"),
                "run_suffix": match.group("run_suffix") or "",
            }
        )
        return info

    stem_lower = stem.lower()
    parent_lower = str(path_obj.parent).lower()
    if "target" in stem_lower or "target_data" in parent_lower:
        info["file_kind"] = "target"
        info["file_role"] = LASER_FILE_ROLES["target"]
    else:
        prefix = stem.split("-", 1)[0].lower()
        if prefix in LASER_FILE_ROLES:
            info["file_kind"] = prefix
            info["file_role"] = LASER_FILE_ROLES[prefix]

    date_match = re.search(r"\d{4}-\d{2}-\d{2}", stem)
    if date_match:
        info["run_date"] = date_match.group(0)

    run_match = re.search(r"run[-_ ]?(\d+)", stem, re.IGNORECASE)
    if run_match:
        info["run_id"] = run_match.group(1)

    return info

def _first_dataset_value(dataset: h5py.Dataset) -> Any | None:
    """Read only the first value needed for output identity discovery."""
    if dataset.size == 0:
        return None
    value = dataset[()] if dataset.ndim == 0 else dataset[0]
    array = np.asarray(value)
    return array.reshape(-1)[0] if array.size else None


def infer_run_number_from_h5(h5_path: str | os.PathLike[str]) -> str | None:
    """Read the run identifier without loading the full acquisition model."""
    with h5py.File(h5_path, "r") as h5_file:
        for candidate in ("PNG:RunNumber", "RunNumber"):
            dataset = h5_file.get(candidate)
            if isinstance(dataset, h5py.Dataset):
                value = _first_dataset_value(dataset)
                if value is not None:
                    break
        else:
            found: list[Any] = []

            def visitor(name: str, obj: Any) -> str | None:
                if not isinstance(obj, h5py.Dataset):
                    return None
                pv_name = decode_hdf5_value(obj.attrs.get("pvname", ""))
                leaf = name.rsplit("/", 1)[-1]
                if leaf in {"PNG:RunNumber", "RunNumber"} or pv_name == "PNG:RunNumber":
                    value = _first_dataset_value(obj)
                    if value is not None:
                        found.append(value)
                        return name
                return None

            h5_file.visititems(visitor)
            if not found:
                return None
            value = found[0]

    value = decode_hdf5_value(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return None
    return str(int(number)) if number.is_integer() else str(number)


def infer_run_date_from_h5(h5_path: str | os.PathLike[str]) -> str | None:
    """Infer an ISO calendar date from the recorded event epoch."""
    with h5py.File(h5_path, "r") as h5_file:
        for candidate in ("phoeniX:epoch", "epoch"):
            dataset = h5_file.get(candidate)
            if not isinstance(dataset, h5py.Dataset):
                continue
            value = _first_dataset_value(dataset)
            try:
                epoch = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(epoch) and epoch > 0:
                return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d")
    return None


def _output_discriminator(
    h5_path: str | os.PathLike[str],
    info: Mapping[str, Any],
    run_date: str,
    *,
    run_id_missing: bool,
) -> str:
    """Preserve only an explicit source part; acquisition type belongs in metadata."""
    del h5_path, run_date, run_id_missing
    run_suffix = str(info.get("run_suffix") or "").strip("_-. ")
    if not run_suffix:
        return ""
    token = re.sub(r"[^A-Za-z0-9]+", "-", run_suffix).strip("-").lower()
    return f"part-{token}" if token else ""

def default_bp_filename(h5_path: str | os.PathLike[str]) -> str:
    """Build a run-oriented BP5 name independent of acquisition mode.

    Canonical inputs use run-<ID>-<date>.bp5. A neutral source discriminator
    is appended only when multiple files can belong to the same run/date.
    """
    info = parse_laser_filename(h5_path)
    run_id = info.get("run_id") or infer_run_number_from_h5(h5_path)
    run_date = info.get("run_date") or infer_run_date_from_h5(h5_path)
    if not run_date:
        raise ValueError(f"Cannot derive run date for output naming: {h5_path}")

    run_id_missing = not bool(run_id)
    if run_id_missing:
        normalized_run_id = "unassigned"
    else:
        normalized_run_id = re.sub(
            r"[^A-Za-z0-9._-]+", "_", str(run_id)
        ).strip("_")

    discriminator = _output_discriminator(
        h5_path,
        info,
        run_date,
        run_id_missing=run_id_missing,
    )
    suffix = f"-{discriminator}" if discriminator else ""
    return f"run-{normalized_run_id}-{run_date}{suffix}.bp5"


def remove_path(path: str | os.PathLike[str]) -> None:
    """Remove a file or directory tree if it exists."""
    path_obj = Path(path)
    if path_obj.is_dir():
        shutil.rmtree(path_obj)
    elif path_obj.exists():
        path_obj.unlink()


def current_timestamp_strings() -> tuple[str, str]:
    """Return local and UTC timestamp strings for provenance attributes."""
    now_local = datetime.now().astimezone()
    now_utc = datetime.now(timezone.utc)
    return (
        now_local.strftime("%Y-%m-%d %H:%M:%S %Z"),
        now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
    )


# -----------------------------------------------------------------------------
# JSON/HDF5 conversion helpers
# -----------------------------------------------------------------------------


def to_jsonable(value: Any) -> Any:
    """Convert NumPy/HDF5-ish values to JSON-serializable Python values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        as_float = float(value)
        if np.isnan(as_float):
            return None
        return as_float
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (bytes, np.bytes_)):
        return decode_hdf5_value(value)
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if value is None:
        return None
    return value


def encode_text_blob(text: str) -> np.ndarray:
    """Encode text as a uint8 array suitable for an openPMD record component."""
    return np.frombuffer((text or "").encode("utf-8"), dtype=np.uint8).copy()


def encode_json_blob(obj: Any) -> np.ndarray:
    """Encode JSON-serializable data as a uint8 array."""
    text = json.dumps(to_jsonable(obj), ensure_ascii=False, indent=2)
    return encode_text_blob(text)


def parse_json_list(value: Any) -> list[str]:
    """Parse a JSON-encoded HDF5 attribute list, returning [] on failure."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = decode_hdf5_value(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [text]
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [str(parsed).strip()] if parsed else []


def h5_attrs_to_dict(attrs: h5py.AttributeManager) -> dict[str, Any]:
    """Return JSON-safe HDF5 attributes."""
    return {str(key): to_jsonable(value) for key, value in attrs.items()}


def get_root_datasets(root: h5py.File) -> dict[str, h5py.Dataset]:
    """Return datasets stored directly at the HDF5 root."""
    return {name: value for name, value in root.items() if is_dataset(value)}


def get_image_groups(root: h5py.File) -> dict[str, h5py.Group]:
    """Return groups containing datasets named image 0, image 1, ..."""
    image_groups: dict[str, h5py.Group] = {}
    for name, value in root.items():
        if not is_group(value):
            continue
        for dataset_name, dataset in value.items():
            if is_dataset(dataset) and re.match(r"^image\s+\d+$", dataset_name):
                image_groups[name] = value
                break
    return image_groups


def infer_requested_sample_count(
    root_attrs: Mapping[str, Any],
    root_datasets: Mapping[str, h5py.Dataset],
) -> int:
    """Infer the requested/source-buffer sample count."""
    if root_attrs.get("N_requested") not in (None, ""):
        try:
            return int(root_attrs["N_requested"])
        except (TypeError, ValueError):
            pass

    first_dims: list[int] = []
    for dataset in root_datasets.values():
        if dataset.ndim >= 1:
            first_dims.append(int(dataset.shape[0]))
    if not first_dims:
        return 0
    return max(set(first_dims), key=first_dims.count)


def infer_valid_sample_mask(
    root_datasets: Mapping[str, h5py.Dataset],
    image_groups: Mapping[str, h5py.Group],
    n_requested: int,
) -> np.ndarray:
    """Infer which rows in the HDF5 sample buffer contain acquired data.

    Preference order:
    1. phoeniX:epoch, because it is the run event/sample clock.
    2. PNG:ShotNumber and PNG:RunNumber, because they are expected to be zero
       in padded rows in the current files.
    3. image frame count, when images are present.
    4. all rows valid.
    """
    if n_requested <= 0:
        return np.zeros(0, dtype=bool)

    candidate_names = ["phoeniX:epoch", "epoch", "PNG:ShotNumber", "PNG:RunNumber"]
    for name in candidate_names:
        dataset = root_datasets.get(name)
        if dataset is None or dataset.ndim != 1 or dataset.shape[0] != n_requested:
            continue
        data = np.asarray(dataset[()])
        if np.issubdtype(data.dtype, np.number):
            mask = np.isfinite(data) & (data != 0)
            if 0 < int(mask.sum()) <= n_requested:
                return mask.astype(bool)

    frame_counts = [count_image_frames(group) for group in image_groups.values()]
    frame_counts = [count for count in frame_counts if count > 0]
    if frame_counts:
        n_valid = min(max(frame_counts), n_requested)
        mask = np.zeros(n_requested, dtype=bool)
        mask[:n_valid] = True
        return mask

    return np.ones(n_requested, dtype=bool)


def count_image_frames(group: h5py.Group) -> int:
    """Return the number of image <index> datasets in an image group."""
    return len(image_dataset_indices(group))


def image_dataset_indices(group: h5py.Group) -> list[int]:
    """Return sorted integer frame indices for datasets named image <index>."""
    indices: list[int] = []
    for name, dataset in group.items():
        if not is_dataset(dataset):
            continue
        match = re.match(r"^image\s+(\d+)$", name)
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)


def trim_to_valid_samples(data: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Trim an array's first axis to valid sample rows when possible."""
    if data.ndim >= 1 and data.shape[0] == valid_mask.shape[0]:
        return np.asarray(data[valid_mask])
    return np.asarray(data)


# -----------------------------------------------------------------------------
# openPMD helpers
# -----------------------------------------------------------------------------


def _openpmd_api():
    """Import openpmd_api lazily so metadata-only tests do not need it."""
    try:
        import openpmd_api as io  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "openpmd_api is required to write BP5 output. Install openPMD-api "
            "in the conversion environment before running laser_to_adios."
        ) from exc
    return io


def make_series(out_bp_path: str | os.PathLike[str]):
    """Create an openPMD Series using ADIOS2 BP5."""
    io = _openpmd_api()
    adios2_config = r'''
    {
      "iteration_encoding": "variable_based",
      "adios2": {
        "engine": {
          "type": "bp5",
          "parameters": {
            "StatsLevel": "1"
          }
        }
      }
    }
    '''
    return io.Series(str(out_bp_path), io.Access_Type.create_linear, adios2_config)


def setup_mesh(mesh: Any, axis_labels: Sequence[str]) -> None:
    """Set common mesh geometry metadata."""
    io = _openpmd_api()
    mesh.geometry = io.Geometry.cartesian
    mesh.axis_labels = list(axis_labels)
    mesh.grid_spacing = [1.0] * len(axis_labels)
    mesh.grid_global_offset = [0.0] * len(axis_labels)


def write_record(mesh: Any, record_name: str, data: np.ndarray) -> None:
    """Write one NumPy array into one openPMD record component."""
    io = _openpmd_api()
    array = np.asarray(data)
    if array.dtype.kind in {"O", "S", "U"}:
        array = encode_json_blob(to_jsonable(array))
    array = np.ascontiguousarray(array)
    record = mesh[record_name]
    record.reset_dataset(io.Dataset(array.dtype, array.shape))
    record.store_chunk(array)


def set_attr_safe(target: Any, key: str, value: Any) -> None:
    """Set an openPMD attribute after converting to supported scalar/string values."""
    if value is None:
        return
    value = to_jsonable(value)
    if isinstance(value, float) and np.isnan(value):
        return
    if isinstance(value, (dict, list, tuple)):
        target.set_attribute(key, json.dumps(value, ensure_ascii=False))
    elif isinstance(value, (str, int, float, bool)):
        target.set_attribute(key, value)
    else:
        target.set_attribute(key, str(value))


def set_attrs(target: Any, attrs: Mapping[str, Any]) -> None:
    """Set many openPMD attributes safely."""
    for key, value in attrs.items():
        set_attr_safe(target, str(key), value)


def write_array_mesh(
    iteration: Any,
    mesh_path: str,
    record_name: str,
    data: np.ndarray,
    axis_labels: Sequence[str],
    attrs: Mapping[str, Any] | None = None,
) -> Any:
    """Create a mesh, write one record component, and attach attributes."""
    mesh = iteration.meshes[mesh_path]
    setup_mesh(mesh, axis_labels)
    write_record(mesh, record_name, data)
    if attrs:
        set_attrs(mesh, attrs)
    return mesh


def write_json_mesh(
    iteration: Any,
    mesh_path: str,
    obj: Any,
    label: str,
    attrs: Mapping[str, Any] | None = None,
) -> Any:
    """Write a JSON object as a UTF-8 uint8 mesh."""
    merged_attrs = {
        "label": label,
        "encoding": "utf-8",
        "contentType": "application/json",
    }
    if attrs:
        merged_attrs.update(dict(attrs))
    return write_array_mesh(
        iteration=iteration,
        mesh_path=mesh_path,
        record_name="json",
        data=encode_json_blob(obj),
        axis_labels=["byte"],
        attrs=merged_attrs,
    )

# -----------------------------------------------------------------------------
# Laser model -> openPMD writer helpers
# -----------------------------------------------------------------------------


EPICS_ATTR_MAP = {
    "pvname": "pv:name",
    "description": "hdf5:description",
    "units": "unitLabel",
    "precision": "epics:precision",
    "lower_ctrl_limit": "epics:lower_ctrl_limit",
    "upper_ctrl_limit": "epics:upper_ctrl_limit",
    "lower_alarm_limit": "epics:lower_alarm_limit",
    "upper_alarm_limit": "epics:upper_alarm_limit",
    "lower_warning_limit": "epics:lower_warning_limit",
    "upper_warning_limit": "epics:upper_warning_limit",
    "lower_disp_limit": "epics:lower_disp_limit",
    "upper_disp_limit": "epics:upper_disp_limit",
    "enum_strs": "epics:enum_strs",
    "type": "epics:type",
    "count": "epics:count",
    "host": "epics:host",
    "access": "epics:access",
}


def pv_mesh_attrs(pv: Any, data_role: str, axis_name: str = "sample") -> dict[str, Any]:
    """Build common openPMD mesh attributes for a PVData-like object."""
    h5_attrs = getattr(pv, "attrs", {}) or {}
    excel = getattr(pv, "excel", {}) or {}
    description = h5_attrs.get("description") or excel.get("excel:description") or f"Laser {data_role} PV"
    pv_name = getattr(pv, "name", "")
    attrs: dict[str, Any] = {
        "description": description,
        "pv:name": pv_name,
        "sourcePV": pv_name,
        "schemaPath": phoenix_logical_mesh_path(pv_name, data_role),
        "dataRole": data_role,
        "sampleAxis": 0 if axis_name == "sample" else None,
        "metadata:sources": [source for source, present in [("hdf5", bool(h5_attrs)), ("excel", bool(excel))] if present],
        "semantic:roles": getattr(pv, "semantic_roles", []),
    }
    for h5_key, openpmd_key in EPICS_ATTR_MAP.items():
        if h5_key in h5_attrs:
            attrs[openpmd_key] = h5_attrs[h5_key]
    for excel_key, value in excel.items():
        attrs[excel_key] = value
    if "unitLabel" not in attrs or attrs.get("unitLabel") in (None, ""):
        attrs["unitLabel"] = "unknown"
        attrs["unitStatus"] = "unmapped"
    else:
        attrs["unitStatus"] = "from_hdf5"
    return attrs


def _metadata_value_is_empty(value: Any) -> bool:
    """Return whether a value should be omitted from series metadata."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, Mapping)):
        return len(value) == 0
    return False


def flatten_metadata_tree(
    tree: Mapping[str, Any],
    prefix: str = "",
) -> dict[str, Any]:
    """Flatten a metadata tree into colon-namespaced ADIOS attributes.

    openPMD Series metadata are attributes rather than arbitrary groups. The
    colon-delimited names retain the agreed logical hierarchy when the BP file
    is inspected while remaining valid custom openPMD attributes.
    """
    flattened: dict[str, Any] = {}
    for raw_key, value in tree.items():
        key = str(raw_key).strip(":")
        if not key:
            continue
        path = f"{prefix}:{key}" if prefix else key
        if isinstance(value, Mapping):
            flattened.update(flatten_metadata_tree(value, path))
        elif not _metadata_value_is_empty(value):
            flattened[path] = value
    return flattened


def _metadata_path_component(value: Any) -> str:
    """Return a stable attribute-path component for an instrument/PV name."""
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return component or "unknown"


def _iso_utc_from_epoch(value: float) -> str:
    return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _run_time_bounds(model: Any) -> tuple[str, str, str]:
    """Infer run time bounds from an epoch scalar when one is available."""
    for pv_name, values in model.scalars.items():
        if str(pv_name).lower().split(":")[-1] != "epoch":
            continue
        array = np.asarray(getattr(values, "data", values)).reshape(-1)
        numeric = np.asarray(array, dtype=np.float64)
        finite = numeric[np.isfinite(numeric) & (numeric > 0.0)]
        if finite.size:
            return _iso_utc_from_epoch(finite.min()), _iso_utc_from_epoch(finite.max()), str(pv_name)
    return "", "", ""


def _declared_pv_names(model: Any) -> list[str]:
    names = (
        list(model.scalar_pvs_declared)
        + list(model.array_pvs_declared)
        + list(model.image_pvs_declared)
        + list(model.scalars)
        + list(model.traces)
        + list(model.coordinates)
        + list(model.images)
        + list(model.timestamps)
    )
    return sorted({str(name) for name in names})


def _normalize_contributors(value: Any) -> dict[str, Mapping[str, Any]]:
    """Normalize optional documentation contributors to indexed groups."""
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items() if isinstance(item, Mapping)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return {
            f"{index:04d}": item
            for index, item in enumerate(value)
            if isinstance(item, Mapping)
        }
    return {}


def build_series_metadata(
    model: Any,
    documentation_meta: Mapping[str, Any],
    *,
    created_at: str,
) -> dict[str, Any]:
    """Build the UCLA Phoenix series-level metadata tree.

    Only values that describe the complete Phoenix run/HDF5 file belong
    here. Event-varying PV values remain iteration-level mesh records.
    """
    h5_path = Path(model.h5_path)
    file_info = dict(model.file_info)
    root_attrs = dict(model.root_attrs)
    pv_names = _declared_pv_names(model)

    try:
        source_size = int(h5_path.stat().st_size)
    except OSError:
        source_size = None

    author_name = os.environ.get("LASER_AUTHOR_NAME", "Chandreyee Bhowmick")
    author_email = os.environ.get("LASER_AUTHOR_EMAIL", "ccb@ornl.gov")
    author_affiliation = os.environ.get(
        "LASER_AUTHOR_AFFILIATION",
        "Oak Ridge National Laboratory",
    )
    author_roles = [
        role.strip()
        for role in os.environ.get(
            "LASER_AUTHOR_ROLES",
            "schemaAuthor,conversionAuthor",
        ).split(",")
        if role.strip()
    ]

    file_role = str(file_info.get("file_role", "unknown"))
    stem_lower = h5_path.stem.lower()
    if file_role == "alignment_run" or "alignment" in stem_lower:
        run_mode = "alignment"
        experiment_class = "alignment"
    elif "target" in stem_lower or "target" in str(h5_path.parent).lower():
        run_mode = "target"
        experiment_class = "targetExperiment"
    elif file_role == "full_system_peening":
        run_mode = "shot"
        experiment_class = "laserPeening"
    else:
        run_mode = "unknown"
        experiment_class = "unknown"

    start_time, end_time, timestamp_source = _run_time_bounds(model)
    run_number = _event_identifier(model, "PNG:RunNumber", 0) if model.num_valid_samples else None
    if run_number is None:
        run_number = file_info.get("run_id")
    run_identifier = f"UCLA-PHOENIX:{run_number if run_number is not None else h5_path.stem}"
    source_created_at = root_attrs.get("created_iso") or root_attrs.get("createdAt")
    conversion_rule = model.conversion_metadata.get("valid_sample_rule", "")

    def component(prefix: str) -> dict[str, Any]:
        return {
            "sourcePVPrefix": prefix,
            "presentInSource": any(name.startswith(prefix) for name in pv_names),
        }

    def components(prefixes: Sequence[str]) -> dict[str, Any]:
        return {
            "sourcePVPrefixes": list(prefixes),
            "presentInSource": any(
                name.startswith(prefix) for name in pv_names for prefix in prefixes
            ),
        }

    diagnostics: dict[str, Any] = {}
    for pv_name in sorted(set(model.image_pvs_declared) | set(model.images)):
        diagnostics[_metadata_path_component(pv_name)] = {
            "sourcePV": str(pv_name),
            "instrumentType": "camera",
            "calibration": {"status": "notProvided"},
        }

    contributors = _normalize_contributors(documentation_meta.get("contributors"))
    scan_metadata = documentation_meta.get("scan")
    if not isinstance(scan_metadata, Mapping):
        scan_metadata = {"present": False}
    run_metadata = documentation_meta.get("run")
    if not isinstance(run_metadata, Mapping):
        run_metadata = {}

    return {
        "provenance": {
            "schema": {
                "name": "UCLA Phoenix laser schema",
                "version": SCHEMA_VERSION,
                "uri": os.environ.get("LASER_SCHEMA_URI", ""),
                "compatibleWith": ["openPMD-1.1.0"],
                "metadataPathConvention": "adiosColonNamespacedAttributes",
            },
            "author": {
                "name": author_name,
                "email": author_email,
                "affiliation": author_affiliation,
                "orcid": os.environ.get("LASER_AUTHOR_ORCID", ""),
                "role": author_roles,
            },
            "contributors": contributors,
            "sourceFiles": {
                "0000": {
                    "role": file_role if file_role != "unknown" else "primaryRunFile",
                    "name": h5_path.name,
                    "uri": h5_path.resolve().as_uri(),
                    "format": "HDF5",
                    "formatVersion": root_attrs.get("hdf5_version"),
                    "sizeBytes": source_size,
                    "createdAt": source_created_at,
                    "fileKind": file_info.get("file_kind"),
                    "fileRole": file_role,
                    "runDate": file_info.get("run_date"),
                    "runSuffix": file_info.get("run_suffix"),
                }
            },
            "sourceSoftware": {
                "name": root_attrs.get("software") or "UCLA Phoenix acquisition system",
                "version": root_attrs.get("software_version"),
                "script": root_attrs.get("script_name"),
                "scriptPath": root_attrs.get("script_path"),
                "repository": root_attrs.get("repository"),
                "commit": root_attrs.get("commit"),
                "host": root_attrs.get("hostname"),
                "user": root_attrs.get("user"),
            },
            "transformation": {
                "method": "phoenixHDF5ToOpenPMD",
                "startedAt": created_at,
                "completedAt": created_at,
                "configuration": "UCLA Phoenix laser schema",
                "softwareRepository": os.environ.get("LASER_SHIM_REPOSITORY", ""),
                "softwareCommit": os.environ.get("LASER_SHIM_COMMIT", ""),
                "eventCountMethod": conversion_rule,
                "unusedAllocationPolicy": "omit",
                "staleValuePolicy": "preserveWithValidityMask",
                "missingValuePolicy": "omitRecordAndDocumentInManifest",
            },
            "validation": {
                "status": "notValidated",
                "warningCount": 0,
                "errorCount": 0,
            },
        },
        "facility": {
            "name": "UCLA Phoenix Laser Facility",
            "shortName": "Phoenix",
            "institution": "University of California, Los Angeles",
            "institutionIdentifier": "https://ror.org/046rm7j60",
            "systemName": "Phoenix",
            "systemIdentifier": "UCLA-PHOENIX",
            "systemType": "laserPeeningFacility",
            "location": "Los Angeles, California, USA",
        },
        "systemSetting": {
            "configurationScope": "series",
            "laserSubsystems": {
                "qt3": component("PNG:QT3:"),
                "qt9": component("PNG:QT9:"),
                "pulsedPower": component("PNG:PP:"),
                "etalon": component("PNG:Etalon:"),
            },
            "timing": {
                "delayGenerators": {
                    "bnc1": component("BNC1:"),
                    "bnc2": component("BNC2:"),
                }
            },
            "thermalManagement": {
                "facilityChiller": component("Chiller:"),
                "frontEndChiller": component("PNG:FE:Chiller:"),
                "coolingFans": component("PNG:CoolingFans:"),
            },
            "vacuum": {"laserTransport": component("PNG:Vacuum:")},
            "controlSystem": {
                "epics": {
                    "framework": "EPICS",
                    "pvCount": len(pv_names),
                    "pvMetadataSource": documentation_meta.get("pv_metadata_source", ""),
                }
            },
        },
        "experimentSetting": {
            "configurationScope": "series",
            "class": experiment_class,
            "campaign": "2026_FES-AmSC",
            "target": {
                **components(("Target:", "TGT:")),
                "positioning": components(("Target:", "Motor")),
            },
            "environment": {
                "vacuum": component("Vacuum:"),
                "gas": component("Vacuum:Gas:"),
            },
            "focusingOptics": {"mappingStatus": "notMapped"},
            "diagnostics": diagnostics,
        },
        "run": {
            "identifier": run_identifier,
            "number": to_jsonable(run_number),
            "mode": run_mode,
            "purpose": run_metadata.get("purpose"),
            "startTime": start_time,
            "endTime": end_time,
            "timeReference": "UnixEpochUTC" if start_time else "unavailable",
            "timeSource": timestamp_source,
            "triggerPV": root_attrs.get("trigger"),
            "triggerMode": root_attrs.get("trigger_mode") or run_metadata.get("triggerMode"),
            "allocatedEventCount": int(model.n_requested),
            "recordedEventCount": int(model.num_valid_samples),
            "status": "empty" if int(model.num_valid_samples) == 0 else "complete",
            "knownIssues": run_metadata.get("knownIssues", []),
            "description": run_metadata.get("description"),
            "scan": dict(scan_metadata),
        },
    }


def set_series_attributes(
    series: Any,
    model: Any,
    documentation_meta: Mapping[str, Any],
) -> None:
    """Write standard openPMD and UCLA Phoenix series attributes."""
    _creation_date, creation_time_utc = current_timestamp_strings()
    conversion_host = os.environ.get("NERSC_HOST") or os.environ.get("HOSTNAME", "")
    author_name = os.environ.get("LASER_AUTHOR_NAME", "Chandreyee Bhowmick")
    author_email = os.environ.get("LASER_AUTHOR_EMAIL", "ccb@ornl.gov")
    author = f"{author_name} <{author_email}>" if author_email else author_name

    # Required openPMD root attributes are managed by openPMD-api in
    # make_series(). particlesPath is absent because Phoenix has no particles.
    set_attrs(
        series,
        {
            "author": author,
            "software": "UCLA Phoenix laser openPMD shim",
            "softwareVersion": SCHEMA_VERSION,
            "date": creation_time_utc,
            "machine": conversion_host,
            "comment": (
                "UCLA Phoenix laser-facility run converted from HDF5 "
                "to an openPMD Series with ADIOS2 BP5 storage."
            ),
        },
    )

    metadata_tree = build_series_metadata(
        model,
        documentation_meta,
        created_at=creation_time_utc,
    )
    set_attrs(series, flatten_metadata_tree(metadata_tree))
    set_attrs(
        series,
        {
            "provenance:sourceFiles:0000:hdf5RootAttributes": model.root_attrs,
            "provenance:transformation:pvManifest": model.pv_manifest,
            "provenance:transformation:imageManifest": model.image_manifest,
        },
    )


def _run_mode(model: Any) -> str:
    """Return the Phoenix run mode shared by all iterations."""
    role = str(model.file_info.get("file_role", ""))
    source = str(model.h5_path).lower()
    if role == "target_run" or "target_data" in source or "target" in Path(source).stem:
        return "target"
    if role == "alignment_run" or "alignment" in Path(source).stem:
        return "alignment"
    if role == "full_system_peening" or Path(source).stem.startswith("png-"):
        return "shot"
    return "unknown"


def _event_value(pv: Any, event_index: int) -> Any | None:
    data = np.asarray(getattr(pv, "data", []))
    if data.ndim == 0:
        return data.item() if event_index == 0 else None
    if event_index >= data.shape[0]:
        return None
    return data[event_index]


def _event_scalar(model: Any, pv_name: str, event_index: int) -> Any | None:
    pv = model.scalars.get(pv_name)
    return _event_value(pv, event_index) if pv is not None else None


def _event_identifier(model: Any, pv_name: str, event_index: int) -> Any | None:
    """Return a source identifier, preserving integral identifiers as integers."""
    value = _event_scalar(model, pv_name, event_index)
    if value is None:
        return None
    item = np.asarray(value).reshape(-1)[0]
    if isinstance(item, (int, np.integer)):
        return int(item)
    if isinstance(item, (float, np.floating)) and np.isfinite(item) and float(item).is_integer():
        return int(item)
    return to_jsonable(item)


def _finite_float(value: Any) -> float | None:
    try:
        number = float(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return None
    return number if np.isfinite(number) else None


def _event_epoch(model: Any, event_index: int) -> tuple[float | None, str]:
    for pv_name in ("phoeniX:epoch", "epoch"):
        value = _finite_float(_event_scalar(model, pv_name, event_index))
        if value is not None and value > 0:
            return value, pv_name
    return None, ""


def _event_dt(model: Any, event_index: int) -> float:
    """Return elapsed time from the preceding recorded event."""
    if event_index <= 0:
        return 0.0
    current, _ = _event_epoch(model, event_index)
    previous, _ = _event_epoch(model, event_index - 1)
    if current is None or previous is None:
        return 0.0
    delta = current - previous
    return float(delta) if np.isfinite(delta) and delta >= 0 else 0.0


def _event_timestamp_attrs(model: Any, pv_name: str, event_index: int) -> dict[str, Any]:
    for source_name, timestamp_pv in model.timestamps.items():
        if timestamp_pv.name != pv_name and source_name.removesuffix(".timestamp") != pv_name:
            continue
        value = _finite_float(_event_value(timestamp_pv, event_index))
        if value is None:
            return {}
        return {
            "sourceTimestamp": value,
            "sourceTimestampUnit": "s",
            "sourceTimestampReference": "Unix epoch",
            "sourceTimestampDataset": source_name,
        }
    return {}


def _event_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 0:
        array = array.reshape(1)
    return array


def _axis_labels_for(array: np.ndarray, preferred: str = "sample") -> list[str]:
    if array.ndim == 1:
        return [preferred]
    return [preferred] + [f"axis{index}" for index in range(1, array.ndim)]


def write_laser_model_to_openpmd(
    model: Any,
    out_bp_path: str | os.PathLike[str],
    documentation_meta: Mapping[str, Any],
) -> str:
    """Compatibility wrapper for the writer module."""
    from writer import write_laser_model_to_openpmd as _write

    return _write(model, out_bp_path, documentation_meta)

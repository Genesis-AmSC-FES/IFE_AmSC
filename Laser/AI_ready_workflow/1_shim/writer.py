"""openPMD/ADIOS2 BP5 serialization for normalized Phoenix laser runs."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from variable_discovery import discover_variables, variable_manifest
from utils import (
    EVENT_IDENTIFIER_PVS,
    _axis_labels_for,
    _event_array,
    _event_dt,
    _event_epoch,
    _event_identifier,
    _event_scalar,
    _event_timestamp_attrs,
    _event_value,
    _finite_float,
    make_series,
    phoenix_logical_mesh_path,
    phoenix_mesh_path,
    pv_mesh_attrs,
    set_attrs,
    set_series_attributes,
    setup_mesh,
    to_jsonable,
    write_array_mesh,
    write_record,
)


def write_scalar_meshes(iteration: Any, model: Any, event_index: int) -> None:
    """Write event-varying scalar PVs as one-element meshes."""
    for pv in model.scalars.values():
        if pv.name in EVENT_IDENTIFIER_PVS:
            continue
        value = _event_value(pv, event_index)
        if value is None:
            continue
        data = _event_array(value)
        if data.size == 0:
            continue
        attrs = pv_mesh_attrs(pv, "scalar", axis_name="value")
        attrs.update(_event_timestamp_attrs(model, pv.name, event_index))

        write_array_mesh(
            iteration,
            phoenix_mesh_path(pv.name, "scalar"),
            "value",
            data,
            _axis_labels_for(data, "value"),
            attrs,
        )


def write_coordinate_meshes(iteration: Any, model: Any, event_index: int) -> None:
    """Preserve explicit time, frequency, wavelength, and bin coordinates."""
    for pv in model.coordinates.values():
        value = _event_value(pv, event_index)
        if value is None:
            continue
        data = _event_array(value)
        if data.size == 0:
            continue
        role = (getattr(pv, "data_role", "coordinate") or "coordinate").split(":", 1)[-1]
        attrs = pv_mesh_attrs(pv, f"coordinate:{role}", axis_name=role)
        attrs.update(_event_timestamp_attrs(model, pv.name, event_index))
        attrs["phoenix:coordinateRole"] = role
        write_array_mesh(
            iteration,
            phoenix_mesh_path(pv.name, "coordinate"),
            "value",
            data,
            _axis_labels_for(data, role),
            attrs,
        )


def write_trace_meshes(iteration: Any, model: Any, event_index: int) -> None:
    """Write one event's waveform/spectrum records and matched coordinates."""
    for pv in model.traces.values():
        value = _event_value(pv, event_index)
        if value is None:
            continue
        signal = _event_array(value)
        if signal.size == 0:
            continue
        coordinate_role = pv.coordinate_role or "sample"
        mesh_path = phoenix_mesh_path(pv.name, "trace")
        mesh = iteration.meshes[mesh_path]
        setup_mesh(mesh, _axis_labels_for(signal, coordinate_role))
        write_record(mesh, "signal", signal)

        attrs = pv_mesh_attrs(pv, "trace", axis_name=coordinate_role)
        attrs.update(_event_timestamp_attrs(model, pv.name, event_index))


        if pv.coordinate_data is not None and pv.coordinate_name is not None:
            coordinate = _event_value(type("Coordinate", (), {"data": pv.coordinate_data})(), event_index)
            if coordinate is not None:
                coordinate_array = _event_array(coordinate)
                if coordinate_array.shape == signal.shape:
                    component = pv.coordinate_role or "coordinate"
                    if component == "histogram_bin":
                        component = "bin"
                    write_record(mesh, component, coordinate_array)
                    attrs["phoenix:coordinateComponent"] = component
                else:
                    attrs["phoenix:coordinateShapeMismatch"] = {
                        "signal": list(signal.shape),
                        "coordinate": list(coordinate_array.shape),
                    }
            attrs["phoenix:coordinateSourcePV"] = pv.coordinate_name
            attrs["coordinateRecord"] = phoenix_mesh_path(
                pv.coordinate_name,
                "coordinate",
            )
            attrs["coordinateSchemaPath"] = phoenix_logical_mesh_path(
                pv.coordinate_name,
                "coordinate",
            )
        set_attrs(mesh, attrs)


def write_image_meshes(iteration: Any, model: Any, event_index: int) -> None:
    """Write the camera frame aligned with this event."""
    for image in model.images.values():
        if event_index >= image.data.shape[0]:
            continue
        frame = _event_array(image.data[event_index])
        if frame.size == 0:
            continue
        mesh_path = phoenix_mesh_path(image.name, "image")
        attrs: dict[str, Any] = {
            "description": "Camera image aligned with this event",
            "pv:name": image.name,
            "dataRole": "image",
            "unitLabel": "counts",
            "sourcePV": image.name,
            "schemaPath": phoenix_logical_mesh_path(image.name, "image"),
            "semantic:roles": image.semantic_roles,
            "frameIndex": int(image.frame_index[event_index]),
            "declaredInRootImagePVs": bool(image.declared),
            "timestamp:meaning": "script_read_attempt_time_not_camera_exposure_time",
        }
        timestamp = _finite_float(image.timestamp[event_index])
        if timestamp is not None:
            attrs.update(
                {
                    "sourceTimestamp": timestamp,
                    "sourceTimestampUnit": "s",
                    "sourceTimestampReference": "Unix epoch",
                }
            )
        if image.array_counter is not None and event_index < image.array_counter.shape[0]:
            attrs["arrayCounter"] = to_jsonable(image.array_counter[event_index])
            attrs["arrayCounterPV"] = image.array_counter_pv
        if image.new_frame_mask is not None and event_index < image.new_frame_mask.shape[0]:
            attrs["newFrame"] = bool(image.new_frame_mask[event_index])
        write_array_mesh(
            iteration,
            mesh_path,
            "value",
            frame,
            ["y", "x"] if frame.ndim == 2 else _axis_labels_for(frame, "pixel"),
            attrs,
        )


def _source_file_attributes(models: Sequence[Any]) -> dict[str, Any]:
    """Describe every HDF5 segment that contributes events to this run."""
    attrs: dict[str, Any] = {}
    for source_index, model in enumerate(models):
        source = Path(model.h5_path)
        prefix = f"provenance:sourceFiles:{source_index:04d}"
        try:
            size_bytes = int(source.stat().st_size)
        except OSError:
            size_bytes = None
        attrs.update(
            {
                f"{prefix}:role": "primaryRunFile" if len(models) == 1 else "runRolloverSegment",
                f"{prefix}:name": source.name,
                f"{prefix}:uri": source.resolve().as_uri(),
                f"{prefix}:format": "HDF5",
                f"{prefix}:sizeBytes": size_bytes,
                f"{prefix}:segmentIndex": source_index,
                f"{prefix}:allocatedEventCount": int(model.n_requested),
                f"{prefix}:recordedEventCount": int(model.num_valid_samples),
                f"{prefix}:fileKind": model.file_info.get("file_kind"),
                f"{prefix}:fileRole": model.file_info.get("file_role"),
                f"{prefix}:runDate": model.file_info.get("run_date"),
                f"{prefix}:runSuffix": model.file_info.get("run_suffix"),
                f"{prefix}:hdf5RootAttributes": model.root_attrs,
                f"{prefix}:pvManifest": model.pv_manifest,
                f"{prefix}:imageManifest": model.image_manifest,
            }
        )
    return attrs


def _combined_variable_manifest(models: Sequence[Any]) -> list[dict[str, str]]:
    """Return the union of variables discovered across rollover segments."""
    combined: list[dict[str, str]] = []
    seen: set[str] = set()
    for model in models:
        for entry in variable_manifest(discover_variables(model)):
            key = json.dumps(entry, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            combined.append(entry)
    return combined


def _series_time_bounds(models: Sequence[Any]) -> tuple[str, str]:
    epochs: list[float] = []
    for model in models:
        for event_index in range(model.num_valid_samples):
            event_time, _ = _event_epoch(model, event_index)
            if event_time is not None:
                epochs.append(float(event_time))
    if not epochs:
        return "", ""
    render = lambda value: datetime.fromtimestamp(value, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    return render(min(epochs)), render(max(epochs))


def write_laser_models_to_openpmd(
    models: Sequence[Any],
    out_bp_path: str | os.PathLike[str],
    documentation_meta: Mapping[str, Any],
) -> str:
    """Serialize all HDF5 rollover segments for one run into one BP5 Series.

    Every recorded source event becomes one openPMD iteration. Iteration indices
    remain continuous across source-file boundaries, while sourceEventIndex and
    sourceFileIndex preserve the original location of every event.
    """
    models = list(models)
    if not models:
        raise ValueError("At least one Laser HDF5 model is required")

    variable_catalog = _combined_variable_manifest(models)
    total_allocated = sum(int(model.n_requested) for model in models)
    total_recorded = sum(int(model.num_valid_samples) for model in models)
    start_time, end_time = _series_time_bounds(models)

    series = make_series(out_bp_path)
    set_series_attributes(series, models[0], documentation_meta)
    aggregate_attrs: dict[str, Any] = {
        "provenance:transformation:discoveredVariableCount": len(variable_catalog),
        "provenance:transformation:variableManifest": variable_catalog,
        "provenance:transformation:rolloverFilesStacked": len(models) > 1,
        "run:sourceFileCount": len(models),
        "run:rolloverSegmentCount": len(models),
        "run:allocatedEventCount": total_allocated,
        "run:recordedEventCount": total_recorded,
        "run:validEventCount": total_recorded,
        "run:unusedAllocatedEventCount": max(total_allocated - total_recorded, 0),
        "run:status": "empty" if total_recorded == 0 else "complete",
    }
    if start_time:
        aggregate_attrs["run:startTime"] = start_time
        aggregate_attrs["run:endTime"] = end_time
        aggregate_attrs["run:timeReference"] = "UnixEpochUTC"
    aggregate_attrs.update(_source_file_attributes(models))
    set_attrs(series, aggregate_attrs)

    global_event_index = 0
    previous_event_time: float | None = None
    for source_file_index, model in enumerate(models):
        source_name = Path(model.h5_path).name
        for source_local_index in range(model.num_valid_samples):
            iteration = series.iterations[global_event_index]
            event_time, _time_source = _event_epoch(model, source_local_index)
            iteration.time = float(event_time) if event_time is not None else 0.0
            if (
                event_time is not None
                and previous_event_time is not None
                and float(event_time) >= previous_event_time
            ):
                iteration.dt = float(event_time) - previous_event_time
            else:
                iteration.dt = _event_dt(model, source_local_index)
            iteration.time_unit_SI = 1.0

            iteration_attrs: dict[str, Any] = {
                "eventIndex": int(global_event_index),
                "sourceEventIndex": int(model.source_indices[source_local_index]),
                "sourceFileIndex": int(source_file_index),
                "sourceFileName": source_name,
            }
            identifiers = {
                "shotNumber": "PNG:ShotNumber",
                "recorderEpoch": "epoch",
                "eventAcquisitionDuration": "dT_this_acquisition",
            }
            for attribute, pv_name in identifiers.items():
                value = (
                    _event_identifier(model, pv_name, source_local_index)
                    if attribute == "shotNumber"
                    else _event_scalar(model, pv_name, source_local_index)
                )
                if attribute == "recorderEpoch":
                    recorder_epoch = _finite_float(value)
                    if recorder_epoch is None or recorder_epoch <= 0.0:
                        continue
                    value = recorder_epoch
                if value is not None:
                    iteration_attrs[attribute] = to_jsonable(value)
            if (
                model.full_system_mask is not None
                and source_local_index < model.full_system_mask.shape[0]
            ):
                iteration_attrs["fullSystemShot"] = bool(
                    model.full_system_mask[source_local_index]
                )
            set_attrs(iteration, iteration_attrs)

            write_scalar_meshes(iteration, model, source_local_index)
            write_coordinate_meshes(iteration, model, source_local_index)
            write_trace_meshes(iteration, model, source_local_index)
            write_image_meshes(iteration, model, source_local_index)
            iteration.close()

            if event_time is not None:
                previous_event_time = float(event_time)
            global_event_index += 1

    series.close()
    return str(out_bp_path)


def write_laser_model_to_openpmd(
    model: Any,
    out_bp_path: str | os.PathLike[str],
    documentation_meta: Mapping[str, Any],
) -> str:
    """Compatibility wrapper for a run stored in one HDF5 file."""
    return write_laser_models_to_openpmd([model], out_bp_path, documentation_meta)

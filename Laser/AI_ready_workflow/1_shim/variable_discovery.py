"""Runtime discovery of variables present in a UCLA Phoenix HDF5 run.

The source files are intentionally heterogeneous: shot, alignment, and target
runs do not have to contain the same PVs. This module builds a catalog only
from variables actually present in the normalized LaserH5Model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from utils import EVENT_IDENTIFIER_PVS, phoenix_logical_mesh_path, phoenix_mesh_path


@dataclass(frozen=True)
class DiscoveredVariable:
    """One source variable and its openPMD destination."""

    source_name: str
    role: str
    mesh_name: str
    schema_path: str
    value: Any

    def manifest_entry(self) -> dict[str, str]:
        return {
            "sourcePV": self.source_name,
            "dataRole": self.role,
            "meshName": self.mesh_name,
            "schemaPath": self.schema_path,
        }


def discover_variables(model: Any) -> list[DiscoveredVariable]:
    """Discover and classify only variables available in this run.

    Unknown PV families are retained under the unmapped schema namespace;
    absence of a known variable is not an error. Physical mesh names are
    validated for collisions before the BP5 Series is created.
    """
    collections = (
        ("scalar", model.scalars),
        ("trace", model.traces),
        ("coordinate", model.coordinates),
        ("image", model.images),
    )
    discovered: list[DiscoveredVariable] = []
    owners: dict[str, str] = {}

    for role, collection in collections:
        for value in collection.values():
            source_name = str(value.name)
            if role == "scalar" and source_name in EVENT_IDENTIFIER_PVS:
                continue
            mesh_name = phoenix_mesh_path(source_name, role)
            previous = owners.get(mesh_name)
            if previous is not None and previous != source_name:
                raise ValueError(
                    f"Phoenix schema collision at {mesh_name!r}: "
                    f"{previous!r} and {source_name!r}"
                )
            owners[mesh_name] = source_name
            discovered.append(
                DiscoveredVariable(
                    source_name=source_name,
                    role=role,
                    mesh_name=mesh_name,
                    schema_path=phoenix_logical_mesh_path(source_name, role),
                    value=value,
                )
            )

    return discovered


def variable_manifest(variables: list[DiscoveredVariable]) -> list[dict[str, str]]:
    """Return a serializable catalog for series-level provenance."""
    return [variable.manifest_entry() for variable in variables]

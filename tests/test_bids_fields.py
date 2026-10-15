"""Sidecar fields match BIDS 1.11 (BEP 020 as merged)."""

from __future__ import annotations

from eye2bids._base import BasePhysioEventsJson, BasePhysioJson

# Fields from BEP 020 drafts that are not in the merged specification
REMOVED_FIELDS = [
    "EyeCameraSettings",
    "FeatureDetectionSettings",
    "GazeMappingSettings",
    "ScreenAOIDefinition",
    "StopTime",
]


def test_physioevents_json_onset_source():
    sidecar = BasePhysioEventsJson()
    assert sidecar["OnsetSource"] == "timestamp"
    assert "ForeignIndexColumn" not in sidecar


def test_physio_json_has_no_removed_fields():
    metadata = dict.fromkeys(REMOVED_FIELDS, "x")
    metadata["SampleCoordinateSystem"] = "gaze-on-screen"
    sidecar = BasePhysioJson(manufacturer="SR-Research", metadata=metadata)
    assert sidecar["SampleCoordinateSystem"] == "gaze-on-screen"
    for field in REMOVED_FIELDS:
        assert field not in sidecar

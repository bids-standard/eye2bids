"""Base classes for sidecar and events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eye2bids.logger import eye2bids_logger

e2b_log = eye2bids_logger()


class BasePhysioEventsJson(dict[str, Any]):
    """Handle content of physioevents sidedar."""

    input_file: Path
    two_eyes: bool
    TaskName: bool

    def __init__(self, metadata: None | dict[str, Any] = None) -> None:

        self["Columns"] = ["onset", "duration", "trial_type", "blink", "message"]
        self["Description"] = "Messages logged by the measurement device"
        self["ForeignIndexColumn"] = "timestamp"

        self["blink"] = {
            "Description": "One indicates if the eye was closed, zero if open."
        }
        self["message"] = {"Description": "String messages logged by the eye-tracker."}
        self["trial_type"] = {
            "Description": (
                "Event type as identified by the eye-tracker's model "
                "((either 'n/a' if not applicabble, 'fixation', or 'saccade')."
            )
        }

        self.update_from_metadata(metadata)

    def update_from_metadata(self, metadata: None | dict[str, Any] = None) -> None:
        """Update content of json side car based on metadata."""
        if metadata is None:
            return None

        self["InstitutionAddress"] = metadata.get("InstitutionAddress")
        self["InstitutionName"] = metadata.get("InstitutionName")
        self["StimulusPresentation"] = {
            "ScreenDistance": metadata.get("ScreenDistance"),
            "ScreenRefreshRate": metadata.get("ScreenRefreshRate"),
            "ScreenSize": metadata.get("ScreenSize"),
        }
        self["TaskName"] = metadata.get("TaskName")

    def output_filename(self, recording: str | None = None) -> str:
        """Generate output filename."""
        filename = self.input_file.stem
        if recording is not None:
            return f"{filename}_recording-{recording}_physioevents.json"
        return f"{filename}_physioevents.json"

    def write(
        self,
        output_dir: Path,
        recording: str | None = None,
        extra_metadata: dict[str, str | list[str] | list[float]] | None = None,
    ) -> None:
        """Write to json."""
        if extra_metadata is not None:
            for key, value in extra_metadata.items():
                self[key] = value

        content = {key: value for key, value in self.items() if self[key] is not None}
        with open(output_dir / self.output_filename(recording=recording), "w") as outfile:
            json.dump(content, outfile, indent=4)


class BasePhysioJson(dict[str, Any]):
    """Handle content of physio sidedar."""

    input_file: Path
    has_validation: bool
    two_eyes: bool

    def __init__(self, manufacturer: str, metadata: dict[str, Any] | None = None) -> None:

        self["Manufacturer"] = manufacturer

        self["Columns"] = ["x_coordinate", "y_coordinate", "pupil_size", "timestamp"]
        self["timestamp"] = {
            "Description": (
                "Timestamp issued by the eye-tracker "
                "indexing the continuous recordings "
                "corresponding to the sampled eye."
            )
        }
        self["x_coordinate"] = {
            "Description": (
                "Gaze position x-coordinate of the recorded eye, "
                "in the coordinate units specified "
                "in the corresponding metadata sidecar."
            ),
            "Units": "a.u.",
        }
        self["y_coordinate"] = {
            "Description": (
                "Gaze position y-coordinate of the recorded eye, "
                "in the coordinate units specified "
                "in the corresponding metadata sidecar."
            ),
            "Units": "a.u.",
        }
        self["pupil_size"] = {
            "Description": (
                "Pupil area of the recorded eye as calculated "
                "by the eye-tracker in arbitrary units "
                "(see EyeLink's documentation for conversion)."
            ),
            "Units": "a.u.",
        }

        self.update_from_metadata(metadata)

    def update_from_metadata(self, metadata: None | dict[str, Any] = None) -> None:
        """Update content of json side car based on metadata."""
        if metadata is None:
            return None

        self["EnvironmentCoordinates"] = metadata.get("EnvironmentCoordinates")
        self["SoftwareVersion"] = metadata.get("SoftwareVersion")
        self["EyeCameraSettings"] = metadata.get("EyeCameraSettings")
        self["EyeTrackerDistance"] = metadata.get("EyeTrackerDistance")
        self["FeatureDetectionSettings"] = metadata.get("FeatureDetectionSettings")
        self["GazeMappingSettings"] = metadata.get("GazeMappingSettings")
        self["RawDataFilters"] = metadata.get("RawDataFilters")
        self["SampleCoordinateSystem"] = metadata.get("SampleCoordinateSystem")
        self["SampleCoordinateUnits"] = metadata.get("SampleCoordinateUnits")
        self["ScreenAOIDefinition"] = metadata.get("ScreenAOIDefinition")

    def output_filename(self, recording: str | None = None) -> str:
        """Generate output filename."""
        filename = self.input_file.stem
        if recording is not None:
            return f"{filename}_recording-{recording}_physio.json"
        return f"{filename}_physio.json"

    def write(
        self,
        output_dir: Path,
        recording: str | None = None,
        extra_metadata: dict[str, str | list[str] | list[float]] | None = None,
    ) -> None:
        """Write to json."""
        if extra_metadata is not None:
            for key, value in extra_metadata.items():
                self[key] = value

        content = {key: value for key, value in self.items() if self[key] is not None}
        with open(output_dir / self.output_filename(recording=recording), "w") as outfile:
            json.dump(content, outfile, indent=4)

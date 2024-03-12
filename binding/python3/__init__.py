from typing import Union
from collections.abc import Mapping, MutableMapping
from copy import deepcopy

import numpy as np

from . import ezc3d
from ._version import __version__


# This is a dummy class that is used as an interface for the group of the parameters
class _GroupParameter:
    def __init__(self, data):
        self.__dict__ = data


class C3dMapper(Mapping):
    def __init__(self, *args, **kw):
        self._storage = dict(*args, **kw)
        return

    def __getitem__(self, key):
        return self._storage[key]

    def __iter__(self):
        return iter(self._storage)

    def __len__(self):
        return len(self._storage)

    def keys(self):
        return self._storage.keys()

    def __eq__(self, other):
        # If the dimensions are wrong, then they are not equal
        if "EZC3D" in self._storage and not "EZC3D" in other._storage:
            if len(self._storage) != len(other._storage) + 1:
                return False
        elif not "EZC3D" in self._storage and "EZC3D" in other._storage:
            if len(self._storage) + 1 != len(other._storage):
                return False
        elif len(self._storage) != len(other._storage):
            return False

        # Check for each element (skipping the EZC3D element if exists)
        for key in self._storage:
            if key == "":
                # If empty
                continue
            elif key == "EZC3D":
                # All keys must be equal except for EZC3D
                continue
            elif not key in other._storage:
                return False

            if isinstance(self._storage[key], C3dMapper) and isinstance(other._storage[key], C3dMapper):
                # If it is a c3d, the child is also a C3DMapper. Recursively call __eq__
                return self._storage[key] == other._storage[key]
            elif isinstance(self._storage[key], dict) and isinstance(other._storage[key], dict):
                if not self.__eq_param__(key, self._storage[key], other._storage[key]):
                    return False
            elif isinstance(self._storage[key], np.ndarray) and isinstance(other._storage[key], np.ndarray):
                try:
                    np.testing.assert_array_equal(self._storage[key], other._storage[key])
                    return True
                except AssertionError:
                    return False
            else:
                # Otherwise it is unknown data, therefore assume they are different
                return False
        return True

    @staticmethod
    def __eq_param__(group, dict1, dict2):
        if len(dict1) != len(dict2):
            return False
        for key in dict1:
            if isinstance(dict1[key], (int, float, str)) and isinstance(dict2[key], (int, float, str)):
                if dict1[key] != dict2[key]:
                    return False
                continue

            if group == "POINT" and key == "DATA_START":
                # POINT:DATA_START is a special key which should not be compared
                continue
            if (
                "value" in dict1[key]
                and "value" in dict2[key]
                and isinstance(dict1[key]["value"], (np.ndarray))
                and isinstance(dict2[key]["value"], (np.ndarray))
                and not dict1[key]["value"].any()
                and not dict2[key]["value"].any()
            ):
                # When data are empty type INT or FLOAT is irrelevant
                if (dict1[key]["type"] == 1 or dict1[key]["type"] == 2) and (
                    dict2[key]["type"] == 1 or dict2[key]["type"] == 2
                ):
                    continue
            if not key in dict2:
                return False
            if "type" in dict1[key] and dict1[key]["type"] > 0:
                if (
                    dict1[key]["description"] != dict2[key]["description"]
                    or dict1[key]["is_locked"] != dict2[key]["is_locked"]
                    or not np.all(np.equal(dict1[key]["value"], dict2[key]["value"]))
                ):
                    return False
            else:
                if dict1[key] != dict2[key]:
                    return False
        return True


class C3dMutableMapper(C3dMapper):
    def __init__(self):
        super(C3dMutableMapper, self).__init__()
        return

    def __setitem__(self, key, value):
        self._storage[key] = value

    def __delitem__(self, key):
        del self._storage[key]


class c3d(C3dMapper):
    def __init__(self, path="", extract_forceplat_data=False, ignore_bad_formatting=False):
        super(c3d, self).__init__()

        # Interface to swig pointers
        if path == "":
            self.c3d_swig = ezc3d.c3d()
        else:
            self.c3d_swig = ezc3d.c3d(path, ignore_bad_formatting)

        rotations_info = ezc3d.RotationsInfo(self.c3d_swig)

        self.extract_forceplat_data = extract_forceplat_data
        self._storage["header"] = c3d.Header(self.c3d_swig.header(), rotations_info)
        self._storage["parameters"] = c3d.Parameter(self.c3d_swig.parameters())
        self._storage["data"] = c3d.Data(self.c3d_swig, self.extract_forceplat_data)
        return
    
    @property
    def header(self):
        return self._storage["header"]
    
    @property
    def parameters(self):
        return self._storage["parameters"]
    
    @property
    def data(self):
        return self._storage["data"]

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        # Create a valid structure
        new = c3d()
        rotations_info = ezc3d.RotationsInfo(self.c3d_swig)
        new.extract_forceplat_data = self.extract_forceplat_data

        new._storage["header"] = c3d.Header(new.c3d_swig.header(), rotations_info)
        new._storage["parameters"] = c3d.Parameter(new.c3d_swig.parameters())
        new._storage["data"] = c3d.Data(new.c3d_swig, new.extract_forceplat_data)

        # Update the structure with a copy of all data
        for header_key in self["header"]:
            for value_key in self["header"][header_key]:
                new["header"][header_key][value_key] = deepcopy(self["header"][header_key][value_key])
        for group_key in self["parameters"]:
            new["parameters"][group_key] = deepcopy(self["parameters"][group_key])
        for data_key in self["data"]:
            new["data"][data_key] = deepcopy(self["data"][data_key])

        return new

    class Header(C3dMapper):
        def __init__(self, swig_header, rotation_info):
            super().__init__()

            # Interface to swig pointers
            self.header = swig_header

            self._storage["points"] = {
                "size": self.header.nb3dPoints(),
                "frame_rate": self.header.frameRate(),
                "first_frame": self.header.firstFrame(),
                "last_frame": self.header.lastFrame(),
            }
            self._storage["analogs"] = {
                "size": self.header.nbAnalogs(),
                "frame_rate": self.header.nbAnalogByFrame() * self.header.frameRate(),
                "first_frame": self.header.nbAnalogByFrame() * self.header.firstFrame(),
                "last_frame": self.header.nbAnalogByFrame() * (self.header.lastFrame() + 1) - 1,
            }
            self._storage["rotations"] = {
                "size": rotation_info.used(),
                "frame_rate": self.header.frameRate() * rotation_info.ratio(),
                "first_frame": rotation_info.ratio() * self.header.firstFrame(),
                "last_frame": rotation_info.ratio() * (self.header.lastFrame() + 1) - 1,
            }
            self._storage["events"] = {
                "size": len(self.header.eventsTime()),
                "events_time": self.header.eventsTime(),
                "events_label": self.header.eventsLabel(),
            }
            self._storage.keys()
            return

    class Parameter(C3dMutableMapper):
        def __init__(self, swig_param):
            super().__init__()

            # Interface to swig pointers
            self.parameters = swig_param

            for group in self.parameters.groups():
                group_name = group.name()
                self.create_group_if_needed(group_name)
                self._storage[group_name]["__METADATA__"]["DESCRIPTION"] = group.description()
                self._storage[group_name]["__METADATA__"]["IS_LOCKED"] = group.isLocked()

                # Add easy accessor to the group 
                setattr(self, group_name, _GroupParameter(self._storage[group_name]))
  
                for parameter in group.parameters():
                    self.add_parameter(group_name, parameter)

                    # There is no need to add an easy accessor to the parameter as it is implicit by the fact that it is added to the KEYS

            return

        def create_group_if_needed(self, group_name):
            # If the group does not exist create it
            if group_name not in self._storage:
                self._storage[group_name] = dict()

                # Add the meta data of the group
                self._storage[group_name]["__METADATA__"] = dict()
                self._storage[group_name]["__METADATA__"]["DESCRIPTION"] = ""
                self._storage[group_name]["__METADATA__"]["IS_LOCKED"] = False

        def add_parameter(self, group_name, param_ezc3d):
            self.create_group_if_needed(group_name)
            param = dict()
            param["type"] = param_ezc3d.type()
            param["description"] = param_ezc3d.description()
            param["is_locked"] = param_ezc3d.isLocked()
            if param_ezc3d.type() == ezc3d.BYTE:
                value = np.array(param_ezc3d.valuesAsByte(), dtype="int").reshape(param_ezc3d.dimension(), order="F")
            elif param_ezc3d.type() == ezc3d.INT:
                value = np.array(param_ezc3d.valuesAsInt(), dtype="int").reshape(param_ezc3d.dimension(), order="F")
            elif param_ezc3d.type() == ezc3d.FLOAT:
                value = np.array(param_ezc3d.valuesAsDouble()).reshape(param_ezc3d.dimension(), order="F")
            elif param_ezc3d.type() == ezc3d.CHAR:
                table = param_ezc3d.valuesAsString()
                value = []
                for element in table:
                    value.append(element)
            else:
                raise RuntimeError("Data type not recognized")
            param["value"] = value

            param_name = param_ezc3d.name()
            if param_name not in self._storage[group_name]:
                self._storage[group_name][param_name] = dict()
            self._storage[group_name][param_name] = param

    class PlatForm(C3dMapper):
        def __init__(self, swig_pf):
            super().__init__()

            self._storage["unit_force"] = swig_pf.forceUnit()
            self._storage["unit_moment"] = swig_pf.momentUnit()
            self._storage["unit_position"] = swig_pf.positionUnit()

            self._storage["cal_matrix"] = swig_pf.calMatrix().to_array()
            self._storage["corners"] = np.ndarray((3, len(swig_pf.corners())))
            for i, corner in enumerate(swig_pf.corners()):
                self._storage["corners"][:, i] = corner.to_array()[:, 0]
            self._storage["origin"] = swig_pf.origin().to_array()[:, 0]

            n_frame = swig_pf.nbFrames()
            forces = swig_pf.forces()
            moments = swig_pf.moments()
            cop = swig_pf.CoP()
            Tz = swig_pf.Tz()
            self._storage["force"] = np.ndarray((3, n_frame))
            self._storage["moment"] = np.ndarray((3, n_frame))
            self._storage["center_of_pressure"] = np.ndarray((3, n_frame))
            self._storage["Tz"] = np.ndarray((3, n_frame))
            for i in range(n_frame):
                self._storage["force"][:, i] = forces[i].to_array()[:, 0]
                self._storage["moment"][:, i] = moments[i].to_array()[:, 0]
                self._storage["center_of_pressure"][:, i] = cop[i].to_array()[:, 0]
                self._storage["Tz"][:, i] = Tz[i].to_array()[:, 0]

    class Data(C3dMutableMapper):
        def __init__(self, swig_c3d, extract_forceplat_data):
            super().__init__()

            # Interface to swig pointers
            self.data = swig_c3d.data()

            self._storage["points"] = swig_c3d.get_points()
            self._storage["meta_points"] = {
                "residuals": swig_c3d.get_point_residuals(),
                "camera_masks": swig_c3d.get_point_camera_masks(),
            }
            self._storage["analogs"] = swig_c3d.get_analogs()

            self._storage["rotations"] = swig_c3d.get_rotations()

            # Add the platform filer if required
            if extract_forceplat_data:
                all_pf = []
                for pf in ezc3d.ForcePlatforms(swig_c3d).forcePlatforms():
                    all_pf.append(c3d.PlatForm(pf))
                self._storage["platform"] = all_pf
            return
        
        @property
        def points(self):
            return self._storage["points"]
        
        @property
        def meta_points(self):
            return self._storage["meta_points"]
        
        @property
        def analogs(self):
            return self._storage["analogs"]
        
        @property
        def rotations(self):
            return self._storage["rotations"]

    def add_parameter(
        self,
        group_name: str,
        parameter_name: str,
        value: Union[list, tuple, np.ndarray, int, float, str],
        description: str = "",
    ):
        """
        Create the parameter properly using the ezc3d API

        :param group_name: The name of the group
        :param parameter_name: The name of the parameter
        :param value: The value the parameter takes
        :param description: The description of the parameter
        """

        param_ezc3d = ezc3d.Parameter(parameter_name, description)
        if isinstance(value, (list, tuple)):
            value = np.array(value)
            if np.issubdtype(value.dtype, np.integer):
                value = value.astype(np.float64)

        if isinstance(value, np.ndarray):
            param_ezc3d.set(value.reshape(-1, order="F"), value.shape)
        else:
            param_ezc3d.set(value)
        self._storage["parameters"].add_parameter(group_name, param_ezc3d)

    def add_event(
        self,
        time: list | tuple,
        context: str = "",
        label: str = "",
        description: str = "",
        subject: str = "",
        icon_id: int = 0,
        generic_flag: int = 0,
    ):
        """
        This function adds an event, warning two events can have the same name (it wont't override it)

        :param time: A list for the time, first element is the time in minute (integer), second is the second (float)
        :param context: The context (usually "Right", "Left" or "General")
        :param label: The name of the event
        :param description: The description of the event
        :param subject: The subject the event is applied to. An empty string is generic or the only subject in the scene
        :param icon_id: The ID of the icon of the event
        :param generic_flag: A generic flag
        """

        if "EVENT" in self["parameters"]:
            event_param = self["parameters"]["EVENT"]
            used = event_param["USED"]["value"].tolist()[0]
            times = event_param["TIMES"]["value"].tolist()
            contexts = event_param["CONTEXTS"]["value"]
            labels = event_param["LABELS"]["value"]
            descriptions = event_param["DESCRIPTIONS"]["value"]
            subjects = event_param["SUBJECTS"]["value"]
            icon_ids = event_param["ICON_IDS"]["value"].tolist()
            generic_flags = event_param["GENERIC_FLAGS"]["value"].tolist()
        else:
            used = 0
            times = [[], []]
            contexts = []
            labels = []
            descriptions = []
            subjects = []
            icon_ids = []
            generic_flags = []

        # Adjust the EVENT group
        used += 1
        times[0] += [time[0]]
        times[1] += [time[1]]
        times = np.array(times)
        contexts += [context]
        labels += [label]
        descriptions += [description]
        subjects += [subject]
        icon_ids += [icon_id]
        generic_flags += [generic_flag]

        # Override the EVENT group
        self.add_parameter("EVENT", "USED", used)
        self.add_parameter("EVENT", "TIMES", times)
        self.add_parameter("EVENT", "CONTEXTS", contexts)
        self.add_parameter("EVENT", "LABELS", labels)
        self.add_parameter("EVENT", "DESCRIPTIONS", descriptions)
        self.add_parameter("EVENT", "SUBJECTS", subjects)
        self.add_parameter("EVENT", "ICON_IDS", icon_ids)
        self.add_parameter("EVENT", "GENERIC_FLAGS", generic_flags)

    def write(self, path: str, *, first_frame_as_zero: bool = False):
        """
        Write a new C3D at path. If any extra parameter is provided, then the non-standard writer is called.
        Please note the resulting C3D may or may not work with third parties

        :param path: The path where to write the file
        :param first_frame_as_zero: If the first frame should be flaged
         as 1 (False, default and starndard) or 0 (True, non-standard)
        """

        # Make sure path is a valid path
        extension = ".c3d"
        if path[-4:] != extension:
            path += extension

        # Check for sanity of the structure
        data_points = self._storage["data"]["points"]
        if len(data_points.shape) != 3:
            raise TypeError("Points should be a numpy with exactly 3 dimensions (XYZ(1) x nPoints x nFrames)")
        nb_point_components = data_points.shape[0]
        nb_points = data_points.shape[1]
        nb_point_frames = data_points.shape[2]
        if nb_point_components < 3 or nb_point_components > 4:
            raise TypeError("Points should be a numpy with first dimension exactly equals to 3 or 4 elements")
        nb_labels = len(self._storage["parameters"]["POINT"]["LABELS"]["value"])
        i = 2
        while f"LABELS{i}" in self._storage["parameters"]["POINT"]:
            nb_labels += len(self._storage["parameters"]["POINT"][f"LABELS{i}"]["value"])
            i += 1
        if nb_points != nb_labels:
            raise ValueError(
                "'c3d['parameters']['POINT']['LABELSX']' must have the same length as " "nPoints of the data."
            )

        if "meta_points" not in self._storage["data"]:
            self._storage["data"]["meta_points"] = {}
        data_meta_points = self._storage["data"]["meta_points"]
        if "residuals" not in data_meta_points or data_meta_points["residuals"].size == 0:
            data_meta_points["residuals"] = np.zeros((1, nb_points, nb_point_frames))
        else:
            if data_meta_points["residuals"].shape[0] != 1:
                raise ValueError(
                    "'c3d['data']['meta_points']['residuals']' must have its first dimension's shape equals to 1.\n"
                    "If you are modifying a pre-existing c3d, it is probably easier to delete "
                    "'c3d['data']['meta_points']' and let ezc3d create a new one by itself"
                )
            if data_meta_points["residuals"].shape[1] != nb_points:
                raise ValueError(
                    "'c3d['data']['meta_points']['residuals']' must have its second dimension's shape equals to the "
                    "number of points.\nIf you are modifying a pre-existing c3d, it is probably easier to delete "
                    "'c3d['data']['meta_points']' and let ezc3d create a new one by itself"
                )
            if data_meta_points["residuals"].shape[2] != nb_point_frames:
                raise ValueError(
                    "'c3d['data']['meta_points']['residuals']' must have its third dimension's shape equals to the "
                    "number of frames.\nIf you are modifying a pre-existing c3d, it is probably easier to delete "
                    "'c3d['data']['meta_points']' and let ezc3d create a new one by itself"
                )
        if "camera_masks" not in data_meta_points or data_meta_points["camera_masks"].size == 0:
            data_meta_points["camera_masks"] = np.zeros((7, nb_points, nb_point_frames), dtype=bool)
        else:
            if data_meta_points["camera_masks"].dtype != np.dtype("bool"):
                raise ValueError("'c3d['data']['meta_points']['camera_masks']' must be of dtype 'bool'.")
            if data_meta_points["camera_masks"].shape[0] != 7:
                raise ValueError(
                    "'c3d['data']['meta_points']['camera_masks']' must have its first dimension's shape equals to 7.\n"
                    "If you are modifying a pre-existing c3d, it is probably easier to delete "
                    "'c3d['data']['meta_points']' and let ezc3d create a new one by itself"
                )
            if data_meta_points["camera_masks"].shape[1] != nb_points:
                raise ValueError(
                    "'c3d['data']['meta_points']['camera_masks']' must have its second dimension's shape equals to the "
                    "number of points.\nIf you are modifying a pre-existing c3d, it is probably easier to delete "
                    "'c3d['data']['meta_points']' and let ezc3d create a new one by itself"
                )
            if data_meta_points["camera_masks"].shape[2] != nb_point_frames:
                raise ValueError(
                    "'c3d['data']['meta_points']['camera_masks']' must have its third dimension's shape equals to the "
                    "number of frames.\nIf you are modifying a pre-existing c3d, it is probably easier to delete "
                    "'c3d['data']['meta_points']' and let ezc3d create a new one by itself"
                )

        data_analogs = self._storage["data"]["analogs"]
        if len(data_analogs.shape) != 3:
            raise TypeError("Analogs should be a numpy with exactly 3 dimensions (1 x nAnalogs x nFrames)")
        nb_analog_components = data_analogs.shape[0]
        nb_analogs = data_analogs.shape[1]
        nb_analog_frames = data_analogs.shape[2]
        if nb_analog_components != 1:
            raise TypeError("Analogs should be a numpy with first dimension exactly equals to 1 element")
        nb_analog_subframes = 0
        if nb_point_frames != 0 and nb_points != 0:
            if self._storage["parameters"]["ANALOG"]["RATE"]["value"][0] == 0:
                if nb_analog_frames % nb_point_frames != 0:
                    raise ValueError("Number of frames of Points and Analogs should be a multiple of an integer")
            else:
                if ~np.isclose(
                    nb_analog_frames * self._storage["parameters"]["POINT"]["RATE"]["value"][0],
                    nb_point_frames * self._storage["parameters"]["ANALOG"]["RATE"]["value"][0]
                ):
                    raise ValueError("Number of frames in the data set must match the analog rate X point frame")

            nb_analog_subframes = int(nb_analog_frames / nb_point_frames)
            self._storage["parameters"]["ANALOG"]["RATE"]["value"] = np.array(
                (nb_analog_subframes * self._storage["parameters"]["POINT"]["RATE"]["value"][0],)
            )
            nb_frames = nb_point_frames
        else:
            nb_frames = nb_analog_frames
            nb_analog_subframes = 1

        nb_labels = len(self._storage["parameters"]["ANALOG"]["LABELS"]["value"])
        i = 2
        while f"LABELS{i}" in self._storage["parameters"]["ANALOG"]:
            nb_labels += len(self._storage["parameters"]["ANALOG"][f"LABELS{i}"]["value"])
            i += 1
        if nb_analogs != nb_labels:
            raise ValueError(
                "'c3d['parameters']['ANALOG']['LABELSX']' must have the same length as nAnalogs of the data. "
            )

        data_rotations = None
        if "rotations" in self._storage["data"]:
            data_rotations = self._storage["data"]["rotations"]
            if len(data_rotations.shape) != 4:
                raise TypeError("Rotations should be a numpy with exactly 4 dimensions (4 x 4 x nRotations x nFrames)")
            if data_rotations.shape[0] != 4 or data_rotations.shape[1] != 4:
                raise TypeError("Rotations should be a numpy with first and second dimension exactly equals to 4 element")
            nb_rotations = data_rotations.shape[2]
            nb_rotations_frames = data_rotations.shape[3]

            # Store the ratio
            if nb_rotations_frames % nb_point_frames != 0:
                raise ValueError("Number of rotations' frame should be an integer multiple of frames")
            self.add_parameter("ROTATION", "RATIO", int(nb_rotations_frames / nb_point_frames))

        # Start from a fresh c3d
        new_c3d = ezc3d.c3d()

        # Fill the header
        new_c3d.header().firstFrame(self._storage["header"]["points"]["first_frame"])

        # Fill the parameters
        groups = self._storage["parameters"]

        # Update some important stuff (names of points and analogs)
        point_labels = groups["POINT"]["LABELS"]["value"]
        i = 2
        while f"LABELS{i}" in groups["POINT"]:
            point_labels.extend(groups["POINT"][f"LABELS{i}"]["value"])
            i += 1
        for point_label in point_labels:
            new_c3d.point(point_label)

        analog_labels = groups["ANALOG"]["LABELS"]["value"]
        i = 2
        while f"LABELS{i}" in groups["ANALOG"]:
            analog_labels.extend(groups["ANALOG"][f"LABELS{i}"]["value"])
            i += 1
        for analog_label in analog_labels:
            new_c3d.analog(analog_label)

        for group in groups:
            # Write the metadata of the group
            if not new_c3d.parameters().isGroup(group):
                new_c3d.parameters().group(ezc3d.Group(group))
            new_c3d.parameters().group(group).description(groups[group]["__METADATA__"]["DESCRIPTION"])
            if groups[group]["__METADATA__"]["IS_LOCKED"]:
                new_c3d.parameters().group(group).lock()
            else:
                new_c3d.parameters().group(group).unlock()

            # Write the parameters of the group
            for param in groups[group]:
                if param == "__METADATA__":
                    continue

                old_param = groups[group][param]
                new_param = ezc3d.Parameter(param)
                dim = [len(old_param["value"])]

                # Copy the parameters into the c3d, but skip those who will be updated automatically later
                if not (
                    (group == "POINT" and param == "USED")
                    or (group == "POINT" and param == "FRAMES")
                    or (group == "POINT" and param == "LABELS")
                    or (group == "POINT" and param == "DESCRIPTIONS" and dim[0] != nb_points)
                    or (group == "ANALOG" and param == "USED")
                    or (group == "ANALOG" and param == "LABELS")
                    or (group == "ANALOG" and param == "SCALE" and len(old_param["value"]) != nb_analogs)
                    or (group == "ANALOG" and param == "OFFSET")
                    or (group == "ANALOG" and param == "UNITS" and len(old_param["value"]) != nb_analogs)
                    or (group == "ANALOG" and param == "DESCRIPTIONS" and dim[0] != nb_analogs)
                ):
                    # Copy data
                    if old_param["type"] == ezc3d.BYTE or old_param["type"] == ezc3d.INT:
                        if isinstance(old_param["value"], np.ndarray):
                            new_param.set(
                                ezc3d.VecInt([int(x) for x in old_param["value"].T.ravel()]), old_param["value"].shape
                            )
                        else:
                            new_param.set(ezc3d.VecInt(old_param["value"]), dim)
                    elif old_param["type"] == ezc3d.FLOAT:
                        if isinstance(old_param["value"], np.ndarray):
                            new_param.set(
                                ezc3d.VecDouble(old_param["value"].T.ravel().astype("float")), old_param["value"].shape
                            )
                        else:
                            new_param.set(ezc3d.VecDouble(old_param["value"]), dim)
                    elif old_param["type"] == ezc3d.CHAR:
                        try:
                            new_param.set(ezc3d.VecString(old_param["value"]), dim)
                        except:
                            raise ValueError(f"Value in parameters {group}:{param} could not be converted to string")
                    else:
                        raise NotImplementedError("Parameter type not implemented yet")
                    new_c3d.parameter(group, new_param)

                # Copy metadata
                new_c3d.parameters().group(group).parameter(param).description(old_param["description"])
                if old_param["is_locked"]:
                    new_c3d.parameters().group(group).parameter(param).lock()
                else:
                    new_c3d.parameters().group(group).parameter(param).unlock()

        # Initialization for speed
        pt = ezc3d.Point()
        pts = ezc3d.Points()
        for i in range(nb_points):
            pts.point(pt)
        c = ezc3d.Channel()
        subframe = ezc3d.AnalogsSubframe()
        for i in range(nb_analogs):
            subframe.channel(c)
        analogs = ezc3d.Analogs()
        for i in range(nb_analog_subframes):
            analogs.subframe(subframe)

        # Fill the data
        new_c3d.import_numpy_data(
            data_points, data_meta_points["residuals"], data_meta_points["camera_masks"], data_analogs, data_rotations
        )

        # Write the file
        if first_frame_as_zero:
            # As soon as at least one non-standard parameter is provided, use the parametrized write
            new_c3d.parametrizedWrite(path, ezc3d.DEFAULT, first_frame_as_zero)
        else:
            new_c3d.write(path)
        return

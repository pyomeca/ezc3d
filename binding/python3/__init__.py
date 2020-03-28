from collections.abc import Mapping, MutableMapping
import numpy as np

from . import ezc3d
from ._version import __version__


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

    def __eq__(self,other):
        # If the dimensions are wrong, then they are not equal
        if 'EZC3D' in self._storage and not 'EZC3D' in other._storage:
            if len(self._storage) != len(other._storage) + 1:
                return False
        elif not 'EZC3D' in self._storage and 'EZC3D' in other._storage:
            if len(self._storage) + 1 != len(other._storage):
                return False
        elif len(self._storage) != len(other._storage):
            return False

        # Check for each element (skipping the EZC3D element if exists)
        for key in self._storage:
            if key == '':
                # If empty
                continue
            elif key == 'EZC3D':
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
            if 'value' in dict1[key] and 'value' in dict2[key] \
                and isinstance(dict1[key]['value'], (np.ndarray)) and isinstance(dict2[key]['value'], (np.ndarray)) \
                and not dict1[key]['value'].any() and not dict2[key]['value'].any():
                # When data are empty type INT or FLOAT is irrelevant
                if (dict1[key]['type'] == 1 or dict1[key]['type'] == 2) \
                    and (dict2[key]['type'] == 1 or dict2[key]['type'] == 2):
                    continue
            if not key in dict2:
                return False
            if 'type' in dict1[key] and dict1[key]['type'] > 0:
                if dict1[key]['description'] != dict2[key]['description'] \
                    or dict1[key]['is_locked'] != dict2[key]['is_locked'] \
                    or not np.all(np.equal(dict1[key]['value'], dict2[key]['value'])):
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
    def __init__(self, path=""):
        super(c3d, self).__init__()

        # Interface to swig pointers
        if path == "":
            self.c3d_swig = ezc3d.c3d()
        else:
            self.c3d_swig = ezc3d.c3d(path)

        self._storage['header'] = c3d.Header(self.c3d_swig.header())
        self._storage['parameters'] = c3d.Parameter(self.c3d_swig.parameters())
        self._storage['data'] = c3d.Data(self.c3d_swig)
        return

    class Header(C3dMapper):
        def __init__(self, swig_header):
            super(c3d.Header, self).__init__()

            # Interface to swig pointers
            self.header = swig_header

            self._storage['points'] = {
                'size': self.header.nb3dPoints(),
                'frame_rate': self.header.frameRate(),
                'first_frame': self.header.firstFrame(),
                'last_frame': self.header.lastFrame()
            }
            self._storage['analogs'] = {
                'size': self.header.nbAnalogs(),
                'frame_rate': self.header.nbAnalogByFrame() * self.header.frameRate(),
                'first_frame': self.header.nbAnalogByFrame() * self.header.firstFrame(),
                'last_frame': self.header.nbAnalogByFrame() * (self.header.lastFrame()+1) - 1
            }
            self._storage['events'] = {
                'size': len(self.header.eventsTime()),
                'events_time': self.header.eventsTime(),
                'events_label': self.header.eventsLabel()
            }
            self._storage.keys()
            return

    class Parameter(C3dMutableMapper):
        def __init__(self, swig_param):
            super(c3d.Parameter, self).__init__()

            # Interface to swig pointers
            self.parameters = swig_param

            for group in self.parameters.groups():
                group_name = group.name()
                self.create_group_if_needed(group_name)
                self._storage[group_name]['__METADATA__']['DESCRIPTION'] = group.description()
                self._storage[group_name]['__METADATA__']['IS_LOCKED'] = group.isLocked()
                for parameter in group.parameters():
                    self.add_parameter(group_name, parameter)
            return

        def create_group_if_needed(self, group_name):
            # If the group does not exist create it
            if group_name not in self._storage:
                self._storage[group_name] = dict()

                # Add the meta data of the group
                self._storage[group_name]['__METADATA__'] = dict()
                self._storage[group_name]['__METADATA__']['DESCRIPTION'] = ""
                self._storage[group_name]['__METADATA__']['IS_LOCKED'] = False

        def add_parameter(self, group_name, param_ezc3d):
            self.create_group_if_needed(group_name)
            param = dict()
            param['type'] = param_ezc3d.type()
            param['description'] = param_ezc3d.description()
            param['is_locked'] = param_ezc3d.isLocked()
            if param_ezc3d.type() == ezc3d.BYTE:
                value = np.array(param_ezc3d.valuesAsByte(), dtype='int').reshape(param_ezc3d.dimension(), order='F')
            elif param_ezc3d.type() == ezc3d.INT:
                value = np.array(param_ezc3d.valuesAsInt(), dtype='int').reshape(param_ezc3d.dimension(), order='F')
            elif param_ezc3d.type() == ezc3d.FLOAT:
                value = np.array(param_ezc3d.valuesAsDouble()).reshape(param_ezc3d.dimension(), order='F')
            elif param_ezc3d.type() == ezc3d.CHAR:
                table = param_ezc3d.valuesAsString()
                value = []
                for element in table:
                    value.append(element)
            param['value'] = value

            param_name = param_ezc3d.name()
            if param_name not in self._storage[group_name]:
                self._storage[group_name][param_name] = dict()
            self._storage[group_name][param_name] = param


    class Data(C3dMutableMapper):
        def __init__(self, swig_c3d):
            super(c3d.Data, self).__init__()

            # Interface to swig pointers
            self.data = swig_c3d.data()

            self._storage['points'] = swig_c3d.get_points()
            self._storage['meta_points'] = {'residuals': swig_c3d.get_point_residuals(), 
                                            'camera_masks': swig_c3d.get_point_camera_masks()
                                           }
            self._storage['analogs'] = swig_c3d.get_analogs()
            return

    def add_parameter(self, group_name, parameter_name, value, description=""):
        # Create the parameter properly using the ezc3d API
        param_ezc3d = ezc3d.Parameter(parameter_name, description)
        param_ezc3d.set(value)
        self._storage['parameters'].add_parameter(group_name, param_ezc3d)

    def write(self, path):
        # Make sure path is a valid path
        extension = ".c3d"
        if path[-4:] != extension:
            path += extension

        # Check for sanity of the structure
        data_points = self._storage['data']['points']
        if len(data_points.shape) != 3:
            raise TypeError("Points should be a numpy with exactly 3 dimensions (XYZ(1) x nPoints x nFrames)")
        nb_point_components = data_points.shape[0]
        nb_points = data_points.shape[1]
        nb_point_frames = data_points.shape[2]
        if nb_point_components < 3 or nb_point_components > 4:
            raise TypeError("Points should be a numpy with first dimension exactly equals to 3 or 4 elements")
        if nb_points != len(self._storage['parameters']['POINT']['LABELS']['value']):
            raise ValueError("'c3d['parameters']['POINT']['LABELS']' must have the same length as nPoints of the data.")
            
        data_meta_points = self._storage['data']['meta_points']
        if data_meta_points['residuals'].size == 0:
            data_meta_points['residuals'] = np.zeros((1, nb_points, nb_point_frames))
        else:
            if data_meta_points['residuals'].shape[0] != 1:
                raise ValueError("'c3d['data']['meta_points']['residuals']' must have its first dimension's shape equals to 1.")
            if data_meta_points['residuals'].shape[1] != nb_points:
                raise ValueError("'c3d['data']['meta_points']['residuals']' must have its second dimension's shape equals to the number of points.")
            if data_meta_points['residuals'].shape[2] != nb_point_frames:
                raise ValueError("'c3d['data']['meta_points']['residuals']' must have its third dimension's shape equals to the number of frames.")
        if data_meta_points['camera_masks'].size == 0:
            data_meta_points['camera_masks'] = np.zeros((7, nb_points, nb_point_frames), dtype=bool)
        else:
            if data_meta_points['camera_masks'].dtype != np.dtype('bool'):
                raise ValueError("'c3d['data']['meta_points']['camera_masks']' must be of dtype 'bool'.")
            if data_meta_points['camera_masks'].shape[0] != 7:
                raise ValueError("'c3d['data']['meta_points']['camera_masks']' must have its first dimension's shape equals to 7.")
            if data_meta_points['camera_masks'].shape[1] != nb_points:
                raise ValueError("'c3d['data']['meta_points']['camera_masks']' must have its second dimension's shape equals to the number of points.")
            if data_meta_points['camera_masks'].shape[2] != nb_point_frames:
                raise ValueError("'c3d['data']['meta_points']['camera_masks']' must have its third dimension's shape equals to the number of frames.")
        

        data_analogs = self._storage['data']['analogs']
        if len(data_analogs.shape) != 3:
            raise TypeError("Analogs should be a numpy with exactly 3 dimensions (1 x nAnalogs x nFrames)")
        nb_analog_components = data_analogs.shape[0]
        nb_analogs = data_analogs.shape[1]
        nb_analog_frames = data_analogs.shape[2]
        if nb_analog_components != 1:
            raise TypeError("Analogs should be a numpy with first dimension exactly equals to 1 element")
        nb_analog_subframes = 0
        if nb_point_frames != 0:
            if self._storage['parameters']['ANALOG']['RATE']['value'][0] == 0:
                if nb_analog_frames % nb_point_frames != 0:
                    raise ValueError("Number of frames of Points and Analogs should be a multiple of an integer")
            else:
                if nb_analog_frames != self._storage['parameters']['ANALOG']['RATE']['value'][0] / \
                        self._storage['parameters']['POINT']['RATE']['value'][0] * nb_point_frames:
                    raise ValueError("Number of frames in the data set must match the analog rate X point frame")

            nb_analog_subframes = int(nb_analog_frames / nb_point_frames)
            self._storage['parameters']['ANALOG']['RATE']['value'] = np.array((
                nb_analog_subframes
                * self._storage['parameters']['POINT']['RATE']['value'][0],
            ))
            nb_frames = nb_point_frames
        else:
            nb_frames = nb_analog_frames
            nb_analog_subframes = 1

        if nb_analogs != len(self._storage['parameters']['ANALOG']['LABELS']['value']):
            raise ValueError("'c3d['parameters']['ANALOG']['LABELS']' must have the same length as "
                             "nAnalogs of the data.")

        # Start from a fresh c3d
        new_c3d = ezc3d.c3d()

        # Fill the header
        new_c3d.header().firstFrame(self._storage['header']['points']['first_frame'])

        # Fill the parameters
        groups = self._storage['parameters']

        # Update some important stuff (names of points and analogs)
        point_labels = groups['POINT']['LABELS']['value']
        for point_label in point_labels:
            new_c3d.point(point_label)

        analog_labels = groups['ANALOG']['LABELS']['value']
        for analog_label in analog_labels:
            new_c3d.analog(analog_label)

        for group in groups:
            # Write the metadata of the group
            if not new_c3d.parameters().isGroup(group):
                new_c3d.parameters().group(ezc3d.Group(group))
            new_c3d.parameters().group(group).description(groups[group]['__METADATA__']['DESCRIPTION'])
            if groups[group]['__METADATA__']['IS_LOCKED']:
                new_c3d.parameters().group(group).lock()
            else:
                new_c3d.parameters().group(group).unlock()

            # Write the parameters of the group
            for param in groups[group]:
                if param == '__METADATA__':
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
                            new_param.set(ezc3d.VecInt(
                                [int(x) for x in old_param["value"].T.ravel()]), old_param["value"].shape)
                        else:
                            new_param.set(ezc3d.VecInt(old_param["value"]), dim)
                    elif old_param["type"] == ezc3d.FLOAT:
                        if isinstance(old_param["value"], np.ndarray):
                            new_param.set(ezc3d.VecDouble(
                                old_param["value"].T.ravel().astype('float')), old_param["value"].shape)
                        else:
                            new_param.set(ezc3d.VecDouble(old_param["value"]), dim)
                    elif old_param["type"] == ezc3d.CHAR:
                        new_param.set(ezc3d.VecString(old_param["value"]), dim)
                    else:
                        raise NotImplementedError("Parameter type not implemented yet")
                    new_c3d.parameter(group, new_param)

                # Copy metadata
                new_c3d.parameters().group(group).parameter(param).description(old_param['description'])
                if old_param['is_locked']:
                    new_c3d.parameters().group(group).parameter(param).lock()
                else:
                    new_c3d.parameters().group(group).parameter(param).unlock()

        # Initialization for speed
        pt = ezc3d.Point()
        pts = ezc3d.Points()
        for i in range(nb_points):
            pts.point(pt)
        c = ezc3d.Channel()
        subframe = ezc3d.SubFrame()
        for i in range (nb_analogs):
            subframe.channel(c)
        analogs = ezc3d.Analogs()
        for i in range (nb_analog_subframes):
            analogs.subframe(subframe)

        # Fill the data
        for f in range(nb_frames):
            for i in range(nb_points):
                if np.isnan(data_points[:, i, f]).any():
                    pt.set(0, 0, 0, -1)
                else:
                    pt.set(data_points[0, i, f], data_points[1, i, f], data_points[2, i, f])
                    pt.residual(data_meta_points['residuals'][0, i, f])
                    pt.cameraMask(data_meta_points['camera_masks'][:, i, f].tolist())
                pts.point(pt, i)

            for sf in range(nb_analog_subframes):
                for i in range(nb_analogs):
                    c.data(data_analogs[0, i, nb_analog_subframes*f + sf])
                    subframe.channel(c, i)
                analogs.subframe(subframe, sf)
            frame = ezc3d.Frame()
            frame.add(pts, analogs)
            new_c3d.frame(frame)

        # Write the file
        new_c3d.write(path)
        return


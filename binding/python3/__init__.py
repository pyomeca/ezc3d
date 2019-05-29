from collections.abc import Mapping, MutableMapping

from . import ezc3d


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


class C3dMutableMapper(C3dMapper):
    def __init__(self):
        super(C3dMutableMapper, self).__init__()
        return

    def __setitem__(self, key, value):
        self._storage[key] = value

    def __delitem__(self, key):
        del self._storage[key]

    def __iter__(self):
        return iter(self._storage)


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
                'events_label': self.header.eventsLabel()  # TODO mapping of std::vector<std::string>
            }
            self._storage.keys()
            return

    class Parameter(C3dMutableMapper):
        def __init__(self, swig_param):
            super(c3d.Parameter, self).__init__()

            # Interface to swig pointers
            self.parameters = swig_param

            for group in self.parameters.groups():
                for parameter in group.parameters():
                    self.add_parameter(group.name(), parameter)
            return

        def add_parameter(self, group_name, param_ezc3d):
            # If the group does not exist create it
            if group_name not in self._storage:
                self._storage[group_name] = dict()

            param = dict()
            param['type'] = param_ezc3d.type()
            param['description'] = param_ezc3d.description()
            if param_ezc3d.type() == ezc3d.BYTE:
                value = param_ezc3d.valuesAsByte()
            elif param_ezc3d.type() == ezc3d.INT:
                value = param_ezc3d.valuesAsInt()
            elif param_ezc3d.type() == ezc3d.FLOAT:
                value = param_ezc3d.valuesAsFloat()
            elif param_ezc3d.type() == ezc3d.CHAR:
                table = param_ezc3d.valuesAsString()
                value = []
                for element in table:
                    value.append(element)
            param['value'] = value

            if param_ezc3d.name() not in self._storage[group_name]:
                self._storage[group_name][param_ezc3d.name()] = dict()
            self._storage[group_name][param_ezc3d.name()] = param


    class Data(C3dMutableMapper):
        def __init__(self, swig_c3d):
            super(c3d.Data, self).__init__()

            # Interface to swig pointers
            self.data = swig_c3d.data()

            self._storage['points'] = swig_c3d.get_points()
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
            self._storage['parameters']['ANALOG']['RATE']['value'] = (
                nb_analog_subframes
                * self._storage['parameters']['POINT']['RATE']['value'][0],
            )
            nb_frames = nb_point_frames
        else:
            nb_frames = nb_analog_frames
            nb_analog_subframes = 1

        if nb_analogs != len(self._storage['parameters']['ANALOG']['LABELS']['value']):
            raise ValueError("'c3d['parameters']['ANALOG']['LABELS']' must have the same length as "
                             "nAnalogs of the data.")

        # Start from a fresh c3d
        new_c3d = ezc3d.c3d()

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
            for param in groups[group]:
                # Copy the parameters into the c3d, but skip those who will be updated automatically later
                if (
                    not (group == "POINT" and param == "USED")
                    and (not (group == "POINT" and param == "FRAMES"))
                    and (not (group == "POINT" and param == "LABELS"))

                    and (not (group == "ANALOG" and param == "USED"))
                    and (not (group == "ANALOG" and param == "LABELS"))
                    and (not (group == "ANALOG" and param == "SCALE"))
                    and (not (group == "ANALOG" and param == "OFFSET"))
                    and (not (group == "ANALOG" and param == "UNITS"))
                ):
                    old_param = groups[group][param]
                    new_param = ezc3d.Parameter(param)
                    dim = [len(old_param["value"])]

                    # Special cases
                    if group == "POINT" and param == "DESCRIPTIONS" and dim[0] != nb_points:
                        continue
                    if group == "ANALOG" and param == "DESCRIPTIONS" and dim[0] != nb_analogs:
                        continue

                    if old_param["type"] == ezc3d.BYTE or old_param["type"] == ezc3d.INT:
                        new_param.set(ezc3d.VecInt(old_param["value"]), dim)
                    elif old_param["type"] == ezc3d.FLOAT:
                        new_param.set(ezc3d.VecFloat(old_param["value"]), dim)
                    elif old_param["type"] == ezc3d.CHAR:
                        new_param.set(ezc3d.VecString(old_param["value"]), dim)
                    else:
                        raise NotImplementedError("Parameter type not implemented yet")
                    new_c3d.parameter(group, new_param)

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
                pt.x(data_points[0, i, f])
                pt.y(data_points[1, i, f])
                pt.z(data_points[2, i, f])
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




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

            self._storage['markers'] = {
                'size': self.header.nb3dPoints(),
                'frame_rate': self.header.frameRate(),
                'first_frame': self.header.firstFrame(),
                'last_frame': self.header.lastFrame()
            }
            self._storage['analogs'] = {
                'size': self.header.nbAnalogs(),
                'frame_rate': self.header.nbAnalogByFrame() * self.header.frameRate(),
                'first_frame': self.header.nbAnalogByFrame() * self.header.firstFrame()+1,
                'last_frame': self.header.nbAnalogByFrame() * self.header.lastFrame()
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
                self._storage[group.name()] = dict()
                for parameter in group.parameters():
                    self._storage[group.name()][parameter.name()] = dict()
                    self._storage[group.name()][parameter.name()]['type'] = parameter.type()
                    self._storage[group.name()][parameter.name()]['description'] = parameter.description()
                    if parameter.type() == ezc3d.BYTE:
                        value = parameter.valuesAsByte()
                    elif parameter.type() == ezc3d.INT:
                        value = parameter.valuesAsInt()
                    elif parameter.type() == ezc3d.FLOAT:
                        value = parameter.valuesAsFloat()
                    elif parameter.type() == ezc3d.CHAR:
                        table = parameter.valuesAsString()
                        value = []
                        for element in table:
                            value.append(element)
                    self._storage[group.name()][parameter.name()]['value'] = value
            return

    class Data(C3dMapper):
        def __init__(self, swig_c3d):
            super(c3d.Data, self).__init__()

            # Interface to swig pointers
            self.data = swig_c3d.data()

            self._storage['points'] = swig_c3d.get_points()
            self._storage['analogs'] = swig_c3d.get_analogs()
            return

    def write(self, path):
        # Make sure path is a valid path
        extension = ".c3d"
        if path[-4:] != extension:
            path += extension

        # Check for sanity of the structure
        data_points = self._storage['data']['points']
        if len(data_points.shape) != 3:
            raise TypeError("Points should be a numpy with 3 exactly dimensions (XYZ(1) x nPoints x nFrames)")
        nb_point_components = data_points.shape[0]
        nb_points = data_points.shape[1]
        nb_point_frames = data_points.shape[2]
        if nb_point_components < 3 or nb_point_components > 4:
            raise TypeError("Points should be a numpy with first dimension exactly equals to 3 or 4 elements")

        data_analogs = self._storage['data']['analogs']
        if len(data_analogs.shape) != 3:
            raise TypeError("Points should be a numpy with 3 exactly dimensions (1 x nAnalogs x nFrames)")
        nb_analog_components = data_analogs.shape[0]
        nb_analogs = data_analogs.shape[1]
        nb_analog_frames = data_analogs.shape[2]
        if nb_analog_components != 1:
            raise TypeError("Points should be a numpy with first dimension exactly equals to 1 element")
        nb_analog_subframes = 0
        if nb_point_frames != 0:
            if nb_analog_frames % nb_point_frames != 0:
                raise ValueError("Number of frames of Points and Analogs should be a multiple of an integer")
            nb_analog_subframes = int(nb_analog_frames / nb_point_frames)

        # Start from a fresh c3d
        new_c3d = ezc3d.c3d()

        # Fill the parameters
        groups = self._storage['parameters']
        for group in groups:
            for param in groups[group]:
                # Copy the parameters into the c3d, but skip those who will be updated automatically later
                if (
                    not (group == "POINT" and param == "USED")
                    and (not (group == "POINT" and param == "FRAMES"))
                    and (not (group == "POINT" and param == "LABELS"))
                    and (not (group == "POINT" and param == "DESCRIPTIONS"))
                    and (not (group == "ANALOG" and param == "USED"))
                    and (not (group == "ANALOG" and param == "LABELS"))
                    and (not (group == "ANALOG" and param == "DESCRIPTIONS"))
                    and (not (group == "ANALOG" and param == "SCALE"))
                    and (not (group == "ANALOG" and param == "OFFSET"))
                    and (not (group == "ANALOG" and param == "UNITS"))
                ):
                    if group == "FORCE_PLATFORM" and param == "CORNERS":
                        print("Coucou")
                    old_param = groups[group][param]
                    new_param = ezc3d.Parameter(param)
                    dim = [len(old_param["value"])]
                    if old_param["type"] == ezc3d.BYTE or old_param["type"] == ezc3d.INT:
                        new_param.set(ezc3d.VecInt(old_param["value"]), dim)
                    elif old_param["type"] == ezc3d.FLOAT:
                        new_param.set(ezc3d.VecFloat(old_param["value"]), dim)
                    elif old_param["type"] == ezc3d.CHAR:
                        new_param.set(ezc3d.VecString(old_param["value"]), dim)
                    else:
                        raise NotImplementedError("Parameter type not implemented yet")
                    new_c3d.addParameter(group, new_param)

        # Update some important stuff (name of markers and analogs)
        point_labels = groups['POINT']['LABELS']['value']
        for point_label in point_labels:
            new_c3d.addMarker(point_label)

        analog_labels = groups['ANALOG']['LABELS']['value']
        for analog_label in analog_labels:
            new_c3d.addAnalog(analog_label)

        # Initialization for speed
        pt = ezc3d.Point()
        pts = ezc3d.Points()
        for i in range(nb_points):
            pts.add(pt)
        c = ezc3d.Channel()
        subframe = ezc3d.SubFrame()
        for i in range (nb_analogs):
            subframe.addChannel(c)
        analogs = ezc3d.Analogs()
        for i in range (nb_analog_subframes):
            analogs.addSubframe(subframe)

        # Fill the data
        for f in range(nb_point_frames):
            for i in range(nb_points):
                pt.name(point_labels[i])
                pt.x(data_points[0, i, f])
                pt.y(data_points[1, i, f])
                pt.z(data_points[2, i, f])
                pts.replace(i, pt)

            for sf in range(nb_analog_subframes):
                for i in range(nb_analogs):
                    c.name(analog_labels[i])
                    c.value(data_analogs[0, i, nb_analog_subframes*sf + sf])
                    subframe.replaceChannel(i, c)
                analogs.replaceSubframe(sf, subframe)
            frame = ezc3d.Frame()
            frame.add(pts, analogs)
            new_c3d.addFrame(frame)

        # Write the file
        new_c3d.write(path)
        return


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
    def __init__(self, path):
        super(c3d, self).__init__()

        # Interface to swig pointers
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
                    self._storage[group.name()][parameter.name()]['description'] = parameter.description()
                    if parameter.type() == 1:  # BYTE
                        value = parameter.valuesAsByte()
                    elif parameter.type() == 2:  # INT
                        value = parameter.valuesAsInt()
                    elif parameter.type() == 4:  # FLOAT
                        value = parameter.valuesAsFloat()
                    elif parameter.type() == -1:  # CHAR
                        table = parameter.valuesAsString()
                        value = []
                        for element in table:
                            value.append(element.c_str())
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

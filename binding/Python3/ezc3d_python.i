/* File : ezc3d_python.i */
%{
#define SWIG_FILE_WITH_INIT
#include "ezc3d.h"
%}

%include "numpy.i"
%init %{
    import_array();
%}
%include <std_vector.i>

%apply (int* IN_ARRAY1, int DIM1) {(int* markers, int nMarkers)};
%apply (int* IN_ARRAY1, int DIM1) {(int* channels, int nChannels)};

%inline %{
PyObject * _get_points(const ezc3d::c3d& c3d, const std::vector<int>& markers)
{
    // Get the data
    size_t nMarkers(markers.size());
    const std::vector<ezc3d::DataNS::Frame>& frames = c3d.data().frames();
    size_t nFrames(frames.size());
    double * data = new double[4 * nMarkers * nFrames];
    for (int f = 0; f < nFrames; ++f){
        for (int m = 0; m < nMarkers; ++m){
            const ezc3d::DataNS::Points3dNS::Point& point(frames[f].points().point(markers[m]));
            data[nMarkers*nFrames*0+nFrames*m+f] = point.x();
            data[nMarkers*nFrames*1+nFrames*m+f] = point.y();
            data[nMarkers*nFrames*2+nFrames*m+f] = point.z();
            data[nMarkers*nFrames*3+nFrames*m+f] = 1;
        }
    }

    // Export them to Python Object
    int nArraySize = 3;
    npy_intp * arraySizes = new npy_intp[nArraySize];
    arraySizes[0] = 4;
    arraySizes[1] = nMarkers;
    arraySizes[2] = nFrames;
    PyArrayObject * c = (PyArrayObject *)PyArray_SimpleNewFromData(nArraySize,arraySizes,NPY_DOUBLE, data);
    delete[] arraySizes;

    // Give ownership to Python so it will free the memory when needed
    PyArray_ENABLEFLAGS(c, NPY_ARRAY_OWNDATA);

    return PyArray_Return(c);
}
%}

%inline %{
PyObject * _get_analogs(const ezc3d::c3d& c3d, const std::vector<int>& analogs)
{
    // Get the data
    size_t nAnalogs(analogs.size());
    const std::vector<ezc3d::DataNS::Frame>& frames = c3d.data().frames();
    size_t nFrames(frames.size());
    int nSubframe(c3d.header().nbAnalogByFrame());
    double * data = new double[nAnalogs * nFrames * nSubframe];
    for (int f = 0; f < nFrames; ++f){
        for (int sf = 0; sf < nSubframe; ++sf){
            const std::vector<ezc3d::DataNS::AnalogsNS::Channel>& channels(frames[f].analogs().subframe(sf).channels());
            for (int a = 0; a < nAnalogs; ++a){
                data[nSubframe*nFrames*a + sf+nSubframe*f] = channels[analogs[a]].value();
            }
        }
    }

    // Export them to Python Object
    int nArraySize = 3;
    npy_intp * arraySizes = new npy_intp[nArraySize];
    arraySizes[0] = 1;
    arraySizes[1] = nAnalogs;
    arraySizes[2] = nFrames * nSubframe;
    PyArrayObject * c = (PyArrayObject *)PyArray_SimpleNewFromData(nArraySize,arraySizes,NPY_DOUBLE, data);
    delete[] arraySizes;

    // Give ownership to Python so it will free the memory when needed
    PyArray_ENABLEFLAGS(c, NPY_ARRAY_OWNDATA);

    return PyArray_Return(c);
}
%}

%extend ezc3d::c3d
{
    // Extend c3d class to get an easy accessor to data points
    PyObject * get_points(){
        std::vector<int> markers;
        for (int i = 0; i < self->header().nb3dPoints(); ++i)
            markers.push_back(i);
        return _get_points(*self, markers);
    }

    PyObject * get_points(int* markers, int nMarkers)
    {
        std::vector<int> _markers;
        for (int i = 0; i < nMarkers; ++i)
            _markers.push_back(markers[i]);
        return _get_points(*self, _markers);
    }

    PyObject * get_points(int marker)
    {
        std::vector<int> markers;
        markers.push_back(marker);
        return _get_points(*self, markers);
    }


    // Extend c3d class to get an easy accessor to data points
    PyObject * get_analogs(){
        std::vector<int> channels;
        for (int i = 0; i < self->header().nbAnalogs(); ++i)
            channels.push_back(i);
        return _get_analogs(*self, channels);
    }

    PyObject * get_analogs(int* channels, int nChannels)
    {
        std::vector<int> _channels;
        for (int i = 0; i < nChannels; ++i)
            _channels.push_back(channels[i]);
        return _get_analogs(*self, _channels);
    }

    PyObject * get_analogs(int channel)
    {
        std::vector<int> channels;
        channels.push_back(channel);
        return _get_analogs(*self, channels);
    }
}

%include ../ezc3d.i




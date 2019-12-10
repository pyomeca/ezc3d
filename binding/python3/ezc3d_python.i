// File : ezc3d_python.i
%{
#define SWIG_FILE_WITH_INIT
#include "ezc3d.h"
#include "Header.h"
#include "Data.h"
#include "Parameters.h"
%}

%include "numpy.i"
%init %{
    import_array();
%}
%include <std_vector.i>

%apply (int* IN_ARRAY1, int DIM1) {(int* points, int nPoints)};
%apply (int* IN_ARRAY1, int DIM1) {(int* channels, int nChannels)};

%rename(console_print) print;

%inline %{
PyObject * _get_points(const ezc3d::c3d& c3d, const std::vector<int>& points)
{
    // Get the data
    size_t nPoints(points.size());
    size_t nFrames(c3d.data().nbFrames());
    double * data = new double[4 * nPoints * nFrames];
    for (size_t f = 0; f < nFrames; ++f){
        for (size_t m = 0; m < nPoints; ++m){
            const ezc3d::DataNS::Points3dNS::Point& point(c3d.data().frame(f).points().point(points[m]));
            data[nPoints*nFrames*0+nFrames*m+f] = point.x();
            data[nPoints*nFrames*1+nFrames*m+f] = point.y();
            data[nPoints*nFrames*2+nFrames*m+f] = point.z();
            data[nPoints*nFrames*3+nFrames*m+f] = 1;
        }
    }

    // Export them to Python Object
    int nArraySize = 3;
    npy_intp * arraySizes = new npy_intp[nArraySize];
    arraySizes[0] = 4;
    arraySizes[1] = nPoints;
    arraySizes[2] = nFrames;
    PyArrayObject * c = (PyArrayObject *)PyArray_SimpleNewFromData(nArraySize,arraySizes,NPY_DOUBLE, data);
    delete[] arraySizes;

    // Give ownership to Python so it will free the memory when needed
    PyArray_ENABLEFLAGS(c, NPY_ARRAY_OWNDATA);

    return PyArray_Return(c);
}
%}

%inline %{
PyObject * _get_point_residuals(
            const ezc3d::c3d& c3d,
            const std::vector<int>& points)
{
    // Get the data
    size_t nPoints(points.size());
    size_t nFrames(c3d.data().nbFrames());
    double * data = new double[1 * nPoints * nFrames];
    for (size_t f = 0; f < nFrames; ++f){
        for (size_t m = 0; m < nPoints; ++m){
            const ezc3d::DataNS::Points3dNS::Point& point(c3d.data().frame(f).points().point(points[m]));
            data[nFrames*m+f] = point.residual();
        }
    }

    // Export them to Python Object
    int nArraySize = 3;
    npy_intp * arraySizes = new npy_intp[nArraySize];
    arraySizes[0] = 1;
    arraySizes[1] = nPoints;
    arraySizes[2] = nFrames;
    PyArrayObject * c = (PyArrayObject *)PyArray_SimpleNewFromData(nArraySize,arraySizes,NPY_DOUBLE, data);
    delete[] arraySizes;

    // Give ownership to Python so it will free the memory when needed
    PyArray_ENABLEFLAGS(c, NPY_ARRAY_OWNDATA);

    return PyArray_Return(c);
}
%}

%inline %{
PyObject * _get_point_camera_masks(
            const ezc3d::c3d& c3d,
            const std::vector<int>& points)
{
    // Get the data
    size_t nPoints(points.size());
    size_t nFrames(c3d.data().nbFrames());
    bool * data = new bool[7 * nPoints * nFrames];
    for (size_t f = 0; f < nFrames; ++f){
        for (size_t m = 0; m < nPoints; ++m){
            const ezc3d::DataNS::Points3dNS::Point& point(c3d.data().frame(f).points().point(points[m]));
            const std::vector<bool>& cam(point.cameraMask());
            for (size_t c = 0; c<cam.size(); ++c){
                data[nPoints*nFrames*c+nFrames*m+f] = cam[c];
            }
        }
    }

    // Export them to Python Object
    int nArraySize = 3;
    npy_intp * arraySizes = new npy_intp[nArraySize];
    arraySizes[0] = 7;
    arraySizes[1] = nPoints;
    arraySizes[2] = nFrames;
    PyArrayObject * c = (PyArrayObject *)PyArray_SimpleNewFromData(nArraySize,arraySizes,NPY_BOOL, data);
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
    size_t nFrames(c3d.data().nbFrames());
    int nSubframe(c3d.header().nbAnalogByFrame());
    double * data = new double[nAnalogs * nFrames * nSubframe];
    for (size_t f = 0; f < nFrames; ++f)
        for (size_t sf = 0; sf < nSubframe; ++sf)
            for (int a = 0; a < nAnalogs; ++a)
                data[nSubframe*nFrames*a + sf+nSubframe*f] = c3d.data().frame(f).analogs().subframe(sf).channel(a).data();

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
        std::vector<int> points;
        for (int i = 0; i < self->header().nb3dPoints(); ++i)
            points.push_back(i);
        return _get_points(*self, points);
    }

    PyObject * get_points(int* points, int nPoints)
    {
        std::vector<int> _points;
        for (int i = 0; i < nPoints; ++i)
            _points.push_back(points[i]);
        return _get_points(*self, _points);
    }

    PyObject * get_points(int point)
    {
        std::vector<int> points;
        points.push_back(point);
        return _get_points(*self, points);
    }


    // Extend c3d class to get an easy accessor to data point residuals
    PyObject * get_point_residuals(){
        std::vector<int> points;
        for (int i = 0; i < self->header().nb3dPoints(); ++i)
            points.push_back(i);
        return _get_point_residuals(*self, points);
    }

    PyObject * get_point_residuals(int* points, int nPoints)
    {
        std::vector<int> _points;
        for (int i = 0; i < nPoints; ++i)
            _points.push_back(points[i]);
        return _get_point_residuals(*self, _points);
    }

    PyObject * get_point_residuals(int point)
    {
        std::vector<int> points;
        points.push_back(point);
        return _get_point_residuals(*self, points);
    }



    // Extend c3d class to get an easy accessor to data point camera masks
    PyObject * get_point_camera_masks(){
        std::vector<int> points;
        for (int i = 0; i < self->header().nb3dPoints(); ++i)
            points.push_back(i);
        return _get_point_camera_masks(*self, points);
    }

    PyObject * get_point_residuals(int* points, int nPoints)
    {
        std::vector<int> _points;
        for (int i = 0; i < nPoints; ++i)
            _points.push_back(points[i]);
        return _get_point_camera_masks(*self, _points);
    }

    PyObject * get_point_residuals(int point)
    {
        std::vector<int> points;
        points.push_back(point);
        return _get_point_camera_masks(*self, points);
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




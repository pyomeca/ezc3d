/* File : ezC3D_python.i */
%{
#define SWIG_FILE_WITH_INIT
#include "ezC3D.h"
%}

%include "numpy.i"
%init %{
    import_array();
%}
%include <std_vector.i>

%apply (int* IN_ARRAY1, int DIM1) {(int* frames, int nFrames)};

%inline %{
PyObject * _finalizePoints(const ezC3D::C3D& c3d,
                           const std::vector<int>& markers,
                           const std::vector<int>& frames)
{
    // Get the data
    int nMarkers = markers.size();
    int nFrames = frames.size();
    double * data = new double[4 * nMarkers * nFrames];
    for (int f = 0; f < frames.size(); ++f){
        const ezC3D::DataNS::Frame& frame(c3d.data().frame(frames[f]));
        for (int m = 0; m < markers.size(); ++m){
            const ezC3D::DataNS::Points3dNS::Point& point(frame.points().point(markers[m]));
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

%extend ezC3D::C3D
{
    PyObject * getPoints(){
        std::vector<int> markers;
        for (int i = 0; i < self->header().nb3dPoints(); ++i)
            markers.push_back(i);

        std::vector<int> frames;
        for (int i = 0; i < self->header().nbFrames(); ++i)
            frames.push_back(i);

        return _finalizePoints(*self, markers, frames);
    }

    PyObject * getPoints(int* frames, int nFrames)
    {
        std::vector<int> markers;
        for (int i = 0; i < self->header().nb3dPoints(); ++i)
            markers.push_back(i);

        std::vector<int> _frames;
        _frames.assign(frames, frames + nFrames);
        return _finalizePoints(*self, markers, _frames);
    }

    PyObject * getPoint(int frame)
    {
        std::vector<int> markers;
        for (int i = 0; i < self->header().nb3dPoints(); ++i)
            markers.push_back(i);

        std::vector<int> frames;
        frames.push_back(frame);

        return _finalizePoints(*self, markers, frames);
    }
}

%include ../ezC3D.i




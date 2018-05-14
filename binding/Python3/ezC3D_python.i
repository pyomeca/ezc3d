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
PyObject * _finalizePoints(int nMarkers, int nFrames, double * data){
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
        int nMarkers = self->header().nb3dPoints();
        int nFrames = self->header().nbFrames();
        double * data = new double[4 * nMarkers * nFrames];
        for (int f = 0; f < nFrames; ++f){
            const ezC3D::DataNS::Frame& frame(self->data().frame(f));
            for (int i = 0; i < frame.points().points().size(); ++i){
                const ezC3D::DataNS::Points3dNS::Point& point(frame.points().point(i));
                data[nMarkers*nFrames*0+nFrames*i+f] = point.x();
                data[nMarkers*nFrames*1+nFrames*i+f] = point.y();
                data[nMarkers*nFrames*2+nFrames*i+f] = point.z();
                data[nMarkers*nFrames*3+nFrames*i+f] = 1;
            }
        }
        return _finalizePoints(nMarkers, nFrames, data);
    }

    PyObject * getPoints(int* frames, int nFrames)
    {
        int nMarkers = self->header().nb3dPoints();
        double * data = new double[4 * nMarkers * nFrames];
        for (int f = 0; f < nFrames; ++f){
            const ezC3D::DataNS::Frame& frame(self->data().frame(frames[f]));
            for (int i = 0; i < frame.points().points().size(); ++i){
                const ezC3D::DataNS::Points3dNS::Point& point(frame.points().point(i));
                data[nMarkers*nFrames*0+nFrames*i+f] = point.x();
                data[nMarkers*nFrames*1+nFrames*i+f] = point.y();
                data[nMarkers*nFrames*2+nFrames*i+f] = point.z();
                data[nMarkers*nFrames*3+nFrames*i+f] = 1;
            }
        }
        return _finalizePoints(nMarkers, nFrames, data);
    }

    PyObject * getPoint(int frame)
    {
        double * data = new double[4 * self->header().nb3dPoints()];
        int nMarkers = self->header().nb3dPoints();

        const ezC3D::DataNS::Frame& f(self->data().frame(frame));
        for (int i = 0; i < f.points().points().size(); ++i){
            data[i + 0*self->header().nb3dPoints()] = f.points().point(i).x();
            data[i + 1*self->header().nb3dPoints()] = f.points().point(i).y();
            data[i + 2*self->header().nb3dPoints()] = f.points().point(i).z();
            data[i + 3*self->header().nb3dPoints()] = 1;
        }
        return _finalizePoints(nMarkers, 1, data);
    }
}

%include ../ezC3D.i




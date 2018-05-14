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


%extend ezC3D::C3D
{
PyObject * getFrame(int frame)
{ 
    const ezC3D::DataNS::Frame& f(self->data().frame(frame));
    
    double * data = new double[self->header().nb3dPoints()];  
    for (int i = 0; i < f.points().points().size(); ++i){
        data[i + 0*self->header().nb3dPoints()] = f.points().point(i).x();
        data[i + 1*self->header().nb3dPoints()] = f.points().point(i).y();
        data[i + 2*self->header().nb3dPoints()] = f.points().point(i).z();
    }
  
    int nArraySize = 2; 
    npy_intp * arraySizes = new npy_intp[nArraySize];
    arraySizes[0] = 3;
    arraySizes[1] = self->header().nb3dPoints();

    PyArrayObject * c = (PyArrayObject *)PyArray_SimpleNewFromData(nArraySize,arraySizes,NPY_DOUBLE, data);
    delete[] arraySizes;

    // Give ownership to Python so it will free the memory when needed
    PyArray_ENABLEFLAGS(c, NPY_ARRAY_OWNDATA);

    return PyArray_Return(c);
}
}

%include ../ezC3D.i




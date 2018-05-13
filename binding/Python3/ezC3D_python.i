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


%apply (double* INPLACE_ARRAY1, int DIM1) {(double* tata, int n_tata)}

%include ../ezC3D.i




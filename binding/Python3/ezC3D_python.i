/* File : ezC3D_python.i */
%{
#define SWIG_FILE_WITH_INIT
%}

%include "numpy.i"
%init %{
    import_array();
%}
%include <std_vector.i>
/*
%apply (size_t DIM1, double* INPLACE_ARRAY1) {(size_t len_, double* vec_)}
%rename (coucou) my_coucou;
%inline %{
int my_coucou(size_t len_, double* vec_) {
    std::vector<double> v;
    v.insert(v.end(), vec_, vec_ + len_);
    return coucou(v);
}
%}
*/
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* tata, int n_tata)}

%include ../ezC3D.i




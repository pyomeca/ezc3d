/* File : example.i */
%module example
%{
#include "/home/pariterre/Documents/Laboratoire/Programmation/ezC3D/include/ezC3D.h"
%}

/*  Instantiate std_vector */
%include <std/std_vector.i>
namespace std {
  %template(VecFrame) vector<ezC3D::DataNS::Frame>;
};

/* Instantiate std_string */
%include <std/std_iostream.i>

/* Includes all neceressary files from the API */
%include "/home/pariterre/Documents/Laboratoire/Programmation/ezC3D/include/ezC3D.h"
%include "/home/pariterre/Documents/Laboratoire/Programmation/ezC3D/include/Header.h"
%include "/home/pariterre/Documents/Laboratoire/Programmation/ezC3D/include/Parameters.h"
%include "/home/pariterre/Documents/Laboratoire/Programmation/ezC3D/include/Data.h"


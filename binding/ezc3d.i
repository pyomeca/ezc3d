/* File : ezC3D.i */
%module ezc3d
%{
#include "ezC3D.h"
%}

/* Instantiate std_string */
%include <std_iostream.i>

/* Instantiate std_vector */
%include <std_vector.i>

// Instantiate templates
namespace std {
   %template(VecInt) vector<int>;
   %template(VecFloat) vector<float>;
   %template(VecString) vector<std::string>;

   %template(VecGroups) vector<ezC3D::ParametersNS::GroupNS::Group>;
   %template(VecParameters) vector<ezC3D::ParametersNS::GroupNS::Parameter>;

   %template(VecFrames) vector<ezC3D::DataNS::Frame>;
   %template(VecPoints) vector<ezC3D::DataNS::Points3dNS::Point>;
   %template(VecAnalogSubFrames) vector<ezC3D::DataNS::AnalogsNS::SubFrame>;
   %template(VecAnalogChannels) vector<ezC3D::DataNS::AnalogsNS::Channel>;

}


/* Includes all neceressary files from the API */
%include "ezC3D.h"
%include "Header.h"
%include "Parameters.h"
%include "Data.h"


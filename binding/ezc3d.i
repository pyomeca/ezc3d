/* File : ezc3d.i */
%module ezc3d
%{
#include "ezc3d.h"
%}

/* Instantiate std_vector */
%include <std_vector.i>

/* Instantiate std_string */
%include <std_iostream.i>

// Instantiate templates
namespace std {
   %template(VecInt) vector<int>;
   %template(VecFloat) vector<float>;
   %template(VecString) vector<std::string>;

   %template(VecGroups) vector<ezc3d::ParametersNS::GroupNS::Group>;
   %template(VecParameters) vector<ezc3d::ParametersNS::GroupNS::Parameter>;

   %template(VecFrames) vector<ezc3d::DataNS::Frame>;
   %template(VecPoints) vector<ezc3d::DataNS::Points3dNS::Point>;
   %template(VecAnalogSubFrames) vector<ezc3d::DataNS::AnalogsNS::SubFrame>;
   %template(VecAnalogChannels) vector<ezc3d::DataNS::AnalogsNS::Channel>;

}


/* Includes all neceressary files from the API */
%include "ezc3d.h"
%include "Header.h"
%include "Parameters.h"
%include "Data.h"


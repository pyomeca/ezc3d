// File : ezc3d.i
%module ezc3d
%{
#include "ezc3d_all.h"
#include "modules/ForcePlatforms.h"
%}

// Instantiate from standard library
%include <std_vector.i>
%include <std_string.i>
%include <std_iostream.i>
%include <std_except.i>

// Instantiate templates
namespace std {
    %template(VecBool) vector<bool>;
    %template(VecInt) vector<int>;
    %template(VecUInt) vector<size_t>;
    %template(VecFloat) vector<float>;
    %template(VecDouble) vector<double>;
    %template(VecString) vector<std::string>;

    %template(VecGroups) vector<ezc3d::ParametersNS::GroupNS::Group>;
    %template(VecParameters) vector<ezc3d::ParametersNS::GroupNS::Parameter>;

    %template(VecFrames) vector<ezc3d::DataNS::Frame>;
    %template(VecPoints) vector<ezc3d::DataNS::Points3dNS::Point>;
    %template(VecAnalogSubFrames) vector<ezc3d::DataNS::AnalogsNS::SubFrame>;
    %template(VecAnalogChannels) vector<ezc3d::DataNS::AnalogsNS::Channel>;

}

// Manage exceptions raised
%include exception.i
%exception {
    try {
        $action
    } catch (const std::invalid_argument& e) {
        SWIG_exception(SWIG_ValueError, e.what());
    } catch (const std::out_of_range& e) {
        SWIG_exception(SWIG_ValueError, e.what());
    } catch (const std::ios_base::failure& e) {
        SWIG_exception(SWIG_IOError, e.what());
    } catch (const std::runtime_error& e) {
        SWIG_exception(SWIG_RuntimeError, e.what());
    } catch (const std::exception& e) {
        SWIG_exception(SWIG_RuntimeError, e.what());
    } catch (...) {
       SWIG_exception(SWIG_UnknownError, "An unknown exception was raise");
    }
}

// Includes all necessary files from the API
%include "ezc3dConfig.h"
%include "ezc3d.h"
%include "math/Matrix.h"
%include "math/Matrix33.h"
%include "math/Vector3d.h"
%include "Header.h"
%include "Parameters.h"
%include "Group.h"
%include "Parameter.h"
%include "Data.h"
%include "Frame.h"
%include "Points.h"
%include "Point.h"
%include "Analogs.h"
%include "Subframe.h"
%include "Channel.h"

%include "modules/ForcePlatforms.h"

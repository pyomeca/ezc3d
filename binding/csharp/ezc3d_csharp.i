
%{
#define SWIG_FILE_WITH_INIT

#include "ezc3d/ezc3d.h"
#include "ezc3d/Header.h"
#include "ezc3d/Data.h"
#include "ezc3d/Parameters.h"
#include "ezc3d/RotationsInfo.h"
%}

// Rename the "lock" and "unlock" methods to avoid conflict with C# keywords
%rename(LockGroup) ezc3d::ParametersNS::GroupNS::Group::lock;
%rename(UnlockGroup) ezc3d::ParametersNS::GroupNS::Group::unlock;
%rename(LockParameter) ezc3d::ParametersNS::GroupNS::Parameter::lock;
%rename(UnlockParameter) ezc3d::ParametersNS::GroupNS::Parameter::unlock;

%module ezc3d_csharp
%include ../ezc3d.i
#ifndef EZC3D_NAMESPACE_H
#define EZC3D_NAMESPACE_H
///
/// \file ezc3d_namespace.h
/// \brief Declaration of the ezc3d namespace
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

///
/// \mainpage Documentation of ezc3d
///
/// \section intro_sec Introduction
///
/// This is the document for the library ezc3d
/// (<a
/// href="http://github.com/pyomeca/ezc3d">http://github.com/pyomeca/ezc3d</a>).
/// The main goal of this library is to eazily create, read and modify c3d (<a
/// href="http://c3d.org">http://c3d.org</a>) files, largely used in
/// biomechanics.
///
/// This documentation was automatically generated for the "Nostalgia"
/// Release 1.1.0 on the 9th of August, 2019.
///
/// \section install_sec Installation
///
/// To install ezc3d, please refer to the README.md file accessible via the
/// github repository or by following this <a
/// href="md__home_pariterre_programmation_ezc3d_README.html">link</a>.
///
/// \section contact_sec Contact
///
/// If you have any questions, comments or suggestions for future development,
/// you are very welcomed to send me an email at <a
/// href="mailto:pariterre@gmail.com">pariterre@gmail.com</a>.
///
/// \section conclusion_sec Conclusion
///
/// Enjoy C3D files!
///

// Includes for standard library
#include <memory>

#include "ezc3d/ezc3dConfig.h"
#ifdef _WIN32
#include <string>
#endif

///
/// \brief Namespace ezc3d
///
/// Useful functions, enum and misc useful for the ezc3d project
///
namespace ezc3d {
// ---- UTILS ---- //
///
/// \brief Enum that describes the size of different types
///
enum DATA_TYPE {
  CHAR = -1,
  BYTE = 1,
  INT = 2,
  WORD = 2,
  FLOAT = 4,
  NO_DATA_TYPE = 10000
};

///
/// \brief The type of processor used to store the data
///
enum PROCESSOR_TYPE {
  INTEL = 84,
  DEC = 85,
  MIPS = 86,
  NO_PROCESSOR_TYPE = INTEL
};

///
/// \brief The order of the parameters in the new c3d file
///
enum WRITE_FORMAT { DEFAULT = 0, NEXUS };

///
/// \brief Remove the spaces at the end of a string
/// \param str The string to remove the trailing spaces from.
///
/// The function receive a string and modify it by remove the trailing spaces
///
EZC3D_VISIBILITY EZC3D_API void removeTrailingSpaces(std::string &str);

///
/// \brief Swap all characters of a string to capital letters
/// \param str The string to capitalize
///
EZC3D_VISIBILITY EZC3D_API std::string toUpper(const std::string &str);

// ---- FORWARD DECLARATION OF THE WHOLE PROJECT STRUCTURE ----//
class c3d;
class Matrix;
class Matrix33;
class Matrix44;
class Matrix66;
class Vector3d;
class Vector6d;
class Header;
class DataStartInfo;

///
/// \brief Namespace that holds the Parameters hierarchy
///
namespace ParametersNS {
class Parameters;

///
/// \brief Namespace that holds the Group and Parameter classes
///
namespace GroupNS {
class Group;
class Parameter;
} // namespace GroupNS
} // namespace ParametersNS

///
/// \brief Namespace that holds the Data hierarchy
///
namespace DataNS {
class Data;

class Frame;
///
/// \brief Namespace that holds the Points hierarchy
///
namespace Points3dNS {
class Points;
class Point;
class Info;
} // namespace Points3dNS
///
/// \brief Namespace that holds the Analogs hierarchy
///
namespace AnalogsNS {
class Analogs;
class SubFrame;
class Channel;
class Info;
} // namespace AnalogsNS

///
/// \brief Namespace that holds the Rotations hierarchy
///
namespace RotationNS {
class Rotations;
class Rotation;
class SubFrame;
class Info;
} // namespace RotationNS
} // namespace DataNS

///
/// \brief Namespace for all the analysis modules
///
namespace Modules {
class ForcePlatform;
class ForcePlatforms;
} // namespace Modules
} // namespace ezc3d
#endif

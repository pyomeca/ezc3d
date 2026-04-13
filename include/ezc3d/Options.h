#ifndef EZC3D_OPTIONS_H
#define EZC3D_OPTIONS_H
///
/// \file Options.h
/// \brief Declaration of Options class
/// \author Pariterre
/// \version 1.0
/// \date October 26th, 2025
///

#include "ezc3d/ezc3dNamespace.h"

class EZC3D_VISIBILITY ezc3d::Options {
public:
  ///
  /// \brief Constructor with default options
  ///
  EZC3D_API
  Options(bool ignoreBadFormatting = false,
          bool keepParametersTrailingSpaces = false)
      : _ignoreBadFormatting(ignoreBadFormatting),
        _keepParametersTrailingSpaces(keepParametersTrailingSpaces) {}

  ///
  /// \brief Whether to ignore bad formatting when reading C3D files
  /// \return Whether to ignore bad formatting when reading C3D files
  ///
  EZC3D_API bool getIgnoreBadFormatting() const { return _ignoreBadFormatting; }

  ///
  /// \brief Whether to keep trailing spaces when reading strings
  /// \return Whether to keep trailing spaces when reading strings
  ///
  EZC3D_API bool getKeepParametersTrailingSpaces() const {
    return _keepParametersTrailingSpaces;
  }

  ///
  /// \brief Create a deep copy of the options
  /// \return A deep copy of the options
  ///
  EZC3D_API Options clone() const {
    return Options(_ignoreBadFormatting, _keepParametersTrailingSpaces);
  }

protected:
  bool _ignoreBadFormatting; ///< Whether to ignore bad formatting when reading
                             ///< C3D files

  bool _keepParametersTrailingSpaces; ///< Whether to keep trailing spaces when
                                      ///< reading strings
};

#endif // EZC3D_OPTIONS_H
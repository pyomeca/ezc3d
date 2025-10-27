#ifndef EZC3D_WRITE_OPTIONS_H
#define EZC3D_WRITE_OPTIONS_H
///
/// \file ezc3dWriteOptions.h
/// \brief Declaration of c3dWriteOptions class
/// \author Pariterre
/// \version 1.0
/// \date October 26th, 2025
///

#include "ezc3d/ezc3dNamespace.h"

class EZC3D_VISIBILITY ezc3d::WriteOptions {
public:
  ///
  /// \brief Constructor with default options
  ///
  EZC3D_API
  WriteOptions(bool collapseStringMatrices = true,
               bool forceZeroBasedOnFrameCount = false)
      : _collapseStringMatrices(collapseStringMatrices),
        _forceZeroBasedOnFrameCount(forceZeroBasedOnFrameCount) {}

  ///
  /// \brief Whether to collapse the string matrices to vector
  EZC3D_API bool getCollapseStringMatrices() const {
    return _collapseStringMatrices;
  }

  ///
  /// \brief According to the standard, the first and
  /// last frame are stored as a one-based value. But some software requires it
  /// to be zero. Leave the user the capability to do so.
  EZC3D_API bool getForceZeroBasedOnFrameCount() const {
    return _forceZeroBasedOnFrameCount;
  }

protected:
  bool _collapseStringMatrices; ///< Whether to collapse the string matrices to
                                ///< vector

  bool _forceZeroBasedOnFrameCount; ///< Whether the first frame is zero-based
                                    ///< or one-based
};

#endif // EZC3D_READ_OPTIONS_H
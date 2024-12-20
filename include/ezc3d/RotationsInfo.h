#ifndef ROTATIONS_INFO_H
#define ROTATIONS_INFO_H
///
/// \file RotationsInfo.cpp
/// \brief Implementation of RotationsInfo class
/// \author Pariterre
/// \version 1.0
/// \date April 30th, 2022
///

#include "ezc3d/ezc3dNamespace.h"

///
/// \brief 3D rotation data
///
class EZC3D_VISIBILITY ezc3d::DataNS::RotationNS::Info {
  //---- CONSTRUCTORS ----//
public:
  ///
  /// \brief Reads and create a proper RotationsInfo class
  /// c3d The c3d structure to read the values from
  ///
  EZC3D_API Info(const ezc3d::c3d &c3d);

  //---- DATA ----//
protected:
  bool _hasGroup; ///< If the group parameter is present

public:
  ///
  /// \brief Returns If the group parameter is present
  /// \return If the group parameter is present
  ///
  EZC3D_API bool hasGroup() const;

protected:
  size_t _dataStart; ///< The data start parameter

public:
  ///
  /// \brief Returns the data start parameter
  /// \return The data start parameter
  ///
  EZC3D_API size_t dataStart() const;

protected:
  size_t _used; ///< The number of Rotations

public:
  ///
  /// \brief Returns the number of Rotations
  /// \return The number of Rotations
  ///
  EZC3D_API size_t used() const;

protected:
  size_t _ratio; ///< The ratio to point data

public:
  ///
  /// \brief Returns the ratio to point data
  /// \return The ratio to point data
  ///
  EZC3D_API size_t ratio() const;

protected:
  PROCESSOR_TYPE _processorType; ///< The type of processor formatting

public:
  ///
  /// \brief Returns the type of processor formatting
  /// \return The type of processor formatting
  ///
  EZC3D_API PROCESSOR_TYPE processorType() const;
};

#endif

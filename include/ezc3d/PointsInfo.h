#ifndef POINTS_INFO_H
#define POINTS_INFO_H
///
/// \file PointsInfo.cpp
/// \brief Implementation of PointsInfo class
/// \author Pariterre
/// \version 1.0
/// \date April 30th, 2022
///

#include "ezc3d/ezc3dNamespace.h"

///
/// \brief 3D rotation data
///
class EZC3D_VISIBILITY ezc3d::DataNS::Points3dNS::Info {
  //---- CONSTRUCTORS ----//
public:
  ///
  /// \brief Reads and create a proper PointsInfo class
  /// c3d The c3d structure to read the values from
  ///
  EZC3D_API Info(const ezc3d::c3d &c3d);

  //---- DATA ----//
protected:
  PROCESSOR_TYPE _processorType; ///< The type of processor formatting

public:
  ///
  /// \brief Returns the type of processor formatting
  /// \return The type of processor formatting
  ///
  EZC3D_API PROCESSOR_TYPE processorType() const;

protected:
  double _scaleFactor; ///< The scale factor for all the points

public:
  ///
  /// \brief Returns the scale factor for all the points
  /// \return The scale factor for all the points
  ///
  EZC3D_API double scaleFactor() const;
};

#endif

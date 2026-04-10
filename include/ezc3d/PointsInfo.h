#ifndef EZC3D_POINTS_INFO_H
#define EZC3D_POINTS_INFO_H
///
/// \file PointsInfo.cpp
/// \brief Implementation of PointsInfo class
/// \author Pariterre
/// \version 1.0
/// \date April 30th, 2022
///

#include "ezc3d/ezc3dNamespace.h"
#include <vector>

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
  std::vector<double> _scaleFactors; ///< The scale factors for all the points

  ///
  /// \brief Returns the scale factors by point from a c3d structure
  /// \return The scale factors by point
  ///
  EZC3D_API std::vector<double>
  scaleFactorsFromC3d(const ezc3d::c3d &c3d) const;

public:
  ///
  /// \brief Returns the scale factors for all the points
  /// \return The scale factors for all the points
  ///
  EZC3D_API std::vector<double> scaleFactors() const;
};

#endif

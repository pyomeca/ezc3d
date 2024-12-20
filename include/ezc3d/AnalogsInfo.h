#ifndef ANALOGS_INFO_H
#define ANALOGS_INFO_H
///
/// \file AnalogInfo.cpp
/// \brief Implementation of AnalogInfo class
/// \author Pariterre
/// \version 1.0
/// \date April 30th, 2022
///

#include "ezc3d/ezc3dNamespace.h"
#include <vector>

///
/// \brief 3D rotation data
///
class EZC3D_VISIBILITY ezc3d::DataNS::AnalogsNS::Info {
  //---- CONSTRUCTORS ----//
public:
  ///
  /// \brief Reads and create a proper AnalogInfo class
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
  std::vector<double> _scaleFactors; ///< The scale factors by channel

public:
  ///
  /// \brief Returns the scale factors by channel
  /// \return The scale factors by channel
  ///
  EZC3D_API const std::vector<double> &scaleFactors() const;

protected:
  double _generalFactor; ///< The general scale factor

public:
  ///
  /// \brief Returns the general scale factor
  /// \return The general scale factor
  ///
  EZC3D_API double generalFactor() const;

protected:
  std::vector<int> _zeroOffset; ///< The offset of the analogs

public:
  ///
  /// \brief Returns the zero offset
  /// \return The zero offset
  ///
  EZC3D_API const std::vector<int> &zeroOffset() const;
};

#endif

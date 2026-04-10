#ifndef EZC3D_ANALOGS_INFO_H
#define EZC3D_ANALOGS_INFO_H
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
/// \brief Information about analog data
///
class EZC3D_VISIBILITY ezc3d::DataNS::AnalogsNS::Info {
  //---- CONSTRUCTORS ----//
public:
  ///
  /// \brief Reads and create a proper AnalogInfo class
  /// c3d The c3d structure to read the values from
  ///
  EZC3D_API Info(const ezc3d::c3d &c3d);

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

  ///
  /// \brief Returns the scale factors by channel from a c3d structure
  /// \param c3d The c3d structure to read the values from
  /// \return The scale factors by channel
  ///
  EZC3D_API std::vector<double>
  scaleFactorsFromC3d(const ezc3d::c3d &c3d) const;

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

  ///
  /// \brief Returns the zero offset from a c3d structure
  /// \param c3d The c3d structure to read the values from
  /// \return The zero offset
  ///
  EZC3D_API std::vector<int> channelOffsetsFromC3d(const ezc3d::c3d &c3d) const;

public:
  ///
  /// \brief Returns the zero offset
  /// \return The zero offset
  ///
  EZC3D_API const std::vector<int> &zeroOffset() const;
};

#endif

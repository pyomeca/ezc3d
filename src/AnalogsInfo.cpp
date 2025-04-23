#define EZC3D_API_EXPORTS
///
/// \file AnalogsInfo.cpp
/// \brief Implementation of AnalogsInfo class
/// \author Pariterre
/// \version 1.0
/// \date April 30th, 2022
///

#include "ezc3d/AnalogsInfo.h"
#include "ezc3d/Header.h"
#include "ezc3d/Parameters.h"
#include "ezc3d/ezc3d.h"

ezc3d::DataNS::AnalogsNS::Info::Info(const ezc3d::c3d &c3d)
    : _processorType(ezc3d::PROCESSOR_TYPE::INTEL),
      _scaleFactors(std::vector<double>()), _generalFactor(-1),
      _zeroOffset(std::vector<int>()) {
  _processorType = c3d.parameters().processorType();

  if (c3d.header().nbAnalogs())
    _scaleFactors = c3d.channelScales();
  _generalFactor = c3d.parameters()
                       .group("ANALOG")
                       .parameter("GEN_SCALE")
                       .valuesAsDouble()[0];
  _zeroOffset = c3d.channelOffsets();
  for (int &offset : _zeroOffset) {
    offset = abs(offset);
  }

  if (c3d.parameters().isGroup("SHADOW")) {
    // The SHADOW company did not respect the standard and put these values in
    // the ANALOG group So we have to assume some default values
    if (_scaleFactors.empty()) {
      for (size_t i = 0; i < c3d.header().nbAnalogs(); ++i)
        _scaleFactors.push_back(1.0);
    }
    if (_zeroOffset.empty()) {
      for (size_t i = 0; i < c3d.header().nbAnalogs(); ++i)
        _zeroOffset.push_back(0);
    }
  }
}

ezc3d::PROCESSOR_TYPE ezc3d::DataNS::AnalogsNS::Info::processorType() const {
  return _processorType;
}

const std::vector<double> &
ezc3d::DataNS::AnalogsNS::Info::scaleFactors() const {
  return _scaleFactors;
}

double ezc3d::DataNS::AnalogsNS::Info::generalFactor() const {
  return _generalFactor;
}

const std::vector<int> &ezc3d::DataNS::AnalogsNS::Info::zeroOffset() const {
  return _zeroOffset;
}

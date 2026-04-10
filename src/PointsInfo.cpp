#define EZC3D_API_EXPORTS
///
/// \file PointsInfo.cpp
/// \brief Implementation of PointsInfo class
/// \author Pariterre
/// \version 1.0
/// \date April 30th, 2022
///

#include "ezc3d/PointsInfo.h"
#include "ezc3d/Header.h"
#include "ezc3d/Parameters.h"
#include "ezc3d/ezc3d.h"

ezc3d::DataNS::Points3dNS::Info::Info(const ezc3d::c3d &c3d)
    : _processorType(ezc3d::PROCESSOR_TYPE::INTEL),
      _scaleFactors(std::vector<double>()) {
  _processorType = c3d.parameters().processorType();

  if (c3d.header().nb3dPoints())
    _scaleFactors = scaleFactorsFromC3d(c3d);
}

std::vector<double> ezc3d::DataNS::Points3dNS::Info::scaleFactorsFromC3d(
    const ezc3d::c3d &c3d) const {
  std::vector<double> scaleFactors =
      c3d.parameters().group("POINT").parameter("SCALE").valuesAsDouble();
  int i = 2;
  while (c3d.parameters().group("POINT").isParameter("SCALE" +
                                                     std::to_string(i))) {
    const auto &scales_tp = c3d.parameters()
                                .group("POINT")
                                .parameter("SCALE" + std::to_string(i))
                                .valuesAsDouble();
    scaleFactors.insert(scaleFactors.end(), scales_tp.begin(), scales_tp.end());
    ++i;
  }
  return scaleFactors;
}

ezc3d::PROCESSOR_TYPE ezc3d::DataNS::Points3dNS::Info::processorType() const {
  return _processorType;
}

std::vector<double> ezc3d::DataNS::Points3dNS::Info::scaleFactors() const {
  return _scaleFactors;
}

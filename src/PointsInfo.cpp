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
    : _processorType(ezc3d::PROCESSOR_TYPE::INTEL), _scaleFactor(-1) {
  _processorType = c3d.parameters().processorType();

  if (c3d.header().nb3dPoints())
    _scaleFactor =
        c3d.parameters().group("POINT").parameter("SCALE").valuesAsDouble()[0];
}

ezc3d::PROCESSOR_TYPE ezc3d::DataNS::Points3dNS::Info::processorType() const {
  return _processorType;
}

double ezc3d::DataNS::Points3dNS::Info::scaleFactor() const {
  return _scaleFactor;
}

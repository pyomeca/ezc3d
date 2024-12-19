#define EZC3D_API_EXPORTS
///
/// \file DataStartInfo.cpp
/// \brief Implementation of DataStartInfo class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "ezc3d/DataStartInfo.h"
#include <stdexcept>

bool ezc3d::DataStartInfo::hasPointDataStart() const {
  return m_hasPointDataStart;
}

void ezc3d::DataStartInfo::setPointDataStart(const std::streampos &value) {
  m_pointDataStart = value;
  if (int(m_pointDataStart) % 512 > 0)
    throw std::out_of_range(
        "Something went wrong in the positioning of the pointer "
        "for writting the data. Please report this error.");
  m_hasPointDataStart = true;
}

const std::streampos &ezc3d::DataStartInfo::pointDataStart() const {
  return m_pointDataStart;
}

bool ezc3d::DataStartInfo::hasHeaderPointDataStart() const {
  return m_hasHeaderPointDataStart;
}

void ezc3d::DataStartInfo::setHeaderPositionInC3dForPointDataStart(
    const std::streampos &position) {
  m_headerPointDataStart = position;
  m_hasHeaderPointDataStart = true;
}

const std::streampos &ezc3d::DataStartInfo::headerPointDataStart() const {
  return m_headerPointDataStart;
}

ezc3d::DATA_TYPE ezc3d::DataStartInfo::headerPointDataStartSize() const {
  return m_headerPointDataStartSize;
}

bool ezc3d::DataStartInfo::hasParameterPointDataStart() const {
  return m_hasParameterPointDataStart;
}

void ezc3d::DataStartInfo::setParameterPositionInC3dForPointDataStart(
    const std::streampos &position) {
  m_parameterPointDataStart = position;
  m_hasParameterPointDataStart = true;
}

const std::streampos &ezc3d::DataStartInfo::parameterPointDataStart() const {
  return m_parameterPointDataStart;
}

ezc3d::DATA_TYPE ezc3d::DataStartInfo::parameterPointDataStartSize() const {
  return m_parameterPointDataStartSize;
}

bool ezc3d::DataStartInfo::hasRotationsDataStart() const {
  return m_hasRotationDataStart;
}

void ezc3d::DataStartInfo::setRotationsDataStart(const std::streampos &value) {
  m_rotationsDataStart = value;
  if (int(m_rotationsDataStart) % 512 > 0)
    throw std::out_of_range(
        "Something went wrong in the positioning of the pointer "
        "for writting the data. Please report this error.");
  m_hasRotationDataStart = true;
}

const std::streampos &ezc3d::DataStartInfo::rotationsDataStart() const {
  return m_rotationsDataStart;
}

bool ezc3d::DataStartInfo::hasParameterRotationsDataStart() const {
  return m_hasRotationDataStart;
}

void ezc3d::DataStartInfo::setParameterPositionInC3dForRotationsDataStart(
    const std::streampos &position) {
  m_parameterRotationsDataStart = position;
  m_hasParameterRotationsDataStart = true;
}

const std::streampos &
ezc3d::DataStartInfo::parameterRotationsDataStart() const {
  return m_parameterRotationsDataStart;
}

ezc3d::DATA_TYPE ezc3d::DataStartInfo::parameterRotationsDataStartSize() const {
  return m_parameterRotationsDataStartSize;
}

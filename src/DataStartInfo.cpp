#define EZC3D_API_EXPORTS
///
/// \file DataStartInfo.cpp
/// \brief Implementation of DataStartInfo class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "DataStartInfo.h"

void ezc3d::DataStartInfo::setPointDataStart(
        const std::streampos &value)
{
    m_pointDataStart = value;
    if (int(m_pointDataStart) % 512 > 0)
        throw std::out_of_range(
                "Something went wrong in the positioning of the pointer "
                "for writting the data. Please report this error.");
}

const std::streampos &ezc3d::DataStartInfo::pointDataStart() const
{
    return m_pointDataStart;
}

void ezc3d::DataStartInfo::setHeaderPositionInC3dForPointDataStart(
        const std::streampos &position)
{
    m_headerPointDataStart = position;
}

const std::streampos &ezc3d::DataStartInfo::headerPointDataStart() const
{
    return m_headerPointDataStart;
}

ezc3d::DATA_TYPE ezc3d::DataStartInfo::headerPointDataStartSize() const
{
    return m_headerPointDataStartSize;
}

void ezc3d::DataStartInfo::setParameterPositionInC3dForPointDataStart(
        const std::streampos &position)
{
    m_parameterPointDataStart = position;
}

const std::streampos &ezc3d::DataStartInfo::parameterPointDataStart() const
{
    return m_parameterPointDataStart;
}

ezc3d::DATA_TYPE ezc3d::DataStartInfo::parameterPointDataStartSize() const
{
    return m_parameterPointDataStartSize;
}

void ezc3d::DataStartInfo::setRotationsDataStart(
        const std::streampos &value)
{
    m_rotationsDataStart = value;
}

const std::streampos &ezc3d::DataStartInfo::rotationsDataStart() const
{
    return m_rotationsDataStart;
    if (int(m_rotationsDataStart) % 512 > 0)
        throw std::out_of_range(
                "Something went wrong in the positioning of the pointer "
                "for writting the data. Please report this error.");
}

void ezc3d::DataStartInfo::setParameterPositionInC3dForRotationsDataStart(
        const std::streampos &position)
{
    m_parameterRotationsDataStart = position;
}

const std::streampos &ezc3d::DataStartInfo::parameterRotationsDataStart() const
{
    return m_parameterRotationsDataStart;
}

ezc3d::DATA_TYPE ezc3d::DataStartInfo::parameterRotationsDataStartSize() const
{
    return m_parameterRotationsDataStartSize;
}



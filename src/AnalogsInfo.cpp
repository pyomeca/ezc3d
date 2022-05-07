#define EZC3D_API_EXPORTS
///
/// \file AnalogsInfo.cpp
/// \brief Implementation of AnalogsInfo class
/// \author Pariterre
/// \version 1.0
/// \date April 30th, 2022
///

#include "AnalogsInfo.h"
#include "ezc3d.h"
#include "Header.h"
#include "Parameters.h"


ezc3d::DataNS::AnalogsNS::Info::Info(
        const ezc3d::c3d &c3d) :
    _processorType(ezc3d::PROCESSOR_TYPE::INTEL),
    _scaleFactors(std::vector<double>()),
    _generalFactor(-1),
    _zeroOffset(std::vector<int>())
{
    _processorType = c3d.parameters().processorType();

    if (c3d.header().nbAnalogs())
        _scaleFactors = c3d.parameters()
                .group("ANALOG").parameter("SCALE").valuesAsDouble();
    _generalFactor = c3d.parameters()
               .group("ANALOG").parameter("GEN_SCALE").valuesAsDouble()[0];
    _zeroOffset = c3d.parameters()
                .group("ANALOG").parameter("OFFSET").valuesAsInt();
    for (int& offset : _zeroOffset){
        offset = abs(offset);
    }
}

ezc3d::PROCESSOR_TYPE ezc3d::DataNS::AnalogsNS::Info::processorType() const
{
    return _processorType;
}

const std::vector<double>& ezc3d::DataNS::AnalogsNS::Info::scaleFactors() const
{
    return _scaleFactors;
}

double ezc3d::DataNS::AnalogsNS::Info::generalFactor() const
{
    return _generalFactor;
}

const std::vector<int>& ezc3d::DataNS::AnalogsNS::Info::zeroOffset() const
{
    return _zeroOffset;
}



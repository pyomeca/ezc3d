#define EZC3D_API_EXPORTS
///
/// \file RotationsInfo.cpp
/// \brief Implementation of RotationsInfo class
/// \author Pariterre
/// \version 1.0
/// \date April 30th, 2022
///

#include "RotationsInfo.h"
#include "ezc3d.h"
#include "Header.h"
#include "Parameters.h"


ezc3d::DataNS::RotationNS::Info::Info(
        const ezc3d::c3d &c3d)
{
    if (!c3d.parameters().isGroup("ROTATION")){
        _hasGroup = false;
        return;
    }
    _hasGroup = true;

    const ezc3d::ParametersNS::GroupNS::Group& group =
            c3d.parameters().group("ROTATION");

    // Do a sanity check before accessing
    if (!group.isParameter("DATA_START")){
        throw std::runtime_error("DATA_START is not present in ROTATION.");
    }
    _dataStart = group.parameter("DATA_START").valuesAsInt()[0];

    if (!group.isParameter("USED")){
        throw std::runtime_error("USED is not present in ROTATION.");
    }
    _used = group.parameter("USED").valuesAsInt()[0];

    if (!group.isParameter("RATIO") && !group.isParameter("RATE")){
        throw std::runtime_error("RATIO or RATE must be present in ROTATION.");
    }
    _ratio = group.isParameter("RATIO") ?
                group.parameter("RATIO").valuesAsInt()[0] :
                group.parameter("RATE").valuesAsDouble()[0] / c3d.header().frameRate();

    _processorType = c3d.parameters().processorType();
}

bool ezc3d::DataNS::RotationNS::Info::hasGroup() const
{
    return _hasGroup;
}

size_t ezc3d::DataNS::RotationNS::Info::dataStart() const
{
    return _dataStart;
}

size_t ezc3d::DataNS::RotationNS::Info::used() const
{
    return _used;
}

size_t ezc3d::DataNS::RotationNS::Info::ratio() const
{
    return _ratio;
}

ezc3d::PROCESSOR_TYPE ezc3d::DataNS::RotationNS::Info::processorType() const
{
    return _processorType;
}




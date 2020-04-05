#define EZC3D_API_EXPORTS
///
/// \file ForcePlatForm.cpp
/// \brief Implementation of ForcePlatForm class
/// \author Pariterre
/// \version 1.0
/// \date March 25th, 2020
///

#include "modules/ForcePlatforms.h"

#include "ezc3d_all.h"

ezc3d::Modules::ForcePlatform::ForcePlatform(
        size_t idx,
        const ezc3d::c3d& c3d)
{
    // Extract the required values from the C3D
    extractUnits(c3d);
    extractType(idx, c3d);
    extractCorners(idx, c3d);
    extractOrigin(idx, c3d);
    extractCalMatrix(idx, c3d);
    computePfReferenceFrame();
    extractData(idx, c3d);
}

const std::string& ezc3d::Modules::ForcePlatform::forceUnit() const
{
    return _unitsForce;
}

const std::string& ezc3d::Modules::ForcePlatform::momentUnit() const
{
    return _unitsMoment;
}

const std::string& ezc3d::Modules::ForcePlatform::positionUnit() const
{
    return _unitsPosition;
}

void ezc3d::Modules::ForcePlatform::extractUnits(
        const ezc3d::c3d &c3d)
{
    const ezc3d::ParametersNS::GroupNS::Group &groupPoint(
                c3d.parameters().group("POINT"));
    const ezc3d::ParametersNS::GroupNS::Group &groupPF(
                c3d.parameters().group("FORCE_PLATFORM"));

    // Position units
    if (groupPoint.isParameter("UNITS")
            && groupPoint.parameter("UNITS").dimension()[0] > 0){
        _unitsPosition = groupPoint.parameter("UNITS").valuesAsString()[0];
    }
    else {
        // Assume meter if not provided
        _unitsPosition = "m";
    }

    // Force units
    if (groupPF.isParameter("UNITS")
            && groupPF.parameter("UNITS").dimension()[0] > 0){
        _unitsForce = groupPF.parameter("UNITS").valuesAsString()[0];
    }
    else {
        // Assume Newton if not provided
        _unitsForce = "N";
    }

    // Moments units
    _unitsMoment = _unitsForce + _unitsPosition;
}

size_t ezc3d::Modules::ForcePlatform::nbFrames() const
{
    return _F.size();
}

size_t ezc3d::Modules::ForcePlatform::type() const
{
    return _type;
}

const ezc3d::Matrix66& ezc3d::Modules::ForcePlatform::calMatrix() const
{
    return _calMatrix;
}

const std::vector<ezc3d::Vector3d>&
ezc3d::Modules::ForcePlatform::corners() const
{
    return _corners;
}

const ezc3d::Vector3d&
ezc3d::Modules::ForcePlatform::meanCorners() const
{
    return _meanCorners;
}

const ezc3d::Vector3d& ezc3d::Modules::ForcePlatform::origin() const
{
    return _origin;
}

const std::vector<ezc3d::Vector3d>&
ezc3d::Modules::ForcePlatform::forces() const
{
    return _F;
}

const std::vector<ezc3d::Vector3d>&
ezc3d::Modules::ForcePlatform::moments() const
{
    return _M;
}

const std::vector<ezc3d::Vector3d>&
ezc3d::Modules::ForcePlatform::CoP() const
{
    return _CoP;
}

const std::vector<ezc3d::Vector3d>&
ezc3d::Modules::ForcePlatform::Tz() const
{
    return _Tz;
}

void ezc3d::Modules::ForcePlatform::extractType(
        size_t idx,
        const ezc3d::c3d &c3d)
{
    const ezc3d::ParametersNS::GroupNS::Group &groupPF(
                c3d.parameters().group("FORCE_PLATFORM"));

    if (groupPF.parameter("TYPE").valuesAsInt().size() < idx + 1){
        throw std::runtime_error("FORCE_PLATFORM:IDX is not fill properly "
                                 "to extract Force platform informations");
    }
    _type = static_cast<size_t>(groupPF.parameter("TYPE").valuesAsInt()[idx]);

    // Make sure that particular type is supported
    if (_type == 1){
        throw std::runtime_error("Type 1 is not supported yet, please "
                                 "open an Issue on github for support");
    }
    else if (_type == 2 || _type == 4){

    }
    else if (_type == 3 || _type == 7){
        throw std::runtime_error("Type 3 (and 7) is not supported yet, please "
                                 "open an Issue on github for support");
    }
    else if (_type == 5){
        throw std::runtime_error("Type 5 is not supported yet, please "
                                 "open an Issue on github for support");
    }
    else if (_type == 6){
        throw std::runtime_error("Type 6 is not supported yet, please "
                                 "open an Issue on github for support");
    }
    else if (_type == 11 || _type == 12){
        throw std::runtime_error("Kistler Split Belt Treadmill is not "
                                 "supported for ForcePlatform analysis");
    }
    else if (_type == 21){
        throw std::runtime_error("AMTI-stairs is not supported "
                                 "for ForcePlatform analysis");
    }
    else {
        throw std::runtime_error("Force platform type is non existant "
                                 "or not supported yet");
    }
}

ezc3d::Modules::ForcePlatforms::ForcePlatforms(
        const ezc3d::c3d &c3d)
{
    size_t nbForcePF(c3d.parameters().group("FORCE_PLATFORM")
                     .parameter("USED").valuesAsInt()[0]);
    for (size_t i=0; i<nbForcePF; ++i){
        _platforms.push_back(ezc3d::Modules::ForcePlatform(i, c3d));
    }
}

void ezc3d::Modules::ForcePlatform::extractCorners(
        size_t idx,
        const ezc3d::c3d &c3d)
{
    const ezc3d::ParametersNS::GroupNS::Group &groupPF(
                c3d.parameters().group("FORCE_PLATFORM"));

    const std::vector<double>& all_corners(
                groupPF.parameter("CORNERS").valuesAsDouble());
    if (all_corners.size() < 12*(idx+1)){
        throw std::runtime_error("FORCE_PLATFORM:CORNER is not fill properly "
                                 "to extract Force platform informations");
    }
    for (size_t i=0; i<4; ++i){
        ezc3d::Vector3d corner;
        for (size_t j=0; j<3; ++j){
            corner(j) = all_corners[idx*12 + i*3 + j];
        }
        _corners.push_back(corner);
        _meanCorners += corner;
    }
    _meanCorners /= 4;
}

void ezc3d::Modules::ForcePlatform::extractOrigin(
        size_t idx,
        const ezc3d::c3d &c3d)
{
    const ezc3d::ParametersNS::GroupNS::Group &groupPF(
                c3d.parameters().group("FORCE_PLATFORM"));

    const std::vector<double>& all_origins(
                groupPF.parameter("ORIGIN").valuesAsDouble());
    if (all_origins.size() < 3*(idx+1)){
        throw std::runtime_error("FORCE_PLATFORM:ORIGIN is not fill properly "
                                 "to extract Force platform informations");
    }
    for (size_t i=0; i<3; ++i){
        _origin(i) = all_origins[idx*3 + i];
    }

    if ((_type == 2 || _type == 4) && _origin(2) > 0.0){
        _origin = -1*_origin;
    }
}

void ezc3d::Modules::ForcePlatform::extractCalMatrix(
        size_t idx,
        const ezc3d::c3d &c3d)
{
    const ezc3d::ParametersNS::GroupNS::Group &groupPF(
                c3d.parameters().group("FORCE_PLATFORM"));

    size_t nChannels;
    if (_type == 2 || _type == 4){
        nChannels = 6;
    }

    if (!groupPF.isParameter("CAL_MATRIX")){
        if (_type == 2){
            // CAL_MATRIX is ignore for type 2
            // If none is found, returns all zeros
            return;
        }
        else {
            throw std::runtime_error(
                        "FORCE_PLATFORM:CAL_MATRIX was not found, but is "
                        "required for the type of force platform");
        }
    }

    // Check dimensions
    const auto& calMatrixParam(groupPF.parameter("CAL_MATRIX"));
    if (calMatrixParam.dimension().size() < 3
            || calMatrixParam.dimension()[2] <= idx){
        if (_type == 2){
            // CAL_MATRIX is ignore for type 2
            // If none is found, returns all zeros
            return;
        }
        else {
            throw std::runtime_error(
                        "FORCE_PLATFORM:CAL_MATRIX is not fill properly "
                        "to extract Force platform informations");
        }
    }

    const auto& val(calMatrixParam.valuesAsDouble());
    size_t skip(calMatrixParam.dimension()[0] * calMatrixParam.dimension()[1]);
    for (size_t i=0; i<nChannels; ++i){
        for (size_t j=0; j<nChannels; ++j){
            _calMatrix(i, j) = val[skip*idx + j*nChannels + i];
        }
    }
}

void ezc3d::Modules::ForcePlatform::computePfReferenceFrame()
{
    ezc3d::Vector3d axisX(_corners[0] - _corners[1]);
    ezc3d::Vector3d axisY(_corners[0] - _corners[3]);
    ezc3d::Vector3d axisZ(axisX.cross(axisY));
    axisY = axisZ.cross(axisX);

    axisX.normalize();
    axisY.normalize();
    axisZ.normalize();

    for (size_t i=0; i<3; ++i){
        _refFrame(i, 0) = axisX(i);
        _refFrame(i, 1) = axisY(i);
        _refFrame(i, 2) = axisZ(i);
    }
}

void ezc3d::Modules::ForcePlatform::extractData(
        size_t idx,
        const ezc3d::c3d &c3d)
{
    const ezc3d::ParametersNS::GroupNS::Group &groupPF(
                c3d.parameters().group("FORCE_PLATFORM"));

    // Get elements from the force platform's type
    size_t nChannels;
    if (_type == 2 || _type == 4){
        nChannels = 6;
    }

    // Check the dimensions of FORCE_PLATFORM:CHANNEL are consistent
    const std::vector<size_t>& dimensions(groupPF.parameter("CHANNEL").dimension());
    if (dimensions[0] < nChannels){
        throw std::runtime_error("FORCE_PLATFORM:CHANNEL is not fill properly "
                                 "to extract Force platform informations");
    }
    if (dimensions[1] < idx + 1){
        throw std::runtime_error("FORCE_PLATFORM:CHANNEL is not fill properly "
                                 "to extract Force platform informations");
    }

    // Get the channels where the force platform are stored in the data
    std::vector<size_t> channel_idx(nChannels);
    const std::vector<int>& all_channel_idx(
                groupPF.parameter("CHANNEL").valuesAsInt());
    for (size_t i=0; i<nChannels; ++i){
        channel_idx[i] = all_channel_idx[idx*dimensions[0] + i] - 1;  // 1-based
    }

    // Get the force and moment from these channel in global reference frame
    size_t nFramesTotal(
                c3d.header().nbFrames()
                * c3d.header().nbAnalogByFrame());
    _F.resize(nFramesTotal);
    _M.resize(nFramesTotal);
    _CoP.resize(nFramesTotal);
    _Tz.resize(nFramesTotal);
    size_t cmp(0);
    for (const auto& frame : c3d.data().frames()){
        for (size_t i=0; i<frame.analogs().nbSubframes(); ++i){
            const auto& subframe(frame.analogs().subframe(i));

            if (_type == 2 || _type == 4){
                ezc3d::Vector6d data_raw;
                ezc3d::Vector3d force_raw;
                ezc3d::Vector3d moment_raw;
                for (size_t j=0; j<3; ++j){
                    data_raw(j) = subframe.channel(channel_idx[j]).data();
                    data_raw(j+3) = subframe.channel(channel_idx[j+3]).data();
                }
                if (_type == 4){
                    data_raw = _calMatrix * data_raw;
                }
                for (size_t j=0; j<3; ++j){
                    force_raw(j) = data_raw(j);
                    moment_raw(j) = data_raw(j+3);
                }
                _F[cmp] = _refFrame * force_raw;
                moment_raw += force_raw.cross(_origin);
                _M[cmp] = _refFrame * moment_raw;

                ezc3d::Vector3d CoP_raw(
                            -moment_raw(1)/force_raw(2),
                            moment_raw(0)/force_raw(2),
                            0);
                _CoP[cmp] = _refFrame * CoP_raw + _meanCorners;
                _Tz[cmp] = _refFrame * static_cast<Vector3d>(
                            moment_raw - force_raw.cross(-1*CoP_raw));
                ++cmp;
            }

        }
    }
}

const std::vector<ezc3d::Modules::ForcePlatform>&
ezc3d::Modules::ForcePlatforms::forcePlatforms() const
{
    return _platforms;
}

const ezc3d::Modules::ForcePlatform&
ezc3d::Modules::ForcePlatforms::forcePlatform(
        size_t idx) const
{
    return _platforms.at(idx);
}

#define EZC3D_API_EXPORTS
///
/// \file Group.cpp
/// \brief Implementation of Group class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "Group.h"
#include "Parameters.h"

ezc3d::ParametersNS::GroupNS::Group::Group(
        const std::string &name,
        const std::string &description) :
    _name(name),
    _description(description),
    _isLocked(false) {

}

void ezc3d::ParametersNS::GroupNS::Group::print() const {
    std::cout << "groupName = " << name() << std::endl;
    std::cout << "isLocked = " << isLocked() << std::endl;
    std::cout << "desc = " << description() << std::endl;

    for (size_t i=0; i < nbParameters(); ++i){
        std::cout << "Parameter " << i << std::endl;
        parameter(i).print();
    }
}

void ezc3d::ParametersNS::GroupNS::Group::write(
        std::fstream &f,
        int groupIdx,
        std::streampos &dataStartPosition) const {
    int nCharName(static_cast<int>(name().size()));
    if (isLocked())
        nCharName *= -1;
    f.write(reinterpret_cast<const char*>(&nCharName),
            1*ezc3d::DATA_TYPE::BYTE);
    if (isLocked())
        nCharName *= -1;
    f.write(reinterpret_cast<const char*>(&groupIdx), 1*ezc3d::DATA_TYPE::BYTE);
    f.write(name().c_str(), nCharName*ezc3d::DATA_TYPE::BYTE);

    // It is not possible already to know in
    // how many bytes the next parameter is
    int blank(0);
    std::streampos pos(f.tellg());
    f.write(reinterpret_cast<const char*>(&blank), 2*ezc3d::DATA_TYPE::BYTE);

    int nCharGroupDescription(static_cast<int>(description().size()));
    f.write(reinterpret_cast<const char*>(&nCharGroupDescription),
            1*ezc3d::DATA_TYPE::BYTE);
    f.write(description().c_str(),
            nCharGroupDescription*ezc3d::DATA_TYPE::BYTE);

    std::streampos currentPos(f.tellg());
    // Go back at the left blank space and write the current position
    f.seekg(pos);
    int nCharToNext = int(currentPos - pos);
    f.write(reinterpret_cast<const char*>(&nCharToNext),
            2*ezc3d::DATA_TYPE::BYTE);
    f.seekg(currentPos);

    std::streampos defaultDataStartPosition(-1);
    for (size_t i=0; i < nbParameters(); ++i)
        if (!name().compare("POINT"))
            parameter(i).write(f, -groupIdx, dataStartPosition);
        else
            parameter(i).write(f, -groupIdx, defaultDataStartPosition);
}

int ezc3d::ParametersNS::GroupNS::Group::read(
        ezc3d::c3d &c3d,
        const ezc3d::ParametersNS::Parameters &params,
        std::fstream &file, int nbCharInName) {
    if (nbCharInName < 0)
        _isLocked = true;
    else
        _isLocked = false;

    // Read name of the group
    _name.assign(
                c3d.readString(
                    file, static_cast<unsigned int>(
                        abs(nbCharInName) * ezc3d::DATA_TYPE::BYTE)));

    // number of byte to the next group from here
    size_t offsetNext(
                c3d.readUint(
                    params.processorType(), file, 2*ezc3d::DATA_TYPE::BYTE));
    // Compute the position of the element in the file
    int nextParamByteInFile;
    if (offsetNext == 0)
        nextParamByteInFile = 0;
    else
        nextParamByteInFile = static_cast<int>(
                    static_cast<size_t>(
                        file.tellg()) + offsetNext - ezc3d::DATA_TYPE::WORD);

    // Byte 5+nbCharInName ==> Number of characters in group description
    int nbCharInDesc(
                c3d.readInt(
                    params.processorType(), file, 1*ezc3d::DATA_TYPE::BYTE));
    // Byte 6+nbCharInName ==> Group description
    if (nbCharInDesc)
        _description = c3d.readString(
                    file, static_cast<unsigned int>(nbCharInDesc));

    // Return how many bytes
    return nextParamByteInFile;
}

bool ezc3d::ParametersNS::GroupNS::Group::isEmpty() const
{
    if (!name().compare("") && nbParameters() == 0) {
        return true;
    }
    else {
        return false;
    }
}

const std::string& ezc3d::ParametersNS::GroupNS::Group::name() const {
    return _name;
}

void ezc3d::ParametersNS::GroupNS::Group::name(
        const std::string &name) {
    _name = name;
}

const std::string& ezc3d::ParametersNS::GroupNS::Group::description() const {
    return _description;
}

void ezc3d::ParametersNS::GroupNS::Group::description(
        const std::string &description) {
    _description = description;
}

bool ezc3d::ParametersNS::GroupNS::Group::isLocked() const {
    return _isLocked;
}

void ezc3d::ParametersNS::GroupNS::Group::lock() {
    _isLocked = true;
}
void ezc3d::ParametersNS::GroupNS::Group::unlock() {
    _isLocked = false;
}

size_t ezc3d::ParametersNS::GroupNS::Group::nbParameters() const {
    return _parameters.size();
}

bool ezc3d::ParametersNS::GroupNS::Group::isParameter(
        const std::string &parameterName) const
{
    try {
        parameterIdx(parameterName);
        return true;
    } catch (std::invalid_argument) {
        return false;
    }
}

size_t ezc3d::ParametersNS::GroupNS::Group::parameterIdx(
        const std::string &parameterName) const {
    for (size_t i = 0; i < nbParameters(); ++i)
        if (!parameter(i).name().compare(parameterName))
            return i;
    throw std::invalid_argument(
                "Group::parameterIdx could not find "
                + parameterName + " in the group " + name());
}

const ezc3d::ParametersNS::GroupNS::Parameter
&ezc3d::ParametersNS::GroupNS::Group::parameter(
        size_t idx) const {
    try {
        return _parameters.at(idx);
    } catch(std::out_of_range) {
        throw std::out_of_range(
                    "Groups::parameter method is trying "
                    "to access the parameter "
                    + std::to_string(idx) +
                    " while the maximum number of parameter is "
                    + std::to_string(nbParameters()) +
                    " in the group " + name() + ".");
    }
}

ezc3d::ParametersNS::GroupNS::Parameter
&ezc3d::ParametersNS::GroupNS::Group::parameter(
        size_t idx) {
    try {
        return _parameters.at(idx);
    } catch(std::out_of_range) {
        throw std::out_of_range(
                    "Groups::parameter method is trying "
                    "to access the parameter "
                    + std::to_string(idx) +
                    " while the maximum number of parameters is "
                    + std::to_string(nbParameters())
                    + " in the group " + name() + ".");
    }
}

const ezc3d::ParametersNS::GroupNS::Parameter
&ezc3d::ParametersNS::GroupNS::Group::parameter(
        const std::string &parameterName) const {
    return parameter(parameterIdx(parameterName));
}

ezc3d::ParametersNS::GroupNS::Parameter
&ezc3d::ParametersNS::GroupNS::Group::parameter(
        const std::string &parameterName) {
    return parameter(parameterIdx(parameterName));
}

int ezc3d::ParametersNS::GroupNS::Group::parameter(
        ezc3d::c3d &c3d,
        const Parameters &params,
        std::fstream &file,
        int nbCharInName) {
    ezc3d::ParametersNS::GroupNS::Parameter p;
    int nextParamByteInFile = p.read(c3d, params, file, nbCharInName);
    parameter(p);
    return nextParamByteInFile;
}

void ezc3d::ParametersNS::GroupNS::Group::parameter(
        const ezc3d::ParametersNS::GroupNS::Parameter &p) {
    if (p.type() == ezc3d::DATA_TYPE::NO_DATA_TYPE)
        throw std::runtime_error("Data type is not set");

    size_t alreadyExistIdx(SIZE_MAX);
    for (size_t i=0; i < _parameters.size(); ++i)
        if (!parameter(i).name().compare(p.name())){
            alreadyExistIdx = i;
            break;
        }
    if (alreadyExistIdx == SIZE_MAX)
        _parameters.push_back(p);
    else
        _parameters[alreadyExistIdx] = p;
}

const std::vector<ezc3d::ParametersNS::GroupNS::Parameter>&
ezc3d::ParametersNS::GroupNS::Group::parameters() const {
    return _parameters;
}

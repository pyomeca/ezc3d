#define EZC3D_API_EXPORTS
#include "Group.h"
// Implementation of Group class


ezc3d::ParametersNS::GroupNS::Group::Group(const std::string &name, const std::string &description) :
    _name(name),
    _description(description),
    _isLocked(false)
{

}

void ezc3d::ParametersNS::GroupNS::Group::print() const
{
    std::cout << "groupName = " << name() << std::endl;
    std::cout << "isLocked = " << isLocked() << std::endl;
    std::cout << "desc = " << description() << std::endl;

    for (size_t i=0; i < nbParameters(); ++i){
        std::cout << "Parameter " << i << std::endl;
        parameter(i).print();
    }
}

void ezc3d::ParametersNS::GroupNS::Group::write(std::fstream &f, int groupIdx, std::streampos &dataStartPosition) const
{
    int nCharName(static_cast<int>(name().size()));
    if (isLocked())
        nCharName *= -1;
    f.write(reinterpret_cast<const char*>(&nCharName), 1*ezc3d::DATA_TYPE::BYTE);
    if (isLocked())
        nCharName *= -1;
    f.write(reinterpret_cast<const char*>(&groupIdx), 1*ezc3d::DATA_TYPE::BYTE);
    f.write(ezc3d::toUpper(name()).c_str(), nCharName*ezc3d::DATA_TYPE::BYTE);

    // It is not possible already to know in how many bytes the next parameter is
    int blank(0);
    std::streampos pos(f.tellg());
    f.write(reinterpret_cast<const char*>(&blank), 2*ezc3d::DATA_TYPE::BYTE);

    int nCharGroupDescription(static_cast<int>(description().size()));
    f.write(reinterpret_cast<const char*>(&nCharGroupDescription), 1*ezc3d::DATA_TYPE::BYTE);
    f.write(description().c_str(), nCharGroupDescription*ezc3d::DATA_TYPE::BYTE);

    std::streampos actualPos(f.tellg());
    // Go back at the left blank space and write the actual position
    f.seekg(pos);
    int nCharToNext = int(actualPos - pos);
    f.write(reinterpret_cast<const char*>(&nCharToNext), 2*ezc3d::DATA_TYPE::BYTE);
    f.seekg(actualPos);

    for (size_t i=0; i < nbParameters(); ++i)
        parameter(i).write(f, -groupIdx, dataStartPosition);

}

int ezc3d::ParametersNS::GroupNS::Group::read(ezc3d::c3d &file, int nbCharInName)
{
    if (nbCharInName < 0)
        _isLocked = true;
    else
        _isLocked = false;

    // Read name of the group
    _name.assign(file.readString(static_cast<unsigned int>(abs(nbCharInName) * ezc3d::DATA_TYPE::BYTE)));

    // number of byte to the next group from here
    size_t offsetNext(file.readUint(2*ezc3d::DATA_TYPE::BYTE));
    // Compute the position of the element in the file
    int nextParamByteInFile;
    if (offsetNext == 0)
        nextParamByteInFile = 0;
    else
        nextParamByteInFile = static_cast<int>(static_cast<size_t>(file.tellg()) + offsetNext - ezc3d::DATA_TYPE::WORD);

    // Byte 5+nbCharInName ==> Number of characters in group description
    int nbCharInDesc(file.readInt(1*ezc3d::DATA_TYPE::BYTE));
    // Byte 6+nbCharInName ==> Group description
    if (nbCharInDesc)
        _description = file.readString(static_cast<unsigned int>(nbCharInDesc));

    // Return how many bytes
    return nextParamByteInFile;
}

const std::string& ezc3d::ParametersNS::GroupNS::Group::name() const
{
    return _name;
}

void ezc3d::ParametersNS::GroupNS::Group::name(const std::string name)
{
    _name = name;
}

const std::string& ezc3d::ParametersNS::GroupNS::Group::description() const
{
    return _description;
}

void ezc3d::ParametersNS::GroupNS::Group::description(const std::string description)
{
    _description = description;
}

bool ezc3d::ParametersNS::GroupNS::Group::isLocked() const
{
    return _isLocked;
}

void ezc3d::ParametersNS::GroupNS::Group::lock()
{
    _isLocked = true;
}
void ezc3d::ParametersNS::GroupNS::Group::unlock()
{
    _isLocked = false;
}

size_t ezc3d::ParametersNS::GroupNS::Group::nbParameters() const
{
    return _parameters.size();
}

size_t ezc3d::ParametersNS::GroupNS::Group::parameterIdx(std::string parameterName) const
{
    for (size_t i = 0; i < nbParameters(); ++i)
        if (!parameter(i).name().compare(parameterName))
            return i;
    throw std::invalid_argument("Group::parameterIdx could not find " + parameterName +
                                " in the group " + name());
}

const ezc3d::ParametersNS::GroupNS::Parameter &ezc3d::ParametersNS::GroupNS::Group::parameter(size_t idx) const
{
    try {
        return _parameters.at(idx);
    } catch(std::out_of_range) {
        throw std::out_of_range("Groups::parameter method is trying to access the parameter "
                                + std::to_string(idx) +
                                " while the maximum number of parameter is "
                                + std::to_string(nbParameters()) + " in the group " + name() + ".");
    }
}

ezc3d::ParametersNS::GroupNS::Parameter &ezc3d::ParametersNS::GroupNS::Group::parameter_nonConst(size_t idx)
{
    try {
        return _parameters.at(idx);
    } catch(std::out_of_range) {
        throw std::out_of_range("Groups::parameter method is trying to access the parameter "
                                + std::to_string(idx) +
                                " while the maximum number of parameters is "
                                + std::to_string(nbParameters()) + " in the group " + name() + ".");
    }
}

const ezc3d::ParametersNS::GroupNS::Parameter &ezc3d::ParametersNS::GroupNS::Group::parameter(std::string parameterName) const
{
    return parameter(parameterIdx(parameterName));
}

ezc3d::ParametersNS::GroupNS::Parameter &ezc3d::ParametersNS::GroupNS::Group::parameter_nonConst(std::string parameterName)
{
    return parameter_nonConst(parameterIdx(parameterName));
}

int ezc3d::ParametersNS::GroupNS::Group::parameter(ezc3d::c3d &file, int nbCharInName)
{
    ezc3d::ParametersNS::GroupNS::Parameter p;
    int nextParamByteInFile = p.read(file, nbCharInName);
    parameter(p);
    return nextParamByteInFile;
}

void ezc3d::ParametersNS::GroupNS::Group::parameter(const ezc3d::ParametersNS::GroupNS::Parameter &p)
{
    if (p.type() == ezc3d::DATA_TYPE::NONE)
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

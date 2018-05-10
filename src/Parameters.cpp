#include "Parameters.h"
ezC3D_NAMESPACE::Parameters::Parameters(ezC3D &file) :
    _parametersStart(0),
    _checksum(0),
    _nbParamBlock(0),
    _processorType(0)
{
    // Read the Parameters Header
    _parametersStart = file.readInt(1*ezC3D_NAMESPACE::READ_SIZE::BYTE, 256*ezC3D_NAMESPACE::READ_SIZE::WORD*(file.header().parametersAddress()-1), std::ios::beg);
    _checksum = file.readInt(1*ezC3D_NAMESPACE::READ_SIZE::BYTE);
    _nbParamBlock = file.readInt(1*ezC3D_NAMESPACE::READ_SIZE::BYTE);
    _processorType = file.readInt(1*ezC3D_NAMESPACE::READ_SIZE::BYTE);

    // Read parameter or group
    int nextParamByteInFile((int)file.tellg() + _parametersStart - ezC3D_NAMESPACE::READ_SIZE::BYTE);
    while (nextParamByteInFile)
    {
        // Check if we spontaneously got to the next parameter. Otherwise c3d is messed up
        if (file.tellg() != nextParamByteInFile)
            throw std::ios_base::failure("Bad c3d formatting");

        // Nb of char in the group name, locked if negative, 0 if we finished the section
        int nbCharInName(file.readInt(1*ezC3D_NAMESPACE::READ_SIZE::BYTE));
        if (nbCharInName == 0)
            break;
        int id(file.readInt(1*ezC3D_NAMESPACE::READ_SIZE::BYTE));

        // Make sure there at least enough group
        for (int i = _groups.size(); i < abs(id); ++i)
            _groups.push_back(ezC3D_NAMESPACE::Parameters::Group());

        // Group ID always negative for groups and positive parameter of group ID
        if (id < 0)
            nextParamByteInFile = group_nonConst(abs(id)-1).read(file, nbCharInName);
        else
            nextParamByteInFile = group_nonConst(id-1).addParameter(file, nbCharInName);
    }
}
int ezC3D_NAMESPACE::Parameters::processorType() const
{
    return _processorType;
}
int ezC3D_NAMESPACE::Parameters::nbParamBlock() const
{
    return _nbParamBlock;
}
int ezC3D_NAMESPACE::Parameters::checksum() const
{
    return _checksum;
}
int ezC3D_NAMESPACE::Parameters::parametersStart() const
{
    return _parametersStart;
}
void ezC3D_NAMESPACE::Parameters::print() const
{
    std::cout << "Parameters header" << std::endl;
    std::cout << "parametersStart = " << parametersStart() << std::endl;
    std::cout << "nbParamBlock = " << nbParamBlock() << std::endl;
    std::cout << "processorType = " << processorType() << std::endl;

    for (int i = 0; i < groups().size(); ++i){
        std::cout << "Group " << i << std::endl;
        group(i).print();
        std::cout << std::endl;
    }
    std::cout << std::endl;
}




ezC3D_NAMESPACE::Parameters::Group::Group()
{

}
const std::vector<ezC3D_NAMESPACE::Parameters::Group>& ezC3D_NAMESPACE::Parameters::groups() const
{
    return _groups;
}
ezC3D_NAMESPACE::Parameters::Group &ezC3D_NAMESPACE::Parameters::group_nonConst(int group)
{
    return _groups[group];
}
const ezC3D_NAMESPACE::Parameters::Group &ezC3D_NAMESPACE::Parameters::group(int group) const
{
    return _groups[group];
}
const ezC3D_NAMESPACE::Parameters::Group &ezC3D_NAMESPACE::Parameters::group(const std::string &groupName) const
{
    for (int i = 0; i < groups().size(); ++i){
        if (!group(i).name().compare(groupName))
            return group(i);
    }
    throw std::invalid_argument("Group name was not found in parameters");
}
void ezC3D_NAMESPACE::Parameters::Group::lock()
{
    _isLocked = true;
}
void ezC3D_NAMESPACE::Parameters::Group::unlock()
{
    _isLocked = false;
}
bool ezC3D_NAMESPACE::Parameters::Group::isLocked() const
{
    return _isLocked;
}
const std::string& ezC3D_NAMESPACE::Parameters::Group::description() const
{
    return _description;
}
const std::string& ezC3D_NAMESPACE::Parameters::Group::name() const
{
    return _name;
}
int ezC3D_NAMESPACE::Parameters::Group::read(ezC3D &file, int nbCharInName)
{
    if (nbCharInName < 0)
        _isLocked = true;
    else
        _isLocked = false;

    // Read name of the group
    _name.assign(file.readString(abs(nbCharInName) * ezC3D_NAMESPACE::READ_SIZE::BYTE));

    // number of byte to the next group from here
    int offsetNext((int)file.readUint(2*ezC3D_NAMESPACE::READ_SIZE::BYTE));
    // Compute the position of the element in the file
    int nextParamByteInFile;
    if (offsetNext == 0)
        nextParamByteInFile = 0;
    else
        nextParamByteInFile = (int)file.tellg() + offsetNext - ezC3D_NAMESPACE::READ_SIZE::WORD;

    // Byte 5+nbCharInName ==> Number of characters in group description
    int nbCharInDesc(file.readInt(1*ezC3D_NAMESPACE::READ_SIZE::BYTE));
    // Byte 6+nbCharInName ==> Group description
    if (nbCharInDesc)
        _description = file.readString(nbCharInDesc);

    // Return how many bytes
    return nextParamByteInFile;
}
int ezC3D_NAMESPACE::Parameters::Group::addParameter(ezC3D &file, int nbCharInName)
{
    ezC3D_NAMESPACE::Parameters::Group::Parameter p;
    int nextParamByteInFile = p.read(file, nbCharInName);
    _parameters.push_back(p);
    return nextParamByteInFile;
}
void ezC3D_NAMESPACE::Parameters::Group::print() const
{
    std::cout << "groupName = " << name() << std::endl;
    std::cout << "isLocked = " << isLocked() << std::endl;
    std::cout << "desc = " << description() << std::endl;

    for (int i=0; i<parameters().size(); ++i){
        std::cout << "Parameter " << i << std::endl;
        parameter(i).print();
    }
}
const std::vector<ezC3D_NAMESPACE::Parameters::Group::Parameter>& ezC3D_NAMESPACE::Parameters::Group::parameters() const
{
    return _parameters;
}

const ezC3D_NAMESPACE::Parameters::Group::Parameter &ezC3D_NAMESPACE::Parameters::Group::parameter(int idx) const
{
    if (idx < 0 || idx >= _parameters.size())
        throw std::out_of_range("Wrong number of parameter");
    return _parameters[idx];
}
std::vector<ezC3D_NAMESPACE::Parameters::Group::Parameter>& ezC3D_NAMESPACE::Parameters::Group::parameters_nonConst()
{
    return _parameters;
}

const ezC3D_NAMESPACE::Parameters::Group::Parameter &ezC3D_NAMESPACE::Parameters::Group::parameter(std::string parameterName) const
{
    for (int i = 0; i < parameters().size(); ++i){
        if (!parameter(i).name().compare(parameterName))
            return parameter(i);
    }
    throw std::invalid_argument("Parameter name was not found within the group");
}




ezC3D_NAMESPACE::Parameters::Group::Parameter::Parameter()
{

}
void ezC3D_NAMESPACE::Parameters::Group::Parameter::lock()
{
    _isLocked = true;
}
void ezC3D_NAMESPACE::Parameters::Group::Parameter::unlock()
{
    _isLocked = false;
}
bool ezC3D_NAMESPACE::Parameters::Group::Parameter::isLocked() const
{
    return _isLocked;
}
const std::string& ezC3D_NAMESPACE::Parameters::Group::Parameter::description() const
{
    return _description;
}
const std::string& ezC3D_NAMESPACE::Parameters::Group::Parameter::name() const
{
    return _name;
}
int ezC3D_NAMESPACE::Parameters::Group::Parameter::read(ezC3D &file, int nbCharInName)
{
    if (nbCharInName < 0)
        _isLocked = true;
    else
        _isLocked = false;

    // Read name of the group
    _name = file.readString(abs(nbCharInName) * ezC3D_NAMESPACE::READ_SIZE::BYTE);

    // number of byte to the next group from here
    int offsetNext((int)file.readUint(2*ezC3D_NAMESPACE::READ_SIZE::BYTE));
    // Compute the position of the element in the file
    int nextParamByteInFile;
    if (offsetNext == 0)
        nextParamByteInFile = 0;
    else
        nextParamByteInFile = (int)file.tellg() + offsetNext - ezC3D_NAMESPACE::READ_SIZE::WORD;

    // -1 sizeof(char), 1 byte, 2 int, 4 float
    int lengthInByte(file.readInt(1*ezC3D_NAMESPACE::READ_SIZE::BYTE));
    if (lengthInByte == -1)
        _data_type = DATA_TYPE::CHAR;
    else if (lengthInByte == 1)
        _data_type = DATA_TYPE::BYTE;
    else if (lengthInByte == 2)
        _data_type = DATA_TYPE::INT;
    else if (lengthInByte == 4)
        _data_type = DATA_TYPE::FLOAT;
    else
        throw std::ios_base::failure ("Parameter type unrecognized");

    // number of dimension of parameter (0 for scalar)
    int nDimensions(file.readInt(1*ezC3D_NAMESPACE::READ_SIZE::BYTE));
    if (nDimensions == 0) // In the special case of a scalar
        _dimension.push_back(1);
    else // otherwise it's a matrix
        for (int i=0; i<nDimensions; ++i)
            _dimension.push_back (file.readInt(1*ezC3D_NAMESPACE::READ_SIZE::BYTE));    // Read the dimension size of the matrix

    // Read the data for the parameters
    if (_data_type == DATA_TYPE::CHAR){
        std::vector<std::string> param_data_string_tp;
        file.readMatrix(_dimension, param_data_string_tp);
        // Vicon c3d organize its text in column-wise format, I am not sure if
        // this is a standard or a custom made stuff
        if (_dimension.size() == 1){
            std::string tp;
            for (int i = 0; i < _dimension[0]; ++i)
                tp += param_data_string_tp[i];
            _param_data_string.push_back(tp);
        }
        else if (_dimension.size() == 2){
            int idx(0);
            for (int i = 0; i < _dimension[1]; ++i){
                std::string tp;
                for (int j = 0; j < _dimension[0]; ++j){
                    tp += param_data_string_tp[idx];
                    ++idx;
                }
                _param_data_string.push_back(tp);
            }
        }
        else
            throw std::ios_base::failure("Parsing char on matrix other than 2d or 1d matrix is not implemented yet");
    }
    else if (_data_type == DATA_TYPE::BYTE)
        file.readMatrix((int)_data_type, _dimension, _param_data_int);
    else if (_data_type == DATA_TYPE::INT)
        file.readMatrix((int)_data_type, _dimension, _param_data_int);
    else if (_data_type == DATA_TYPE::FLOAT)
        file.readMatrix(_dimension, _param_data_float);


    // Byte 5+nbCharInName ==> Number of characters in group description
    int nbCharInDesc(file.readInt(1*ezC3D_NAMESPACE::READ_SIZE::BYTE));
    // Byte 6+nbCharInName ==> Group description
    if (nbCharInDesc)
        _description = file.readString(nbCharInDesc);

    // Return how many bytes
    return nextParamByteInFile;
}

void ezC3D_NAMESPACE::Parameters::Group::Parameter::print() const
{
    std::cout << "parameterName = " << name() << std::endl;
    std::cout << "isLocked = " << isLocked() << std::endl;

    // Data are not separated according to _dimension, which could help to read
    if (_data_type == DATA_TYPE::CHAR)
        for (int i = 0; i < _param_data_string.size(); ++i)
            std::cout << "param_data_string[" << i << "] = " << _param_data_string[i] << std::endl;
    if (_data_type == DATA_TYPE::BYTE)
        for (int i = 0; i < _param_data_int.size(); ++i)
            std::cout << "param_data[" << i << "] = " << _param_data_int[i] << std::endl;
    if (_data_type == DATA_TYPE::INT)
        for (int i = 0; i < _param_data_int.size(); ++i)
            std::cout << "param_data[" << i << "] = " << _param_data_int[i] << std::endl;
    if (_data_type == DATA_TYPE::FLOAT)
        for (int i = 0; i < _param_data_float.size(); ++i)
            std::cout << "param_data[" << i << "] = " << _param_data_float[i] << std::endl;

    std::cout << "description = " << _description << std::endl;
}


const std::vector<std::string>& ezC3D_NAMESPACE::Parameters::Group::Parameter::stringValues() const
{
    if (_data_type != DATA_TYPE::CHAR)
        throw std::invalid_argument("This parameter is not string");
    return _param_data_string;
}

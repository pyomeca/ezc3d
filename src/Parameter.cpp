#define EZC3D_API_EXPORTS
///
/// \file Parameter.cpp
/// \brief Implementation of Parameter class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "Parameter.h"
#include "Parameters.h"

ezc3d::ParametersNS::GroupNS::Parameter::Parameter(
        const std::string &name,
        const std::string &description) :
    _name(name),
    _description(description),
    _isLocked(false),
    _data_type(ezc3d::DATA_TYPE::NO_DATA_TYPE),
    _isEmpty(true) {

}

void ezc3d::ParametersNS::GroupNS::Parameter::print() const {
    std::cout << "parameterName = " << name() << std::endl;
    std::cout << "isLocked = " << isLocked() << std::endl;

    // Data are not separated according to _dimension, which could help to read
    if (_data_type == DATA_TYPE::CHAR)
        for (unsigned int i = 0; i < _param_data_string.size(); ++i)
            std::cout << "param_data_string[" << i << "] = "
                      << _param_data_string[i] << std::endl;
    if (_data_type == DATA_TYPE::BYTE)
        for (unsigned int i = 0; i < _param_data_int.size(); ++i)
            std::cout << "param_data[" << i << "] = "
                      << _param_data_int[i] << std::endl;
    if (_data_type == DATA_TYPE::INT)
        for (unsigned int i = 0; i < _param_data_int.size(); ++i)
            std::cout << "param_data[" << i << "] = "
                      << _param_data_int[i] << std::endl;
    if (_data_type == DATA_TYPE::FLOAT)
        for (unsigned int i = 0; i < _param_data_double.size(); ++i)
            std::cout << "param_data[" << i << "] = "
                      << _param_data_double[i] << std::endl;

    std::cout << "description = " << _description << std::endl;
}

void ezc3d::ParametersNS::GroupNS::Parameter::write(
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
    f.write(reinterpret_cast<const char*>(&groupIdx),
            1*ezc3d::DATA_TYPE::BYTE);
    f.write(name().c_str(), nCharName*ezc3d::DATA_TYPE::BYTE);

    // It is not possible already to know in how many
    // bytes the next parameter is
    int blank(0);
    std::streampos pos(f.tellg());
    f.write(reinterpret_cast<const char*>(&blank),
            2*ezc3d::DATA_TYPE::BYTE);

    // Recalculate the dimension of the values
    std::vector<size_t> dimension(_dimension);
    if (_data_type == DATA_TYPE::CHAR) {
        // Assusimng dimension[0] is the number of characters
        // and dimension[1] is the number of string
        dimension[0] = longestElement();

        // Remove unecessary dimension
        if (dimension.size() == 2 && dimension[1] == 1) {
            dimension = {dimension[0]};
        }
    }

    // Write the parameter values
    f.write(reinterpret_cast<const char*>(&_data_type),
            1*ezc3d::DATA_TYPE::BYTE);
    size_t size_dim(dimension.size());
    // If it is a scalar, store it as so
    if (dimension.size() == 1 && dimension[0] == 1
            && _data_type != DATA_TYPE::CHAR){
        int _size_dim(0);
        f.write(reinterpret_cast<const char*>(&_size_dim),
                1*ezc3d::DATA_TYPE::BYTE);
    }
    else{
        f.write(reinterpret_cast<const char*>(&size_dim),
                1*ezc3d::DATA_TYPE::BYTE);
        for (unsigned int i=0; i<dimension.size(); ++i)
            f.write(reinterpret_cast<const char*>(&dimension[i]),
                    1*ezc3d::DATA_TYPE::BYTE);
    }

    int hasSize(0);
    if (dimension.size() > 0){
        hasSize = 1;
        for (unsigned int i=0; i<dimension.size(); ++i)
            hasSize *= dimension[i];
    }
    if (hasSize > 0){
        if (_data_type == DATA_TYPE::CHAR){
            if (dimension.size() == 1){
                f.write(_param_data_string[0].c_str(),
                        static_cast<int>(
                            _param_data_string[0].size())
                        * static_cast<int>(DATA_TYPE::BYTE));
            } else {
                writeImbricatedParameter(f, dimension, 1);
            }
        } else {
            if (static_cast<int>(dataStartPosition) != -1
                    && !_name.compare("DATA_START")){
                // -1 means that it is not the POINT group

                // This is a special case defined in the standard where you write
                // the number of blocks up to the data
                dataStartPosition = f.tellg();
                f.write(reinterpret_cast<const char*>(&blank),
                        2*ezc3d::DATA_TYPE::BYTE);
            } else
                writeImbricatedParameter(f, dimension);
        }
    }

    // Write description of the parameter
    int nCharDescription(static_cast<int>(description().size()));
    f.write(reinterpret_cast<const char*>(&nCharDescription),
            1*ezc3d::DATA_TYPE::BYTE);
    f.write(description().c_str(),
            nCharDescription*ezc3d::DATA_TYPE::BYTE);

    // Go back at the left blank space and write the current position
    std::streampos currentPos(f.tellg());
    f.seekg(pos);
    int nCharToNext = int(currentPos - pos);
    f.write(reinterpret_cast<const char*>(&nCharToNext),
            2*ezc3d::DATA_TYPE::BYTE);
    f.seekg(currentPos);
}

size_t ezc3d::ParametersNS::GroupNS::Parameter::writeImbricatedParameter(
        std::fstream &f,
        const std::vector<size_t>& dim,
        size_t currentIdx, size_t cmp) const {
    for (size_t i=0; i<dim[currentIdx]; ++i)
        if (currentIdx == dim.size()-1){
            if (_data_type == DATA_TYPE::BYTE)
                f.write(reinterpret_cast<const char*>(&(_param_data_int[cmp])),
                        static_cast<int>(_data_type));
            else if (_data_type == DATA_TYPE::INT)
                f.write(reinterpret_cast<const char*>(&(_param_data_int[cmp])),
                        static_cast<int>(_data_type));
            else if (_data_type == DATA_TYPE::FLOAT){
                float param(static_cast<float>(_param_data_double[cmp]));
                f.write(reinterpret_cast<const char*>(&param),
                        static_cast<int>(_data_type));
            }
            else if (_data_type == DATA_TYPE::CHAR){
                std::string toWrite(_param_data_string[cmp]);
                toWrite.resize(dim[0], ' '); // Pad with x20
                f.write(toWrite.c_str(),
                        static_cast<int>(dim[0] * DATA_TYPE::BYTE));
            }
            ++cmp;
        }
        else
            cmp = writeImbricatedParameter(f, dim, currentIdx + 1, cmp);
    return cmp;
}

int ezc3d::ParametersNS::GroupNS::Parameter::read(
        ezc3d::c3d &c3d,
        const Parameters &params,
        std::fstream &file,
        int nbCharInName) {
    if (nbCharInName < 0)
        _isLocked = true;
    else
        _isLocked = false;

    // Read name of the group
    _name = c3d.readString(file,
                           static_cast<unsigned int>(abs(nbCharInName)
                                                     * ezc3d::DATA_TYPE::BYTE));

    // number of byte to the next group from here
    int offsetNext(static_cast<int>(c3d.readUint(params.processorType(), file,
                                                 2*ezc3d::DATA_TYPE::BYTE)));
    // Compute the position of the element in the file
    int nextParamByteInFile;
    if (offsetNext == 0)
        nextParamByteInFile = 0;
    else
        nextParamByteInFile = static_cast<int>(file.tellg())
                + offsetNext - ezc3d::DATA_TYPE::WORD;

    // -1 sizeof(char), 1 byte, 2 int, 4 float
    int lengthInByte(c3d.readInt(params.processorType(), file,
                                 1*ezc3d::DATA_TYPE::BYTE));
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
    int nDimensions(c3d.readInt(params.processorType(), file,
                                1*ezc3d::DATA_TYPE::BYTE));
    if (nDimensions == 0 && _data_type != DATA_TYPE::CHAR)
        // In the special case of a scalar
        _dimension.push_back(1);
    else // otherwise it's a matrix
        for (int i=0; i<nDimensions; ++i)
            // Read the dimension size of the matrix
            _dimension.push_back (c3d.readUint(params.processorType(), file,
                                               1*ezc3d::DATA_TYPE::BYTE));

    // Read the data for the parameters
    if (_data_type == DATA_TYPE::CHAR){
        c3d.readParam(file, _dimension, _param_data_string);
        // Readjust dimension in case the original c3d put
        // trailling \0
        for (unsigned int i=0; i<_dimension.size(); ++i){
            _dimension[i];
        }

    }
    else if (_data_type == DATA_TYPE::BYTE)
        c3d.readParam(params.processorType(), file,
                      static_cast<unsigned int>(_data_type), _dimension,
                      _param_data_int);
    else if (_data_type == DATA_TYPE::INT)
        c3d.readParam(params.processorType(), file,
                      static_cast<unsigned int>(_data_type), _dimension,
                      _param_data_int);
    else if (_data_type == DATA_TYPE::FLOAT)
        c3d.readParam(params.processorType(), file, _dimension,
                      _param_data_double);


    // Byte 5+nbCharInName ==> Number of characters in group description
    int nbCharInDesc(c3d.readInt(params.processorType(), file,
                                 1*ezc3d::DATA_TYPE::BYTE));
    // Byte 6+nbCharInName ==> Group description
    if (nbCharInDesc)
        _description = c3d.readString(
                    file, static_cast<unsigned int>(nbCharInDesc));
    setEmptyFlag();

    // Return how many bytes
    return nextParamByteInFile;
}

const std::string& ezc3d::ParametersNS::GroupNS::Parameter::name() const {
    return _name;
}

void ezc3d::ParametersNS::GroupNS::Parameter::name(
        const std::string &paramName) {
    _name = paramName;
}

const std::string&
ezc3d::ParametersNS::GroupNS::Parameter::description() const {
    return _description;
}

void ezc3d::ParametersNS::GroupNS::Parameter::description(
        const std::string &description) {
    _description = description;
}

bool ezc3d::ParametersNS::GroupNS::Parameter::isLocked() const {
    return _isLocked;
}

void ezc3d::ParametersNS::GroupNS::Parameter::lock() {
    _isLocked = true;
}

void ezc3d::ParametersNS::GroupNS::Parameter::unlock() {
    _isLocked = false;
}

size_t ezc3d::ParametersNS::GroupNS::Parameter::longestElement() const{
    if (_data_type != DATA_TYPE::CHAR)
        throw std::invalid_argument(
                "longestElement only make sense for CHAR data");
    if (_dimension.size() == 1)
        return _param_data_string[0].size();
    else {
        size_t longestSoFar(0);
        for (size_t i = 0; i<_param_data_string.size(); ++i){
            if (_param_data_string[i].size() > longestSoFar)
                longestSoFar = _param_data_string[i].size();
        }
        return longestSoFar;
    }
}

const std::vector<size_t>&
ezc3d::ParametersNS::GroupNS::Parameter::dimension() const {
    return _dimension;
}

bool ezc3d::ParametersNS::GroupNS::Parameter::isDimensionConsistent(
        size_t dataSize,
        const std::vector<size_t> &dimension) const {
    if (dataSize == 0){
        int dim(1);
        for (unsigned int i=0; i<dimension.size(); ++i)
            dim *= dimension[i];
        if (dimension.size() == 0 || dim == 0)
            return true;
        else
            return false;
    }

    size_t dimesionSize(1);
    for (unsigned int i=0; i<dimension.size(); ++i)
        dimesionSize *= dimension[i];
    if (dataSize == dimesionSize)
        return true;
    else
        return false;
}

void ezc3d::ParametersNS::GroupNS::Parameter::setEmptyFlag() {
    if (_dimension.size() ==0 || (_dimension.size() == 1 && _dimension[0] == 0))
        _isEmpty = true;
    else
        _isEmpty = false;
}

ezc3d::DATA_TYPE ezc3d::ParametersNS::GroupNS::Parameter::type() const {
    return _data_type;
}

void ezc3d::ParametersNS::GroupNS::Parameter::set(int data) {
    set(std::vector<int>()={data});
}

void ezc3d::ParametersNS::GroupNS::Parameter::set(size_t data) {
    set(static_cast<int>(data));
}

void ezc3d::ParametersNS::GroupNS::Parameter::set(
        const std::vector<int> &data,
        const std::vector<size_t> &dimension) {
    std::vector<size_t> dimensionCopy;
    if (dimension.size() == 0){
        dimensionCopy.push_back(data.size());
    } else {
        dimensionCopy = dimension;
    }
    if (!isDimensionConsistent(data.size(), dimensionCopy))
        throw std::range_error(
                "Dimension of the data does not correspond to sent dimensions");
    _data_type = ezc3d::DATA_TYPE::INT;
    _param_data_int = data;
    _dimension = dimensionCopy;
    setEmptyFlag();
}

void ezc3d::ParametersNS::GroupNS::Parameter::set(float data) {
    set(std::vector<double>()={static_cast<double>(data)});
}

void ezc3d::ParametersNS::GroupNS::Parameter::set(double data) {
    set(std::vector<double>()={data});
}

void ezc3d::ParametersNS::GroupNS::Parameter::set(
        const std::vector<double> &data,
        const std::vector<size_t> &dimension) {
    std::vector<size_t> dimensionCopy;
    if (dimension.size() == 0){
        dimensionCopy.push_back(data.size());
    } else {
        dimensionCopy = dimension;
    }
    if (!isDimensionConsistent(data.size(), dimensionCopy))
        throw std::range_error(
                "Dimension of the data does not correspond to sent dimensions");

    _data_type = ezc3d::DATA_TYPE::FLOAT;
    _param_data_double = data;
    _dimension = dimensionCopy;
    setEmptyFlag();
}

void ezc3d::ParametersNS::GroupNS::Parameter::set(
        const std::string& data) {
    set(std::vector<std::string>() = {data});
}

void ezc3d::ParametersNS::GroupNS::Parameter::set(
        const std::vector<std::string> &data,
        const std::vector<size_t> &dimension) {
    std::vector<size_t> dimensionCopy;
    if (dimension.size() == 0){
        dimensionCopy.push_back(data.size());
    } else {
        dimensionCopy = dimension;
    }
    if (!isDimensionConsistent(data.size(), dimensionCopy))
        throw std::range_error(
                "Dimension of the data does not correspond to sent dimensions");
    // Insert the length of the longest string
    size_t first_dim(0);
    for (unsigned int i=0; i<data.size(); ++i)
        if (data[i].size() > first_dim)
            first_dim = data[i].size();
    std::vector<size_t> dimensionWithStrLen = dimensionCopy;
    dimensionWithStrLen.insert(dimensionWithStrLen.begin(), first_dim);

    _data_type = ezc3d::DATA_TYPE::CHAR;
    _param_data_string = data;
    _dimension = dimensionWithStrLen;
    setEmptyFlag();
}

const std::vector<int>&
ezc3d::ParametersNS::GroupNS::Parameter::valuesAsByte() const {
    if (!_isEmpty && _data_type != DATA_TYPE::BYTE)
        throw std::invalid_argument(_name + " parameter is not a BYTE");
    return _param_data_int;
}

const std::vector<int>&
ezc3d::ParametersNS::GroupNS::Parameter::valuesAsInt() const {
    if (!_isEmpty && _data_type != DATA_TYPE::INT)
        throw std::invalid_argument(_name + " parameter is not an INT");
    return _param_data_int;
}

const std::vector<double>&
ezc3d::ParametersNS::GroupNS::Parameter::valuesAsDouble() const {
    if (!_isEmpty && _data_type != DATA_TYPE::FLOAT)
        throw std::invalid_argument(_name + " parameter is not a FLOAT");
    return _param_data_double;
}

const std::vector<std::string>&
ezc3d::ParametersNS::GroupNS::Parameter::valuesAsString() const {
    if (!_isEmpty && _data_type != DATA_TYPE::CHAR)
        throw std::invalid_argument(_name + " parameter is not string");

    return _param_data_string;
}


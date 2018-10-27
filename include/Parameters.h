#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <stdexcept>
#include <memory>

#include <ezc3d.h>

class EZC3D_API ezc3d::ParametersNS::Parameters{
public:
    Parameters();
    Parameters(ezc3d::c3d &file);
    void print() const;
    void write(std::fstream &f) const;

    const std::vector<ezc3d::ParametersNS::GroupNS::Group>& groups() const;
    const ezc3d::ParametersNS::GroupNS::Group& group(int group) const;
    const ezc3d::ParametersNS::GroupNS::Group& group(const std::string& groupName) const;
    int groupIdx(const std::string& groupName) const;
    ezc3d::ParametersNS::GroupNS::Group& group_nonConst(int group);
    void addGroup(const ezc3d::ParametersNS::GroupNS::Group& g);

    int parametersStart() const;
    int checksum() const;
    int nbParamBlock() const;
    int processorType() const;

protected:
    std::vector<ezc3d::ParametersNS::GroupNS::Group> _groups; // Holder for the group of parameters

    // Read the Parameters Header
    int _parametersStart;   // Byte 1 ==> if 1 then it starts at byte 3 otherwise at byte 512*parametersStart
    int _checksum;         // Byte 2 ==> should be 80 if it is a c3d
    int _nbParamBlock;      // Byte 3 ==> Number of parameter blocks to follow
    int _processorType;     // Byte 4 ==> Processor type (83 + [1 Inter, 2 DEC, 3 MIPS])
};


class EZC3D_API ezc3d::ParametersNS::GroupNS::Group{
public:
    Group(const std::string &name = "", const std::string &description = "");

    int read(ezc3d::c3d &file, int nbCharInName);
    int addParameter(ezc3d::c3d &file, int nbCharInName);
    void addParameter(const ezc3d::ParametersNS::GroupNS::Parameter& p);
    void print() const;
    void write(std::fstream &f, int groupIdx, std::streampos &dataStartPosition) const;

    // Getter for the group
    void lock();
    void unlock();
    bool isLocked() const;
    const std::string& name() const;
    const std::string& description() const;
    const std::vector<ezc3d::ParametersNS::GroupNS::Parameter>& parameters() const;
    const ezc3d::ParametersNS::GroupNS::Parameter& parameter(int idx) const;
    std::vector<ezc3d::ParametersNS::GroupNS::Parameter>& parameters_nonConst();
    int parameterIdx(std::string parameterName) const;
    const ezc3d::ParametersNS::GroupNS::Parameter& parameter(std::string parameterName) const;

protected:
    bool _isLocked; // If the group should not be modified

    std::string _name;
    std::string _description;

    std::vector<ezc3d::ParametersNS::GroupNS::Parameter> _parameters; // Holder for the parameters of the group
};


class EZC3D_API ezc3d::ParametersNS::GroupNS::Parameter{
public:
    Parameter(const std::string &name = "", const std::string &description = "");

    int read(ezc3d::c3d &file, int nbCharInName);
    void set(const std::vector<int>& data, const std::vector<int>& dimension);
    void set(const std::vector<float>& data, const std::vector<int>& dimension);
    void set(const std::vector<std::string>& data, const std::vector<int>& dimension);
    void print() const;
    void write(std::fstream &f, int groupIdx, std::streampos &dataStartPosition) const;

    // Getter for the group
    const std::vector<int> dimension() const;
    void lock();
    void unlock();
    bool isLocked() const;
    void name(const std::string paramName);
    const std::string& name() const;
    const std::string& description() const;

    ezc3d::DATA_TYPE type() const;
    const std::vector<std::string>& valuesAsString() const;
    const std::vector<int>& valuesAsByte() const;
    const std::vector<int>& valuesAsInt() const;
    const std::vector<float>& valuesAsFloat() const;

protected:

    bool _isLocked; // If the group should not be modified
    unsigned int writeImbricatedParameter(std::fstream &f, const std::vector<int>& dim, unsigned int currentIdx=0, unsigned int cmp=0) const;
    bool isDimensionConsistent(int dataSize, const std::vector<int>& dimension) const;

    std::vector<int> _dimension; // Mapping of the data vector
    ezc3d::DATA_TYPE _data_type; // What kind of data there is in the parameter
    std::vector<int> _param_data_int; // Actual parameter
    std::vector<float> _param_data_float; // Actual parameter
    std::vector<std::string> _param_data_string; // Actual parameter

    std::string _name;
    std::string _description;
};



#endif

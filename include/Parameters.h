#ifndef __PARAMETERS_H__
#define __PARAMETERS_H__

#include "ezC3D.h"
#include <stdexcept>

class ezC3D::Parameters{
public:
    Parameters(ezC3D::Reader &file);
    void read(ezC3D::Reader &file);
    void print() const;

    class Group;
    const std::vector<Group>& groups() const;
    const Group& group(int group) const;
    const Group& group(const std::string& groupName) const;
    Group& group_nonConst(int group);

    int parametersStart() const;
    int checksum() const;
    int nbParamBlock() const;
    int processorType() const;
    //class Group::Parameter;

protected:
    std::vector<Group> _groups; // Holder for the group of parameters

    // Read the Parameters Header
    int _parametersStart;   // Byte 1 ==> if 1 then it starts at byte 3 otherwise at byte 512*parametersStart
    int _checksum;         // Byte 2 ==> should be 80 if it is a c3d
    int _nbParamBlock;      // Byte 3 ==> Number of parameter blocks to follow
    int _processorType;     // Byte 4 ==> Processor type (83 + [1 Inter, 2 DEC, 3 MIPS])
};


class ezC3D::Parameters::Group{
public:
    Group();

    int read(ezC3D::Reader &file, int nbCharInName);
    int addParameter(ezC3D::Reader &file, int nbCharInName);
    void print() const;

    class Parameter;
    // Getter for the group
    void lock();
    void unlock();
    bool isLocked() const;
    const std::string& name() const;
    const std::string& description() const;
    const std::vector<Parameter>& parameters() const;
    const Parameter& parameter(int idx) const;
    std::vector<Parameter>& parameters_nonConst();
    const Parameter& parameter(std::string parameterName) const;

protected:
    bool _isLocked; // If the group should not be modified

    std::string _name;
    std::string _description;

    std::vector<Parameter> _parameters; // Holder for the parameters of the group
};
class ezC3D::Parameters::Group::Parameter{
public:
    Parameter();

    int read(ezC3D::Reader &file, int nbCharInName);
    void print() const;

    // Getter for the group
    void lock();
    void unlock();
    bool isLocked() const;
    const std::string& name() const;
    const std::string& description() const;

    const std::vector<std::string>& stringValues() const;
protected:
    enum DATA_TYPE{
        CHAR = -1,
        BYTE = 1,
        INT = 2,
        FLOAT = 4
    };

    bool _isLocked; // If the group should not be modified

    std::vector<int> _dimension; // Mapping of the data vector
    DATA_TYPE _data_type; // What kind of data there is in the parameter
    std::vector<int> _param_data_int; // Actual parameter
    std::vector<float> _param_data_float; // Actual parameter
    std::vector<std::string> _param_data_string; // Actual parameter

    std::string _name;
    std::string _description;
};



#endif

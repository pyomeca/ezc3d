#ifndef __PARAMETERS_H__
#define __PARAMETERS_H__

#include "ezC3D.h"
#include <stdexcept>

class ezC3D::Parameters{
public:
    Parameters(ezC3D &file);
    void read(ezC3D &file);
    void print() const;

protected:
    class Group;

public:
    const std::vector<ezC3D::Parameters::Group>& groups() const;
    const ezC3D::Parameters::Group& group(int group) const;
    ezC3D::Parameters::Group& group_nonConst(int group);

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

    int read(ezC3D &file, int nbCharInName);
    int addParameter(ezC3D &file, int nbCharInName);
    void print() const;

protected:
    class Parameter;

public:
    // Getter for the group
    void lock();
    void unlock();
    bool isLocked() const;
    const std::string& name() const;
    const std::string& description() const;
    const std::vector<Parameter>& parameters() const;
    const Parameter& parameter(int idx) const;
    std::vector<Parameter>& parameters_nonConst();

protected:
    bool _isLocked; // If the group should not be modified

    std::string _name;
    std::string _description;

    class Parameter;
    std::vector<Parameter> _parameters; // Holder for the parameters of the group
};
class ezC3D::Parameters::Group::Parameter{
public:
    Parameter();

    int read(ezC3D &file, int nbCharInName);
    void print() const;

    // Getter for the group
    void lock();
    void unlock();
    bool isLocked() const;
    const std::string& name() const;
    const std::string& description() const;

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

#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <Group.h>

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

#endif

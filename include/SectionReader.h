#ifndef __SECTION_READER_H__
#define __SECTION_READER_H__

#include "ezC3D.h"
#include <stdexcept>

class ezC3D::Header{
public:
    Header(ezC3D &file);
    void read(ezC3D &file);
    void print() const;

    // Getter on the parameters
    int parametersAddress() const;
    int checksum() const;
    int nb3dPoints() const;
    int nbAnalogsMeasurement() const;
    int firstFrame() const;
    int lastFrame() const;
    int nbMaxInterpGap() const;
    int scaleFactor() const;
    int dataStartAnalog() const;
    int nbAnalogByFrame() const;
    double frameRate() const;
    int emptyBlock1() const;
    int keyLabelPresent() const;
    int firstBlockKeyLabel() const;
    int fourCharPresent() const;
    int nbEvents() const;
    int emptyBlock2() const;
    const std::vector<float>& eventsTime() const;
    float eventsTime(int i) const;
    int eventsDisplay() const;
    int emptyBlock3() const;
    const std::string& eventsLabel() const;
    int emptyBlock4() const;
    int nbFrames() const;
    int nbAnalogs() const;

protected:
    // Read the Header
    int _parametersAddress;         // Byte 1.1
    int _checksum;                 // Byte 1.2 ==> 80 if it is a C3D
    int _nb3dPoints;                // Byte 2 ==> number of stored trajectories
    int _nbAnalogsMeasurement;      // Byte 3 ==> number of analog data
    int _firstFrame; // 1-based!    // Byte 4 ==> first frame in the file
    int _lastFrame;                 // Byte 5 ==> last frame in the file
    int _nbMaxInterpGap;            // Byte 6 ==> maximal gap for interpolation
    int _scaleFactor;               // Byte 7-8 ==> convert int to 3d reference frame, floating point if negative
    int _dataStartAnalog;           // Byte 9 ==> Number of first block for 3D and analog data
    int _nbAnalogByFrame;           // Byte 10 ==> Number of analog by frame
    double _frameRate;              // Byte 11-12 ==> 3d frame rate in Hz (floating point)
    int _emptyBlock1;               // Byte 13-147
    int _keyLabelPresent;           // Byte 148 ==> 12345 if Label and range are present
    int _firstBlockKeyLabel;        // Byte 149 ==> First block of key labels (if present)
    int _fourCharPresent;           // Byte 150 ==> 12345 if 4 char event labels are supported (otherwise 2 char)
    int _nbEvents;                  // Byte 151 ==> Number of defined time events (0 to 18)
    int _emptyBlock2;               // Byte 152
    std::vector<float> _eventsTime; // Byte 153-188 ==> Event times (floating-point) in seconds
    int _eventsDisplay;             // Byte 189-197 ==> Event display (0x00 = ON, 0x01 = OFF)
    int _emptyBlock3;               // Byte 198
    std::string _eventsLabel;       // Byte 199-234 ==> Event labels (4 char by label)
    int _emptyBlock4;               // Byte 235-256
};


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

#include "SectionReader.h"

ezC3D::Header::Header(ezC3D &file)
{
    read(file);
}


int ezC3D::Header::nbFrames() const
{
    return _lastFrame - _firstFrame;
}

int ezC3D::Header::nbAnalogs() const
{
    return _nbAnalogsMeasurement / _nbAnalogByFrame;
}

int ezC3D::Header::emptyBlock4() const
{
    return _emptyBlock4;
}

const std::string& ezC3D::Header::eventsLabel() const
{
    return _eventsLabel;
}

int ezC3D::Header::emptyBlock3() const
{
    return _emptyBlock3;
}

int ezC3D::Header::eventsDisplay() const
{
    return _eventsDisplay;
}

float ezC3D::Header::eventsTime(int idx) const
{
    return _eventsTime[idx];
}

int ezC3D::Header::emptyBlock2() const
{
    return _emptyBlock2;
}

int ezC3D::Header::nbEvents() const
{
    return _nbEvents;
}

int ezC3D::Header::fourCharPresent() const
{
    return _fourCharPresent;
}

int ezC3D::Header::firstBlockKeyLabel() const
{
    return _firstBlockKeyLabel;
}

int ezC3D::Header::keyLabelPresent() const
{
    return _keyLabelPresent;
}

int ezC3D::Header::emptyBlock1() const
{
    return _emptyBlock1;
}

double ezC3D::Header::frameRate() const
{
    return _frameRate;
}

int ezC3D::Header::nbAnalogByFrame() const
{
    return _nbAnalogByFrame;
}

int ezC3D::Header::dataStartAnalog() const
{
    return _dataStartAnalog;
}

int ezC3D::Header::scaleFactor() const
{
    return _scaleFactor;
}

int ezC3D::Header::nbMaxInterpGap() const
{
    return _nbMaxInterpGap;
}

int ezC3D::Header::lastFrame() const
{
    return _lastFrame;
}

int ezC3D::Header::firstFrame() const
{
    return _firstFrame;
}

int ezC3D::Header::nbAnalogsMeasurement() const
{
    return _nbAnalogsMeasurement;
}

int ezC3D::Header::nb3dPoints() const
{
    return _nb3dPoints;
}

int ezC3D::Header::iChecksum() const
{
    return _iChecksum;
}

int ezC3D::Header::parametersAddress() const
{
    return _parametersAddress;
}

// Read the Header
void ezC3D::Header::read(ezC3D &file)
{
    // Parameter address
    _parametersAddress = file.readInt(1*ezC3D::READ_SIZE::BYTE, 0, std::ios::beg);
    _iChecksum = file.readInt(1*ezC3D::READ_SIZE::BYTE);
    if (_iChecksum != 80) // If checkbyte is wrong
        throw std::ios_base::failure("File must be a valid c3d file");

    // Number of data
    _nb3dPoints = file.readInt(1*ezC3D::READ_SIZE::WORD);
    _nbAnalogsMeasurement = file.readInt(1*ezC3D::READ_SIZE::WORD);

    // Idx of first and last frame
    _firstFrame = file.readInt(1*ezC3D::READ_SIZE::WORD) - 1; // 1-based!
    _lastFrame = file.readInt(1*ezC3D::READ_SIZE::WORD);

    // Some info
    _nbMaxInterpGap = file.readInt(1*ezC3D::READ_SIZE::WORD);
    _scaleFactor = file.readInt(2*ezC3D::READ_SIZE::WORD);

    // Parameters of analog data
    _dataStartAnalog = file.readInt(1*ezC3D::READ_SIZE::WORD);
    _nbAnalogByFrame = file.readInt(1*ezC3D::READ_SIZE::WORD);
    _frameRate = file.readFloat();
    _emptyBlock1 = file.readInt(135*ezC3D::READ_SIZE::WORD);

    // Parameters of keys
    _keyLabelPresent = file.readInt(1*ezC3D::READ_SIZE::WORD);
    _firstBlockKeyLabel = file.readInt(1*ezC3D::READ_SIZE::WORD);
    _fourCharPresent = file.readInt(1*ezC3D::READ_SIZE::WORD);

    // Parameters of events
    _nbEvents = file.readInt(1*ezC3D::READ_SIZE::WORD);
    _emptyBlock2 = file.readInt(1*ezC3D::READ_SIZE::WORD);
    for (int i = 0; i < 18; ++i)
        _eventsTime.push_back(file.readFloat());
    _eventsDisplay = file.readInt(9*ezC3D::READ_SIZE::WORD);
    _emptyBlock3 = file.readInt(1*ezC3D::READ_SIZE::WORD);
    _eventsLabel = file.readString(36*ezC3D::READ_SIZE::WORD);
    _emptyBlock4 = file.readInt(22*ezC3D::READ_SIZE::WORD);
}

void ezC3D::Header::print() const{
    std::cout << "HEADER" << std::endl;
    std::cout << "nb3dPoints = " << _nb3dPoints << std::endl;
    std::cout << "nbAnalogsMeasurement = " << _nbAnalogsMeasurement << std::endl;
    std::cout << "nbAnalogs = " << nbAnalogs() << std::endl;
    std::cout << "firstFrame = " << _firstFrame << std::endl;
    std::cout << "lastFrame = " << _lastFrame << std::endl;
    std::cout << "lastFrame = " << nbFrames() << std::endl;
    std::cout << "nbMaxInterpGap = " << _nbMaxInterpGap << std::endl;
    std::cout << "scaleFactor = " << _scaleFactor << std::endl;
    std::cout << "dataStartAnalog = " << _dataStartAnalog << std::endl;
    std::cout << "nbAnalogByFrame = " << _nbAnalogByFrame << std::endl;
    std::cout << "frameRate = " << _frameRate << std::endl;
    std::cout << "emptyBlock1 = " << _emptyBlock1 << std::endl;
    std::cout << "keyLabelPresent = " << _keyLabelPresent << std::endl;
    std::cout << "firstBlockKeyLabel = " << _firstBlockKeyLabel << std::endl;
    std::cout << "fourCharPresent = " << _fourCharPresent << std::endl;
    std::cout << "nbEvents = " << _nbEvents << std::endl;
    std::cout << "emptyBlock2 = " << _emptyBlock2 << std::endl;
    for (int i=0; i<_eventsTime.size(); ++i)
        std::cout << "eventsTime[" << i << "] = " << _eventsTime[i] << std::endl;
    std::cout << "eventsDisplay = " << _eventsDisplay << std::endl;
    std::cout << "emptyBlock3 = " << _emptyBlock3 << std::endl;
    std::cout << "eventsLabel = " << _eventsLabel << std::endl;
    std::cout << "emptyBlock4 = " << _emptyBlock4 << std::endl;
    std::cout << std::endl;
}



ezC3D::Parameter::Parameter(ezC3D &file) :
    _parametersStart(0),
    _iChecksum(0),
    _nbParamBlock(0),
    _processorType(0)
{
    // Read the Parameters Header
    _parametersStart = file.readInt(1*ezC3D::READ_SIZE::BYTE, 256*ezC3D::READ_SIZE::WORD*(file.header()->parametersAddress()-1), std::ios::beg);
    _iChecksum = file.readInt(1*ezC3D::READ_SIZE::BYTE);
    _nbParamBlock = file.readInt(1*ezC3D::READ_SIZE::BYTE);
    _processorType = file.readInt(1*ezC3D::READ_SIZE::BYTE);

    // Read parameter or group
    int nextParamByteInFile((int)file.tellg() + _parametersStart - ezC3D::READ_SIZE::BYTE);
    while (nextParamByteInFile)
    {
        // Check if we spontaneously got to the next parameter. Otherwise c3d is messed up
        if (file.tellg() != nextParamByteInFile)
            throw std::ios_base::failure("Bad c3d formatting");

        // Nb of char in the group name, locked if negative, 0 if we finished the section
        int nbCharInName(file.readInt(1*ezC3D::READ_SIZE::BYTE));
        if (nbCharInName == 0)
            break;
        int id(file.readInt(1*ezC3D::READ_SIZE::BYTE));

        // Make sure there at least enough group
        for (int i = _groups.size(); i < abs(id); ++i)
            _groups.push_back(ezC3D::Parameter::Group());

        // Group ID always negative for groups and positive parameter of group ID
        if (id < 0)
            nextParamByteInFile = group_nonConst(abs(id)).readGroup(file, nbCharInName);
        //else
            nextParamByteInFile = _groups[id].addParameter(nbCharInName);



    }
}
int ezC3D::Parameter::processorType() const
{
    return _processorType;
}
int ezC3D::Parameter::nbParamBlock() const
{
    return _nbParamBlock;
}
int ezC3D::Parameter::iChecksum() const
{
    return _iChecksum;
}
int ezC3D::Parameter::parametersStart() const
{
    return _parametersStart;
}
void ezC3D::Parameter::print() const
{
    std::cout << "Parameters header" << std::endl;
    std::cout << "_parametersStart = " << _parametersStart << std::endl;
    std::cout << "_iChecksum = " << _iChecksum << std::endl;
    std::cout << "nbParamBlock = " << _nbParamBlock << std::endl;
    std::cout << "processorType = " << _processorType << std::endl;

    for (int i = 0; i < _groups.size(); ++i){
        std::cout << "Group " << i << std::endl;
    }
    std::cout << std::endl;
}




ezC3D::Parameter::Group::Group() :
    _isLocked(false),
    _name(""),
    _description("")
{

}
const std::vector<ezC3D::Parameter::Group>& ezC3D::Parameter::groups() const
{
    return _groups;
}
ezC3D::Parameter::Group &ezC3D::Parameter::group_nonConst(int group){
    return _groups[group];
}
const ezC3D::Parameter::Group &ezC3D::Parameter::group(int group) const
{
    return _groups[group];
}
void ezC3D::Parameter::Group::lock()
{
    _isLocked = true;
}
void ezC3D::Parameter::Group::unlock()
{
    _isLocked = false;
}
int ezC3D::Parameter::Group::readGroup(ezC3D &file, int nbCharInName)
{
    if (nbCharInName < 0)
        _isLocked = true;
    else
        _isLocked = false;

    // Read name of the group
    _name = file.readString(abs(nbCharInName) * ezC3D::READ_SIZE::BYTE);

    // number of byte to the next group from here
    int offsetNext((int)file.readUint(2*ezC3D::READ_SIZE::BYTE));
    // Compute the position of the element in the file
    int nextParamByteInFile;
    if (offsetNext == 0)
        nextParamByteInFile = 0;
    else
        nextParamByteInFile = (int)file.tellg() + offsetNext - ezC3D::READ_SIZE::WORD;

    // Byte 5+nbCharInName ==> Number of characters in group description
    int nbCharInDesc(file.readInt(1*ezC3D::READ_SIZE::BYTE));
    // Byte 6+nbCharInName ==> Group description
    if (nbCharInDesc)
        _description = file.readString(nbCharInDesc);

    // Return how many bytes
    return nextParamByteInFile;
}
void ezC3D::Parameter::Group::print() const
{
    std::cout << "isLocked = " << _isLocked << std::endl;
    std::cout << "groupName = " << _name << std::endl;
    std::cout << "desc = " << _description << std::endl;
}








//        // -1 sizeof(char), 1 byte, 2 int, 4 float
//        int lengthInByte        (file.readInt(1*ezC3D::READ_SIZE::BYTE));

//        // number of dimension of parameter (0 for scalar)
//        int nDimensions         (file.readInt(1*ezC3D::READ_SIZE::BYTE));
//        std::vector<int> dimension;
//        if (nDimensions == 0) // In the special case of a scalar
//            dimension.push_back(1);
//        else // otherwise it's a matrix
//            for (int i=0; i<nDimensions; ++i)
//                dimension.push_back (file.readInt(1*ezC3D::READ_SIZE::BYTE));    // Read the dimension size of the matrix

//        // Read the data for the parameters
//        std::vector<int> param_data_int;
//        std::vector<std::string> param_data_string;
//        if (lengthInByte > 0)
//            file.readMatrix(lengthInByte, dimension, param_data_int);
//        else {
//            std::vector<std::string> param_data_string_tp;
//            file.readMatrix(dimension, param_data_string_tp);
//            // Vicon c3d organize its text in column-wise format, I am not sure if
//            // this is a standard or a custom made stuff
//            if (dimension.size() == 1){
//                std::string tp;
//                for (int i = 0; i < dimension[0]; ++i)
//                    tp += param_data_string_tp[i];
//                param_data_string.push_back(tp);
//            }
//            else if (dimension.size() == 2){
//                int idx(0);
//                for (int i = 0; i < dimension[1]; ++i){
//                    std::string tp;
//                    for (int j = 0; j < dimension[0]; ++j){
//                        tp += param_data_string_tp[idx];
//                        ++idx;
//                    }
//                    param_data_string.push_back(tp);
//                }
//            }
//            else
//                throw std::ios_base::failure("Parsing char on matrix other than 2d or 1d matrix is not implemented yet");
//
//
//        std::cout << "lengthInByte = " << lengthInByte << std::endl;
//        std::cout << "nDimensions = " << nDimensions << std::endl;
//        for (int i = 0; i< dimension.size(); ++i)
//            std::cout << "dimension[" << i << "] = " << dimension[i] << std::endl;
//        for (int i = 0; i< param_data_int.size(); ++i)
//            std::cout << "param_data_int[" << i << "] = " << param_data_int[i] << std::endl;
//        for (int i = 0; i< param_data_string.size(); ++i)
//            std::cout << "param_data_string[" << i << "] = " << param_data_string[i] << std::endl;
//    }

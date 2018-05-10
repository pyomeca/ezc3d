#include "Header.h"

ezC3D::Header::Header(ezC3D::C3D &file)
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
const std::vector<float>& ezC3D::Header::eventsTime() const
{
    return _eventsTime;
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
int ezC3D::Header::checksum() const
{
    return _checksum;
}
int ezC3D::Header::parametersAddress() const
{
    return _parametersAddress;
}
// Read the Header
void ezC3D::Header::read(ezC3D::C3D &file)
{
    // Parameter address
    _parametersAddress = file.readInt(1*ezC3D::READ_SIZE::BYTE, 0, std::ios::beg);
    _checksum = file.readInt(1*ezC3D::READ_SIZE::BYTE);
    if (_checksum != 80) // If checkbyte is wrong
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
    std::cout << "nb3dPoints = " << nb3dPoints() << std::endl;
    std::cout << "nbAnalogsMeasurement = " << nbAnalogsMeasurement() << std::endl;
    std::cout << "nbAnalogs = " << nbAnalogs() << std::endl;
    std::cout << "firstFrame = " << firstFrame() << std::endl;
    std::cout << "lastFrame = " << lastFrame() << std::endl;
    std::cout << "lastFrame = " << nbFrames() << std::endl;
    std::cout << "nbMaxInterpGap = " << nbMaxInterpGap() << std::endl;
    std::cout << "scaleFactor = " << scaleFactor() << std::endl;
    std::cout << "dataStartAnalog = " << dataStartAnalog() << std::endl;
    std::cout << "nbAnalogByFrame = " << nbAnalogByFrame() << std::endl;
    std::cout << "frameRate = " << frameRate() << std::endl;
    std::cout << "keyLabelPresent = " << keyLabelPresent() << std::endl;
    std::cout << "firstBlockKeyLabel = " << firstBlockKeyLabel() << std::endl;
    std::cout << "fourCharPresent = " << fourCharPresent() << std::endl;
    std::cout << "nbEvents = " << nbEvents() << std::endl;
    for (int i=0; i<eventsTime().size(); ++i)
        std::cout << "eventsTime[" << i << "] = " << eventsTime(i) << std::endl;
    std::cout << "eventsDisplay = " << eventsDisplay() << std::endl;
    std::cout << "eventsLabel = " << eventsLabel() << std::endl;
    std::cout << std::endl;
}


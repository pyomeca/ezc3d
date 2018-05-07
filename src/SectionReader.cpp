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

#define EZC3D_API_EXPORTS
#include "Header.h"

ezc3d::Header::Header():
    _parametersAddress(2),
    _checksum(0x50),
    _nb3dPoints(0),
    _nbAnalogsMeasurement(0),
    _firstFrame(0),
    _lastFrame(0),
    _nbMaxInterpGap(10),
    _scaleFactor(-1),
    _dataStart(1),
    _nbAnalogByFrame(0),
    _frameRate(0),
    _emptyBlock1(0),
    _keyLabelPresent(0),
    _firstBlockKeyLabel(0),
    _fourCharPresent(0x3039),
    _nbEvents(0),
    _emptyBlock2(0),
    _emptyBlock3(0),
    _emptyBlock4(0)
{
    _eventsTime.resize(18);
    _eventsDisplay.resize(9);
    _eventsLabel.resize(18);
}

ezc3d::Header::Header(ezc3d::c3d &file) :
    _parametersAddress(2),
    _checksum(0x50),
    _nb3dPoints(0),
    _nbAnalogsMeasurement(0),
    _firstFrame(0),
    _lastFrame(0),
    _nbMaxInterpGap(10),
    _scaleFactor(-1),
    _dataStart(1),
    _nbAnalogByFrame(0),
    _frameRate(0),
    _emptyBlock1(0),
    _keyLabelPresent(0),
    _firstBlockKeyLabel(0),
    _fourCharPresent(0x3039),
    _nbEvents(0),
    _emptyBlock2(0),
    _emptyBlock3(0),
    _emptyBlock4(0)
{
    _eventsTime.resize(18);
    _eventsDisplay.resize(9);
    _eventsLabel.resize(18);
    read(file);
}



int ezc3d::Header::nbFrames() const
{
    if (nb3dPoints() == 0 && nbAnalogs() == 0)
        return 0;
    else
        return _lastFrame - _firstFrame + 1;
}

void ezc3d::Header::nbAnalogs(int n)
{
    _nbAnalogsMeasurement = n * _nbAnalogByFrame;
}
int ezc3d::Header::nbAnalogs() const
{
    if (_nbAnalogByFrame == 0)
        return 0;
    else
        return _nbAnalogsMeasurement / _nbAnalogByFrame;
}
int ezc3d::Header::emptyBlock4() const
{
    return _emptyBlock4;
}
const std::vector<std::string>& ezc3d::Header::eventsLabel() const
{
    return _eventsLabel;
}
const std::string& ezc3d::Header::eventsLabel(int idx) const
{
    if (idx < 0 || static_cast<size_t>(idx) >= _eventsLabel.size())
        throw std::invalid_argument("Event label not found");
    return _eventsLabel[static_cast<unsigned int>(idx)];
}
int ezc3d::Header::emptyBlock3() const
{
    return _emptyBlock3;
}
std::vector<int> ezc3d::Header::eventsDisplay() const
{
    return _eventsDisplay;
}
int ezc3d::Header::eventsDisplay(int idx) const{
    if (idx < 0 || static_cast<size_t>(idx) >= _eventsDisplay.size())
        throw std::invalid_argument("Event display not found");
    return _eventsDisplay[static_cast<unsigned int>(idx)];
}
const std::vector<float>& ezc3d::Header::eventsTime() const
{
    return _eventsTime;
}
float ezc3d::Header::eventsTime(int idx) const
{
    if (idx < 0 || static_cast<size_t>(idx) >= _eventsTime.size())
        throw std::invalid_argument("Event time not found");
    return _eventsTime[static_cast<unsigned int>(idx)];
}
int ezc3d::Header::emptyBlock2() const
{
    return _emptyBlock2;
}
int ezc3d::Header::nbEvents() const
{
    return _nbEvents;
}
int ezc3d::Header::fourCharPresent() const
{
    return _fourCharPresent;
}
int ezc3d::Header::firstBlockKeyLabel() const
{
    return _firstBlockKeyLabel;
}
int ezc3d::Header::keyLabelPresent() const
{
    return _keyLabelPresent;
}
int ezc3d::Header::emptyBlock1() const
{
    return _emptyBlock1;
}
void ezc3d::Header::frameRate(float f)
{
    _frameRate = f;
}
float ezc3d::Header::frameRate() const
{
    return _frameRate;
}
void ezc3d::Header::nbAnalogByFrame(int nb)
{
    int analogs(nbAnalogs());
    _nbAnalogByFrame = nb;
    nbAnalogs(analogs);
}
int ezc3d::Header::nbAnalogByFrame() const
{
    return _nbAnalogByFrame;
}
int ezc3d::Header::dataStart() const
{
    return _dataStart;
}
int ezc3d::Header::scaleFactor() const
{
    return _scaleFactor;
}
int ezc3d::Header::nbMaxInterpGap() const
{
    return _nbMaxInterpGap;
}
void ezc3d::Header::firstFrame(int frame)
{
    _firstFrame = frame;
}
int ezc3d::Header::firstFrame() const
{
    return _firstFrame;
}
void ezc3d::Header::lastFrame(int frame)
{
    _lastFrame = frame;
}
int ezc3d::Header::lastFrame() const
{
    return _lastFrame;
}
int ezc3d::Header::nbAnalogsMeasurement() const
{
    return _nbAnalogsMeasurement;
}
void ezc3d::Header::nb3dPoints(int n)
{
    _nb3dPoints = n;
}
int ezc3d::Header::nb3dPoints() const
{
    return _nb3dPoints;
}
int ezc3d::Header::checksum() const
{
    return _checksum;
}
int ezc3d::Header::parametersAddress() const
{
    return _parametersAddress;
}
// Read the Header
void ezc3d::Header::read(ezc3d::c3d &file)
{
    // Parameter address
    _parametersAddress = file.readInt(1*ezc3d::DATA_TYPE::BYTE, 0, std::ios::beg);
    _checksum = file.readInt(1*ezc3d::DATA_TYPE::BYTE);
    if (_checksum != 80) // If checkbyte is wrong
        throw std::ios_base::failure("File must be a valid c3d file");

    // Number of data
    _nb3dPoints = file.readInt(1*ezc3d::DATA_TYPE::WORD);
    _nbAnalogsMeasurement = file.readInt(1*ezc3d::DATA_TYPE::WORD);

    // Idx of first and last frame
    _firstFrame = file.readInt(1*ezc3d::DATA_TYPE::WORD) - 1; // 1-based!
    _lastFrame = file.readInt(1*ezc3d::DATA_TYPE::WORD) - 1;

    // Some info
    _nbMaxInterpGap = file.readInt(1*ezc3d::DATA_TYPE::WORD);
    _scaleFactor = file.readInt(2*ezc3d::DATA_TYPE::WORD);

    // Parameters of analog data
    _dataStart = file.readInt(1*ezc3d::DATA_TYPE::WORD);
    _nbAnalogByFrame = file.readInt(1*ezc3d::DATA_TYPE::WORD);
    _frameRate = file.readFloat();
    _emptyBlock1 = file.readInt(135*ezc3d::DATA_TYPE::WORD);

    // Parameters of keys
    _keyLabelPresent = file.readInt(1*ezc3d::DATA_TYPE::WORD);
    _firstBlockKeyLabel = file.readInt(1*ezc3d::DATA_TYPE::WORD);
    _fourCharPresent = file.readInt(1*ezc3d::DATA_TYPE::WORD);

    // Parameters of events
    _nbEvents = file.readInt(1*ezc3d::DATA_TYPE::WORD);
    _emptyBlock2 = file.readInt(1*ezc3d::DATA_TYPE::WORD);
    for (unsigned int i = 0; i < _eventsTime.size(); ++i)
        _eventsTime[i] = file.readFloat();
    for (unsigned int i = 0; i < _eventsDisplay.size(); ++i)
        _eventsDisplay[i] = file.readInt(1*ezc3d::DATA_TYPE::WORD);
    _emptyBlock3 = file.readInt(1*ezc3d::DATA_TYPE::WORD);
    for (unsigned int i = 0; i<_eventsLabel.size(); ++i)
        _eventsLabel[i] = file.readString(2*ezc3d::DATA_TYPE::WORD);
    _emptyBlock4 = file.readInt(22*ezc3d::DATA_TYPE::WORD);
}
void ezc3d::Header::print() const{
    std::cout << "HEADER" << std::endl;
    std::cout << "nb3dPoints = " << nb3dPoints() << std::endl;
    std::cout << "nbAnalogsMeasurement = " << nbAnalogsMeasurement() << std::endl;
    std::cout << "nbAnalogs = " << nbAnalogs() << std::endl;
    std::cout << "firstFrame = " << firstFrame() << std::endl;
    std::cout << "lastFrame = " << lastFrame() << std::endl;
    std::cout << "nbFrames = " << nbFrames() << std::endl;
    std::cout << "nbMaxInterpGap = " << nbMaxInterpGap() << std::endl;
    std::cout << "scaleFactor = " << scaleFactor() << std::endl;
    std::cout << "dataStart = " << dataStart() << std::endl;
    std::cout << "nbAnalogByFrame = " << nbAnalogByFrame() << std::endl;
    std::cout << "frameRate = " << frameRate() << std::endl;
    std::cout << "keyLabelPresent = " << keyLabelPresent() << std::endl;
    std::cout << "firstBlockKeyLabel = " << firstBlockKeyLabel() << std::endl;
    std::cout << "fourCharPresent = " << fourCharPresent() << std::endl;
    std::cout << "nbEvents = " << nbEvents() << std::endl;
    for (int i=0; i<static_cast<int>(eventsTime().size()); ++i)
        std::cout << "eventsTime[" << i << "] = " << eventsTime(i) << std::endl;
    for (int i=0; i<static_cast<int>(eventsTime().size()); ++i)
        std::cout << "eventsDisplay[" << i << "] = " << eventsDisplay(i) << std::endl;
    for (int i=0; i<static_cast<int>(eventsLabel().size()); ++i)
        std::cout << "eventsLabel[" << i << "] = " << eventsLabel(i) << std::endl;
    std::cout << std::endl;
}

void ezc3d::Header::write(std::fstream &f) const
{
    // write the checksum byte and the start point of header
    f.write(reinterpret_cast<const char*>(&_parametersAddress), ezc3d::BYTE);
    f.write(reinterpret_cast<const char*>(&_checksum), ezc3d::BYTE);

    // Number of data
    f.write(reinterpret_cast<const char*>(&_nb3dPoints), 1*ezc3d::DATA_TYPE::WORD);
    f.write(reinterpret_cast<const char*>(&_nbAnalogsMeasurement), 1*ezc3d::DATA_TYPE::WORD);

    // Idx of first and last frame
    int firstFrame(_firstFrame + 1); // 1-based!
    int lastFrame(_lastFrame + 1); // 1-based!
    f.write(reinterpret_cast<const char*>(&firstFrame), 1*ezc3d::DATA_TYPE::WORD);
    f.write(reinterpret_cast<const char*>(&lastFrame), 1*ezc3d::DATA_TYPE::WORD);

    // Some info
    f.write(reinterpret_cast<const char*>(&_nbMaxInterpGap), 1*ezc3d::DATA_TYPE::WORD);
    f.write(reinterpret_cast<const char*>(&_scaleFactor), 2*ezc3d::DATA_TYPE::WORD);

    // Parameters of analog data
    f.write(reinterpret_cast<const char*>(&_dataStart), 1*ezc3d::DATA_TYPE::WORD);
    f.write(reinterpret_cast<const char*>(&_nbAnalogByFrame), 1*ezc3d::DATA_TYPE::WORD);
    float frameRate(static_cast<float>(_frameRate));
    f.write(reinterpret_cast<const char*>(&frameRate), 2*ezc3d::DATA_TYPE::WORD);
    for (int i=0; i<135; ++i)
        f.write(reinterpret_cast<const char*>(&_emptyBlock1), 1*ezc3d::DATA_TYPE::WORD);

    // Parameters of keys
    f.write(reinterpret_cast<const char*>(&_keyLabelPresent), 1*ezc3d::DATA_TYPE::WORD);
    f.write(reinterpret_cast<const char*>(&_firstBlockKeyLabel), 1*ezc3d::DATA_TYPE::WORD);
    f.write(reinterpret_cast<const char*>(&_fourCharPresent), 1*ezc3d::DATA_TYPE::WORD);

    // Parameters of events
    f.write(reinterpret_cast<const char*>(&_nbEvents), 1*ezc3d::DATA_TYPE::WORD);
    f.write(reinterpret_cast<const char*>(&_emptyBlock2), 1*ezc3d::DATA_TYPE::WORD);
    for (unsigned int i = 0; i < _eventsTime.size(); ++i)
        f.write(reinterpret_cast<const char*>(&_eventsTime[i]), 2*ezc3d::DATA_TYPE::WORD);
    for (unsigned int i = 0; i < _eventsDisplay.size(); ++i)
        f.write(reinterpret_cast<const char*>(&_eventsDisplay[i]), 1*ezc3d::DATA_TYPE::WORD);
    f.write(reinterpret_cast<const char*>(&_emptyBlock3), 1*ezc3d::DATA_TYPE::WORD);
    for (unsigned int i = 0; i < _eventsLabel.size(); ++i){
        const char* event = _eventsLabel[i].c_str();
        f.write(event, 2*ezc3d::DATA_TYPE::WORD);
    }
    for (int i=0; i<22; ++i)
        f.write(reinterpret_cast<const char*>(&_emptyBlock4), 1*ezc3d::DATA_TYPE::WORD);
}


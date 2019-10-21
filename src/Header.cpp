#define EZC3D_API_EXPORTS
///
/// \file Header.cpp
/// \brief Implementation of Header class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "Header.h"
#include "Parameters.h"

ezc3d::Header::Header():
    _nbOfZerosBeforeHeader(0),
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
    _emptyBlock2(0),
    _emptyBlock3(0),
    _emptyBlock4(0),
    _keyLabelPresent(0),
    _firstBlockKeyLabel(0),
    _fourCharPresent(0x3039),
    _nbEvents(0) {
    _eventsTime.resize(18);
    _eventsDisplay.resize(9);
    _eventsLabel.resize(18);
}

ezc3d::Header::Header(
        ezc3d::c3d &c3d,
        std::fstream &file) :
    _nbOfZerosBeforeHeader(0),
    _parametersAddress(2),
    _checksum(0),
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
    _emptyBlock2(0),
    _emptyBlock3(0),
    _emptyBlock4(0),
    _keyLabelPresent(0),
    _firstBlockKeyLabel(0),
    _fourCharPresent(0x3039),
    _nbEvents(0) {
    _eventsTime.resize(18);
    _eventsDisplay.resize(9);
    _eventsLabel.resize(18);
    read(c3d, file);
}

void ezc3d::Header::print() const {
    std::cout << "HEADER" << std::endl;
    std::cout << "nb3dPoints = " << nb3dPoints() << std::endl;
    std::cout << "nbAnalogsMeasurement = "
              << nbAnalogsMeasurement() << std::endl;
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
    for (size_t i=0; i < eventsTime().size(); ++i)
        std::cout << "eventsTime[" << i << "] = "
                  << eventsTime(i) << std::endl;
    for (size_t i=0; i < eventsDisplay().size(); ++i)
        std::cout << "eventsDisplay[" << i << "] = "
                  << eventsDisplay(i) << std::endl;
    for (size_t i=0; i < eventsLabel().size(); ++i)
        std::cout << "eventsLabel[" << i << "] = "
                  << eventsLabel(i) << std::endl;
    std::cout << std::endl;
}

void ezc3d::Header::write(
        std::fstream &f,
        std::streampos &dataStartPosition) const {
    // write the checksum byte and the start point of header
    int parameterAddessDefault(2);
    f.write(reinterpret_cast<const char*>(
                &parameterAddessDefault), ezc3d::BYTE);
    int checksum(0x50);
    f.write(reinterpret_cast<const char*>(&checksum), ezc3d::BYTE);

    // Number of data
    f.write(reinterpret_cast<const char*>(&_nb3dPoints),
            1*ezc3d::DATA_TYPE::WORD);
    f.write(reinterpret_cast<const char*>(&_nbAnalogsMeasurement),
            1*ezc3d::DATA_TYPE::WORD);

    // Idx of first and last frame
    size_t firstFrame(_firstFrame + 1); // 1-based!
    size_t lastFrame(_lastFrame + 1); // 1-based!
    if (lastFrame > 0xFFFF)
        // Combine this with group("POINT").parameter("FRAMES") = -1
        lastFrame = 0xFFFF;
    f.write(reinterpret_cast<const char*>(&firstFrame),
            1*ezc3d::DATA_TYPE::WORD);
    f.write(reinterpret_cast<const char*>(&lastFrame),
            1*ezc3d::DATA_TYPE::WORD);

    // Some info
    f.write(reinterpret_cast<const char*>(&_nbMaxInterpGap),
            1*ezc3d::DATA_TYPE::WORD);
    f.write(reinterpret_cast<const char*>(&_scaleFactor),
            2*ezc3d::DATA_TYPE::WORD);

    // Parameters of analog data
    dataStartPosition = f.tellg();
    // dataStartPosition is to be changed when we know where the data are
    f.write(reinterpret_cast<const char*>(&_dataStart),
            1*ezc3d::DATA_TYPE::WORD);
    f.write(reinterpret_cast<const char*>(&_nbAnalogByFrame),
            1*ezc3d::DATA_TYPE::WORD);
    float frameRate(_frameRate);
    f.write(reinterpret_cast<const char*>(&frameRate),
            2*ezc3d::DATA_TYPE::WORD);
    for (int i=0; i<135; ++i)
        f.write(reinterpret_cast<const char*>(&_emptyBlock1),
                1*ezc3d::DATA_TYPE::WORD);

    // Parameters of keys
    f.write(reinterpret_cast<const char*>(&_keyLabelPresent),
            1*ezc3d::DATA_TYPE::WORD);
    f.write(reinterpret_cast<const char*>(&_firstBlockKeyLabel),
            1*ezc3d::DATA_TYPE::WORD);
    f.write(reinterpret_cast<const char*>(&_fourCharPresent),
            1*ezc3d::DATA_TYPE::WORD);

    // Parameters of events
    f.write(reinterpret_cast<const char*>(&_nbEvents),
            1*ezc3d::DATA_TYPE::WORD);
    f.write(reinterpret_cast<const char*>(&_emptyBlock2),
            1*ezc3d::DATA_TYPE::WORD);
    for (unsigned int i = 0; i < _eventsTime.size(); ++i)
        f.write(reinterpret_cast<const char*>(&_eventsTime[i]),
                2*ezc3d::DATA_TYPE::WORD);
    for (unsigned int i = 0; i < _eventsDisplay.size(); ++i)
        f.write(reinterpret_cast<const char*>(&_eventsDisplay[i]),
                1*ezc3d::DATA_TYPE::WORD);
    f.write(reinterpret_cast<const char*>(&_emptyBlock3),
            1*ezc3d::DATA_TYPE::WORD);
    std::vector<std::string> eventsLabel(_eventsLabel);
    for (unsigned int i = 0; i < eventsLabel.size(); ++i){
        eventsLabel[i].resize(2*ezc3d::DATA_TYPE::WORD);
        f.write(eventsLabel[i].c_str(), 2*ezc3d::DATA_TYPE::WORD);
    }
    for (int i=0; i<22; ++i)
        f.write(reinterpret_cast<const char*>(&_emptyBlock4),
                1*ezc3d::DATA_TYPE::WORD);
}

void ezc3d::Header::read(ezc3d::c3d &c3d, std::fstream &file)
{
    // Parameter address assuming Intel processor
    _parametersAddress = c3d.readUint(PROCESSOR_TYPE::INTEL, file,
                                      1*ezc3d::DATA_TYPE::BYTE, 0,
                                      std::ios::beg);

    // For some reason, some Vicon's file has lot of "0" at the beginning
    // of the file
    // This part loop up to the point no 0 is found
    while (!_parametersAddress){
        _parametersAddress = c3d.readUint(PROCESSOR_TYPE::INTEL,file,
                                          1*ezc3d::DATA_TYPE::BYTE);
        if (file.eof())
            throw std::ios_base::failure("File is empty");
        ++_nbOfZerosBeforeHeader;
    }

    _checksum = c3d.readUint(PROCESSOR_TYPE::INTEL, file,
                             1*ezc3d::DATA_TYPE::BYTE);
    if (_checksum != 0x50) // If checkbyte is wrong
        throw std::ios_base::failure("File must be a valid c3d file");

    // Find which formatting is used
    ezc3d::PROCESSOR_TYPE processorType(readProcessorType(c3d, file));

    // Number of data
    _nb3dPoints = c3d.readUint(processorType, file,
                               1*ezc3d::DATA_TYPE::WORD);
    _nbAnalogsMeasurement = c3d.readUint(processorType, file,
                                         1*ezc3d::DATA_TYPE::WORD);

    // Idx of first and last frame
    _firstFrame = c3d.readUint(processorType, file,
                               1*ezc3d::DATA_TYPE::WORD);
    // First frame is 1-based, but some forgot hence they put 0..
    if (_firstFrame != 0)
        _firstFrame -= 1;
    _lastFrame = c3d.readUint(processorType, file, 1*ezc3d::DATA_TYPE::WORD);
    // Last frame is 1-based, but some forgot  hence they put 0..
    if (_lastFrame != 0)
        _lastFrame -= 1;

    // Some info
    _nbMaxInterpGap = c3d.readUint(processorType, file,
                                   1*ezc3d::DATA_TYPE::WORD);
    _scaleFactor = c3d.readFloat(processorType, file,
                                 2*ezc3d::DATA_TYPE::WORD);

    // Parameters of analog data
    _dataStart = c3d.readUint(processorType, file,
                              1*ezc3d::DATA_TYPE::WORD);
    _nbAnalogByFrame = c3d.readUint(processorType, file,
                                    1*ezc3d::DATA_TYPE::WORD);
    _frameRate = c3d.readFloat(processorType, file);
    _emptyBlock1 = c3d.readInt(processorType, file,
                               135*ezc3d::DATA_TYPE::WORD);

    // Parameters of keys
    _keyLabelPresent = c3d.readUint(processorType, file,
                                    1*ezc3d::DATA_TYPE::WORD);
    _firstBlockKeyLabel = c3d.readUint(processorType, file,
                                       1*ezc3d::DATA_TYPE::WORD);
    _fourCharPresent = c3d.readUint(processorType, file,
                                    1*ezc3d::DATA_TYPE::WORD);

    // Parameters of events
    _nbEvents = c3d.readUint(processorType, file,
                             1*ezc3d::DATA_TYPE::WORD);
    _emptyBlock2 = c3d.readInt(processorType, file,
                               1*ezc3d::DATA_TYPE::WORD);
    for (unsigned int i = 0; i < _eventsTime.size(); ++i)
        _eventsTime[i] = c3d.readFloat(processorType, file);
    for (unsigned int i = 0; i < _eventsDisplay.size(); ++i)
        _eventsDisplay[i] = c3d.readUint(processorType, file,
                                         1*ezc3d::DATA_TYPE::WORD);
    _emptyBlock3 = c3d.readInt(processorType, file,
                               1*ezc3d::DATA_TYPE::WORD);
    for (unsigned int i = 0; i<_eventsLabel.size(); ++i)
        _eventsLabel[i] = c3d.readString(file,
                                         2*ezc3d::DATA_TYPE::WORD);
    _emptyBlock4 = c3d.readInt(processorType, file,
                               22*ezc3d::DATA_TYPE::WORD);
}

size_t ezc3d::Header::nbOfZerosBeforeHeader() const {
    return _nbOfZerosBeforeHeader;
}

size_t ezc3d::Header::parametersAddress() const {
    return _parametersAddress;
}

ezc3d::PROCESSOR_TYPE ezc3d::Header::readProcessorType(
        c3d &c3d, std::fstream &file) {
    // Remember the current position of the cursor
    std::streampos dataPos = file.tellg();

    // Read the processor type (assuming Intel type)
    size_t parametersAddress(
                c3d.readUint(
                    PROCESSOR_TYPE::INTEL, file, 1*ezc3d::DATA_TYPE::BYTE,
                    0, std::ios::beg));
    size_t processorType = c3d.readUint(
                PROCESSOR_TYPE::INTEL, file, 1*ezc3d::DATA_TYPE::BYTE,
                static_cast<int>(
                    256*ezc3d::DATA_TYPE::WORD*(parametersAddress-1))
                + 3*ezc3d::DATA_TYPE::BYTE, std::ios::beg);

    // Put back the cursor in the file
    file.seekg(dataPos);

    if (processorType == 84)
        return ezc3d::PROCESSOR_TYPE::INTEL;
    else if (processorType == 85)
        return ezc3d::PROCESSOR_TYPE::DEC;
    else if (processorType == 86)
        return ezc3d::PROCESSOR_TYPE::MIPS;
    else
        throw std::runtime_error("Could not read the processor type");
}

size_t ezc3d::Header::checksum() const {
    return _checksum;
}

size_t ezc3d::Header::nb3dPoints() const {
    return _nb3dPoints;
}

void ezc3d::Header::nb3dPoints(
        size_t numberOfPoints) {
    _nb3dPoints = numberOfPoints;
}

size_t ezc3d::Header::nbAnalogs() const {
    if (_nbAnalogByFrame == 0)
        return 0;
    else
        return _nbAnalogsMeasurement / _nbAnalogByFrame;
}

void ezc3d::Header::nbAnalogs(
        size_t nbOfAnalogs) {
    _nbAnalogsMeasurement = nbOfAnalogs * _nbAnalogByFrame;
}

size_t ezc3d::Header::nbAnalogsMeasurement() const {
    return _nbAnalogsMeasurement;
}

size_t ezc3d::Header::nbFrames() const {
    if (nb3dPoints() == 0 && nbAnalogs() == 0)
        return 0;
    else
        return _lastFrame - _firstFrame + 1;
}

size_t ezc3d::Header::firstFrame() const {
    return _firstFrame;
}

void ezc3d::Header::firstFrame(
        size_t frame) {
    _firstFrame = frame;
}

size_t ezc3d::Header::lastFrame() const {
    return _lastFrame;
}

void ezc3d::Header::lastFrame(
        size_t frame) {
    _lastFrame = frame;
}

size_t ezc3d::Header::nbMaxInterpGap() const {
    return _nbMaxInterpGap;
}

float ezc3d::Header::scaleFactor() const {
    return _scaleFactor;
}

size_t ezc3d::Header::dataStart() const {
    return _dataStart;
}

size_t ezc3d::Header::nbAnalogByFrame() const {
    return _nbAnalogByFrame;
}

void ezc3d::Header::nbAnalogByFrame(
        size_t nbOfAnalogsByFrame) {
    size_t analogs(nbAnalogs());
    _nbAnalogByFrame = nbOfAnalogsByFrame;
    nbAnalogs(analogs);
}

float ezc3d::Header::frameRate() const {
    return _frameRate;
}

void ezc3d::Header::frameRate(
        float pointFrameRate) {
    _frameRate = pointFrameRate;
}

int ezc3d::Header::emptyBlock1() const {
    return _emptyBlock1;
}

int ezc3d::Header::emptyBlock2() const {
    return _emptyBlock2;
}

int ezc3d::Header::emptyBlock3() const {
    return _emptyBlock3;
}

int ezc3d::Header::emptyBlock4() const {
    return _emptyBlock4;
}

size_t ezc3d::Header::keyLabelPresent() const {
    return _keyLabelPresent;
}

size_t ezc3d::Header::firstBlockKeyLabel() const {
    return _firstBlockKeyLabel;
}

size_t ezc3d::Header::fourCharPresent() const {
    return _fourCharPresent;
}

size_t ezc3d::Header::nbEvents() const {
    return _nbEvents;
}

const std::vector<float>& ezc3d::Header::eventsTime() const {
    return _eventsTime;
}

float ezc3d::Header::eventsTime(size_t idx) const {
    try {
        return _eventsTime.at(idx);
    } catch(std::out_of_range) {
        throw std::out_of_range(
                    "Header::eventsTime method is trying to access the event "
                    + std::to_string(idx) +
                    " while the maximum number of events is "
                    + std::to_string(nbEvents()) + ".");
    }
}

std::vector<size_t> ezc3d::Header::eventsDisplay() const {
    return _eventsDisplay;
}

size_t ezc3d::Header::eventsDisplay(
        size_t idx) const {
    try {
        return _eventsDisplay.at(idx);
    } catch(std::out_of_range) {
        throw std::out_of_range(
                    "Header::eventsDisplay method is trying "
                    "to access the event "
                    + std::to_string(idx) +
                    " while the maximum number of events is "
                    + std::to_string(nbEvents()) + ".");
    }
}

const std::vector<std::string>& ezc3d::Header::eventsLabel() const {
    return _eventsLabel;
}

const std::string& ezc3d::Header::eventsLabel(size_t idx) const {
    try {
        return _eventsLabel.at(idx);
    } catch(std::out_of_range) {
        throw std::out_of_range(
                    "Header::eventsLabel method is trying to access the event "
                    + std::to_string(idx) +
                    " while the maximum number of events is "
                    + std::to_string(nbEvents()) + ".");
    }
}

#define EZC3D_API_EXPORTS
///
/// \file ezc3d.cpp
/// \brief Implementation of ezc3d class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "ezc3d.h"

void ezc3d::removeTrailingSpaces(std::string& s){
    // Remove the spaces at the end of the strings
    for (int i = static_cast<int>(s.size()); i >= 0; --i)
        if (s.size() > 0 && s[s.size()-1] == ' ')
            s.pop_back();
        else
            break;
}

std::string ezc3d::toUpper(const std::string &str){
    std::string new_str = str;
    std::transform(new_str.begin(), new_str.end(), new_str.begin(), ::toupper);
    return new_str;
}

ezc3d::c3d::c3d():
    _filePath(""),
    m_nByteToRead_float(4*ezc3d::DATA_TYPE::BYTE)
{
    c_float = new char[m_nByteToRead_float + 1];

    _header = std::shared_ptr<ezc3d::Header>(new ezc3d::Header());
    _parameters = std::shared_ptr<ezc3d::ParametersNS::Parameters>(new ezc3d::ParametersNS::Parameters());
    _data = std::shared_ptr<ezc3d::DataNS::Data>(new ezc3d::DataNS::Data());
}

ezc3d::c3d::c3d(const std::string &filePath):
    std::fstream(filePath, std::ios::in | std::ios::binary),
    _filePath(filePath),
    m_nByteToRead_float(4*ezc3d::DATA_TYPE::BYTE)
{
    c_float = new char[m_nByteToRead_float + 1];

    if (!is_open())
        throw std::ios_base::failure("Could not open the c3d file");

    // Read all the section
    _header = std::shared_ptr<ezc3d::Header>(new ezc3d::Header(*this));
    _parameters = std::shared_ptr<ezc3d::ParametersNS::Parameters>(new ezc3d::ParametersNS::Parameters(*this));

    // header may be inconsistent with the parameters, so it must be update to make sure sizes are consistent
    updateHeader();

    // Now read the actual data
    _data = std::shared_ptr<ezc3d::DataNS::Data>(new ezc3d::DataNS::Data(*this));

    // Parameters and header may be inconsistent with actual data, so reprocess them if needed
    updateParameters();

    // Close the file
    close();
}

ezc3d::c3d::~c3d()
{
    delete c_float;
}

void ezc3d::c3d::print() const
{
    header().print();
    parameters().print();
    data().print();
}

void ezc3d::c3d::write(const std::string& filePath) const
{
    std::fstream f(filePath, std::ios::out | std::ios::binary);

    // Write the header
    this->header().write(f);

    // Write the parameters
    // We must copy parameters since there is no way to make sure that the number of frames is not higher than 0xFFFF
    ezc3d::ParametersNS::Parameters params(parameters());
    int nFrames(this->parameters().group("POINT").parameter("FRAMES").valuesAsInt()[0]);
    if (nFrames > 0xFFFF){
        ezc3d::ParametersNS::GroupNS::Parameter frames("FRAMES");
        frames.set(-1);
        params.group_nonConst("POINT").parameter(frames);
    }
    params.write(f);

    // Write the data
    this->data().write(f);

    f.close();
}

void ezc3d::c3d::readFile(unsigned int nByteToRead, char * c, int nByteFromPrevious,
                     const  std::ios_base::seekdir &pos)
{
    if (pos != 1)
        this->seekg (nByteFromPrevious, pos); // Move to number analogs
    this->read (c, nByteToRead);
    c[nByteToRead] = '\0'; // Make sure last char is NULL
}

unsigned int ezc3d::c3d::hex2uint(const char * val, unsigned int len){
    int ret(0);
    for (unsigned int i = 0; i < len; i++)
        ret |= static_cast<int>(static_cast<unsigned char>(val[i])) * static_cast<int>(pow(0x100, i));
    return static_cast<unsigned int>(ret);
}

int ezc3d::c3d::hex2int(const char * val, unsigned int len){
    unsigned int tp(hex2uint(val, len));

    // convert to signed int
    // Find max int value
    unsigned int max(0);
    for (unsigned int i=0; i<len; ++i)
        max |= 0xFF * static_cast<unsigned int>(pow(0x100, i));

    // If the value is over uint_max / 2 then it is a negative number
    int out;
    if (tp > max / 2)
        out = static_cast<int>(tp - max - 1);
    else
        out = static_cast<int>(tp);

    return out;
}

int ezc3d::c3d::readInt(unsigned int nByteToRead, int nByteFromPrevious,
            const std::ios_base::seekdir &pos)
{
    char* c = new char[nByteToRead + 1];
    readFile(nByteToRead, c, nByteFromPrevious, pos);

    // make sure it is an int and not an unsigned int
    int out(hex2int(c, nByteToRead));
    delete[] c;
    return out;
}

size_t ezc3d::c3d::readUint(unsigned int nByteToRead, int nByteFromPrevious,
            const std::ios_base::seekdir &pos)
{
    char* c = new char[nByteToRead + 1];
    readFile(nByteToRead, c, nByteFromPrevious, pos);

    // make sure it is an int and not an unsigned int
    size_t out(hex2uint(c, nByteToRead));
    delete[] c;
    return out;
}

float ezc3d::c3d::readFloat(int nByteFromPrevious,
                const std::ios_base::seekdir &pos)
{
    readFile(m_nByteToRead_float, c_float, nByteFromPrevious, pos);
    float out (*reinterpret_cast<float*>(c_float));
    return out;
}

std::string ezc3d::c3d::readString(unsigned int nByteToRead, int nByteFromPrevious,
                              const std::ios_base::seekdir &pos)
{
    char* c = new char[nByteToRead + 1];
    readFile(nByteToRead, c, nByteFromPrevious, pos);
    std::string out(c);
    delete[] c;
    return out;
}

void ezc3d::c3d::readParam(unsigned int dataLenghtInBytes, const std::vector<size_t> &dimension,
                       std::vector<int> &param_data, size_t currentIdx)
{
    for (size_t i = 0; i < dimension[currentIdx]; ++i)
        if (currentIdx == dimension.size()-1)
            param_data.push_back (readInt(dataLenghtInBytes*ezc3d::DATA_TYPE::BYTE));
        else
            readParam(dataLenghtInBytes, dimension, param_data, currentIdx + 1);
}

void ezc3d::c3d::readParam(const std::vector<size_t> &dimension,
                       std::vector<float> &param_data, size_t currentIdx)
{
    for (size_t i = 0; i < dimension[currentIdx]; ++i)
        if (currentIdx == dimension.size()-1)
            param_data.push_back (readFloat());
        else
            readParam(dimension, param_data, currentIdx + 1);
}

void ezc3d::c3d::readParam(const std::vector<size_t> &dimension,
                       std::vector<std::string> &param_data_string)
{
    std::vector<std::string> param_data_string_tp;
    _readMatrix(dimension, param_data_string_tp);

    // Vicon c3d stores text length on first dimension, I am not sure if
    // this is a standard or a custom made stuff. I implemented it like that for now
    if (dimension.size() == 1){
        if (dimension[0] != 0) {
            std::string tp;
            for (size_t j = 0; j < dimension[0]; ++j)
                tp += param_data_string_tp[j];
            ezc3d::removeTrailingSpaces(tp);
            param_data_string.push_back(tp);
        }
    }
    else
        _dispatchMatrix(dimension, param_data_string_tp, param_data_string);
}

size_t ezc3d::c3d::_dispatchMatrix(const std::vector<size_t> &dimension,
                                 const std::vector<std::string> &param_data_in,
                                 std::vector<std::string> &param_data_out, size_t idxInParam,
                                 size_t currentIdx)
{
    for (size_t i = 0; i < dimension[currentIdx]; ++i)
        if (currentIdx == dimension.size()-1){
            std::string tp;
            for (size_t j = 0; j < dimension[0]; ++j){
                tp += param_data_in[idxInParam];
                ++idxInParam;
            }
            ezc3d::removeTrailingSpaces(tp);
            param_data_out.push_back(tp);
        }
        else
            idxInParam = _dispatchMatrix(dimension, param_data_in, param_data_out, idxInParam, currentIdx + 1);
    return idxInParam;
}

void ezc3d::c3d::_readMatrix(const std::vector<size_t> &dimension,
                       std::vector<std::string> &param_data, size_t currentIdx)
{
    for (size_t i = 0; i < dimension[currentIdx]; ++i)
        if (currentIdx == dimension.size()-1)
            param_data.push_back(readString(ezc3d::DATA_TYPE::BYTE));
        else
            _readMatrix(dimension, param_data, currentIdx + 1);
}

const ezc3d::Header& ezc3d::c3d::header() const
{
    return *_header;
}

const ezc3d::ParametersNS::Parameters& ezc3d::c3d::parameters() const
{
    return *_parameters;
}

const ezc3d::DataNS::Data& ezc3d::c3d::data() const
{
    return *_data;
}

void ezc3d::c3d::parameter(const std::string &groupName, const ezc3d::ParametersNS::GroupNS::Parameter &p)
{
    if (!p.name().compare("")){
        throw std::invalid_argument("Parameter must have a name");
    }

    size_t idx;
    try {
        idx = parameters().groupIdx(groupName);
    } catch (std::invalid_argument) {
        _parameters->group(ezc3d::ParametersNS::GroupNS::Group(groupName));
        idx = parameters().groupIdx(groupName);
    }

    _parameters->group_nonConst(idx).parameter(p);

    // Do a sanity check on the header if important stuff like number of frames or number of elements is changed
    updateHeader();
}

void ezc3d::c3d::lockGroup(const std::string &groupName)
{
    _parameters->group_nonConst(groupName).lock();
}

void ezc3d::c3d::unlockGroup(const std::string &groupName)
{
    _parameters->group_nonConst(groupName).unlock();
}

void ezc3d::c3d::frame(const ezc3d::DataNS::Frame &f, size_t idx)
{
    // Make sure f.points().points() is the same as data.f[ANY].points()
    size_t nPoints(static_cast<size_t>(parameters().group("POINT").parameter("USED").valuesAsInt()[0]));
    if (nPoints != 0 && f.points().nbPoints() != nPoints)
        throw std::runtime_error("Number of points in POINT:USED parameter must equal"
                                 "the number of points sent in the frame");

    std::vector<std::string> labels(parameters().group("POINT").parameter("LABELS").valuesAsString());
    for (size_t i=0; i<labels.size(); ++i)
        try {
            f.points().pointIdx(labels[i]);
        } catch (std::invalid_argument) {
            throw std::invalid_argument("All the points in the frame must appear in the POINT:LABELS parameter");
        }

    if (f.points().nbPoints() > 0 && static_cast<double>(parameters().group("POINT").parameter("RATE").valuesAsFloat()[0]) == 0.0){
        throw std::runtime_error("Point frame rate must be specified if you add some");
    }
    if (f.analogs().nbSubframes() > 0 && static_cast<double>(parameters().group("ANALOG").parameter("RATE").valuesAsFloat()[0]) == 0.0){
        throw std::runtime_error("Analog frame rate must be specified if you add some");
    }

    size_t nAnalogs(static_cast<size_t>(parameters().group("ANALOG").parameter("USED").valuesAsInt()[0]));
    size_t subSize(f.analogs().nbSubframes());
    if (subSize != 0){
        size_t nChannel(f.analogs().subframe(0).nbChannels());
        size_t nAnalogByFrames(header().nbAnalogByFrame());
        if (!(nAnalogs==0 && nAnalogByFrames==0) && nChannel != nAnalogs )
            throw std::runtime_error("Number of analogs in ANALOG:USED parameter must equal "
                                     "the number of analogs sent in the frame");
    }

    // Replace the jth frame
    _data->frame(f, idx);
    updateParameters();
}

void ezc3d::c3d::point(const std::string &name){
    if (data().nbFrames() > 0){
        std::vector<ezc3d::DataNS::Frame> dummy_frames;
        ezc3d::DataNS::Points3dNS::Points dummy_pts;
        ezc3d::DataNS::Points3dNS::Point emptyPoint;
        emptyPoint.name(name);
        dummy_pts.point(emptyPoint);
        ezc3d::DataNS::Frame frame;
        frame.add(dummy_pts);
        for (size_t f=0; f<data().nbFrames(); ++f)
            dummy_frames.push_back(frame);
        point(dummy_frames);
    } else {
        updateParameters({name});
    }
}

void ezc3d::c3d::point(const std::vector<ezc3d::DataNS::Frame>& frames)
{
    if (frames.size() == 0 || frames.size() != data().nbFrames())
        throw std::invalid_argument("Size of the array of frames must equal the number of frames already "
                                    "present in the data set");
    if (frames[0].points().nbPoints() == 0)
        throw std::invalid_argument("Points in the frames cannot be empty");

    std::vector<std::string> labels(parameters().group("POINT").parameter("LABELS").valuesAsString());
    for (size_t idx = 0; idx < frames[0].points().nbPoints(); ++idx){
        const std::string &name(frames[0].points().point(idx).name());
        for (size_t i=0; i<labels.size(); ++i)
            if (!name.compare(labels[i]))
                throw std::invalid_argument("The point you try to create already exists in the data set");

        for (size_t f=0; f<data().nbFrames(); ++f)
            _data->frame_nonConst(f).points_nonConst().point(frames[f].points().point(idx));
    }
    updateParameters();
}

void ezc3d::c3d::analog(const std::string &name)
{
    if (data().nbFrames() > 0){
        std::vector<ezc3d::DataNS::Frame> dummy_frames;
        ezc3d::DataNS::AnalogsNS::SubFrame dummy_subframes;
        ezc3d::DataNS::AnalogsNS::Channel emptyChannel;
        emptyChannel.name(name);
        emptyChannel.data(0);
        ezc3d::DataNS::Frame frame;
        dummy_subframes.channel(emptyChannel);
        for (size_t sf = 0; sf < header().nbAnalogByFrame(); ++sf)
            frame.analogs_nonConst().subframe(dummy_subframes);
        for (size_t f=0; f<data().nbFrames(); ++f)
            dummy_frames.push_back(frame);
        analog(dummy_frames);
    } else {
        updateParameters({}, {name});
    }
}

void ezc3d::c3d::analog(const std::vector<ezc3d::DataNS::Frame> &frames)
{
    if (frames.size() != data().nbFrames())
        throw std::invalid_argument("Size of the array of frames must equal the number of frames already "
                                    "present in the data set");
    if (frames[0].analogs().nbSubframes() != header().nbAnalogByFrame())
        throw std::invalid_argument("Size of the subframes in the frames must equal the number of subframes "
                                    "already present in the data set");
    if (frames[0].analogs().subframe(0).nbChannels() == 0)
        throw std::invalid_argument("Channels in the frame cannot be empty");

    std::vector<std::string> labels(parameters().group("ANALOG").parameter("LABELS").valuesAsString());
    for (size_t idx = 0; idx < frames[0].analogs().subframe(0).nbChannels(); ++idx){
        const std::string &name(frames[0].analogs().subframe(0).channel(idx).name());
        for (size_t i=0; i<labels.size(); ++i)
            if (!name.compare(labels[i]))
                throw std::invalid_argument("The channel you try to create already exists in the data set");

        for (size_t f=0; f < data().nbFrames(); ++f){
            for (size_t sf=0; sf < header().nbAnalogByFrame(); ++sf){
                _data->frame_nonConst(f).analogs_nonConst().subframe_nonConst(sf).channel(frames[f].analogs().subframe(sf).channel(idx));
            }
        }
    }
    updateParameters();
}

void ezc3d::c3d::updateHeader()
{
    // Parameter is always consider as the right value. If there is a discrepancy between them, change the header
    if (static_cast<size_t>(parameters().group("POINT").parameter("FRAMES").valuesAsInt()[0]) != header().nbFrames()){
        _header->firstFrame(0);
        _header->lastFrame(static_cast<size_t>(parameters().group("POINT").parameter("FRAMES").valuesAsInt()[0]) - 1);
    }
    float pointRate(parameters().group("POINT").parameter("RATE").valuesAsFloat()[0]);
    float buffer(10000); // For decimal truncature
    if (static_cast<int>(pointRate*buffer) != static_cast<int>(header().frameRate()*buffer)){
        _header->frameRate(pointRate);
    }
    if (static_cast<size_t>(parameters().group("POINT").parameter("USED").valuesAsInt()[0]) != header().nb3dPoints()){
        _header->nb3dPoints(static_cast<size_t>(parameters().group("POINT").parameter("USED").valuesAsInt()[0]));
    }

    // Compare the subframe with data when possible, otherwise go with the parameters
    if (_data != nullptr && data().nbFrames() > 0 && data().frame(0).analogs().nbSubframes() != 0) {
        if (data().frame(0).analogs().nbSubframes() != static_cast<size_t>(header().nbAnalogByFrame()))
            _header->nbAnalogByFrame(data().frame(0).analogs().nbSubframes());
    } else {
        // Should always be greater than 0, but we have to take in account Optotrak lazyness
        if (parameters().group("ANALOG").nbParameters()){
            if (static_cast<size_t>(pointRate) == 0){
                if (static_cast<size_t>(header().nbAnalogByFrame()) != 1)
                    _header->nbAnalogByFrame(1);
            } else {
                if (static_cast<size_t>(parameters().group("ANALOG").parameter("RATE").valuesAsFloat()[0] / pointRate)  != static_cast<size_t>(header().nbAnalogByFrame()))
                    _header->nbAnalogByFrame(static_cast<size_t>(parameters().group("ANALOG").parameter("RATE").valuesAsFloat()[0] / pointRate));
            }
        }
    }

    // Should always be greater than 0, but we have to take in account Optotrak lazyness
    if (parameters().group("ANALOG").nbParameters()){
        if (static_cast<size_t>(parameters().group("ANALOG").parameter("USED").valuesAsInt()[0]) != header().nbAnalogs())
            _header->nbAnalogs(static_cast<size_t>(parameters().group("ANALOG").parameter("USED").valuesAsInt()[0]));
    } else
        _header->nbAnalogs(0);
}

void ezc3d::c3d::updateParameters(const std::vector<std::string> &newPoints, const std::vector<std::string> &newAnalogs)
{
    if (data().nbFrames() != 0 && newPoints.size() > 0)
        throw std::runtime_error("newPoints in updateParameters should only be called on empty c3d");
    if (data().nbFrames() != 0 && newAnalogs.size() > 0)
        throw std::runtime_error("newAnalogs in updateParameters should only be called on empty c3d");

    // If frames has been added
    ezc3d::ParametersNS::GroupNS::Group& grpPoint(_parameters->group_nonConst(parameters().groupIdx("POINT")));
    size_t nFrames(data().nbFrames());
    if (nFrames != static_cast<size_t>(grpPoint.parameter("FRAMES").valuesAsInt()[0])){
        size_t idx(grpPoint.parameterIdx("FRAMES"));
        grpPoint.parameter_nonConst(idx).set(nFrames);
    }

    // If points has been added
    size_t nPoints;
    if (data().nbFrames() > 0)
        nPoints = data().frame(0).points().nbPoints();
    else
        nPoints = parameters().group("POINT").parameter("LABELS").valuesAsString().size() + newPoints.size();
    if (nPoints != static_cast<size_t>(grpPoint.parameter("USED").valuesAsInt()[0])){
        grpPoint.parameter_nonConst("USED").set(nPoints);

        size_t idxLabels(grpPoint.parameterIdx("LABELS"));
        size_t idxDescriptions(grpPoint.parameterIdx("DESCRIPTIONS"));
        size_t idxUnits(grpPoint.parameterIdx("UNITS"));
        std::vector<std::string> labels;
        std::vector<std::string> descriptions;
        std::vector<std::string> units;
        for (size_t i = 0; i < nPoints; ++i){
            std::string name;
            if (data().nbFrames() == 0){
                if (i < parameters().group("POINT").parameter("LABELS").valuesAsString().size())
                    name = parameters().group("POINT").parameter("LABELS").valuesAsString()[i];
                else
                    name = newPoints[i - parameters().group("POINT").parameter("LABELS").valuesAsString().size()];
            } else {
                name = data().frame(0).points().point(i).name();
            }
            labels.push_back(name);
            descriptions.push_back("");
            units.push_back("mm");
        }
        grpPoint.parameter_nonConst(idxLabels).set(labels);
        grpPoint.parameter_nonConst(idxDescriptions).set(descriptions);
        grpPoint.parameter_nonConst(idxUnits).set(units);
    }

    // If analogous data has been added
    ezc3d::ParametersNS::GroupNS::Group& grpAnalog(_parameters->group_nonConst(parameters().groupIdx("ANALOG")));
    size_t nAnalogs;
    if (data().nbFrames() > 0){
        if (data().frame(0).analogs().nbSubframes() > 0)
            nAnalogs = data().frame(0).analogs().subframe(0).nbChannels();
        else
            nAnalogs = 0;
    } else
        nAnalogs = parameters().group("ANALOG").parameter("LABELS").valuesAsString().size() + newAnalogs.size();
    if (nAnalogs != static_cast<size_t>(grpAnalog.parameter("USED").valuesAsInt()[0])){
        grpAnalog.parameter_nonConst("USED").set(nAnalogs);

        size_t idxLabels(static_cast<size_t>(grpAnalog.parameterIdx("LABELS")));
        size_t idxDescriptions(static_cast<size_t>(grpAnalog.parameterIdx("DESCRIPTIONS")));
        std::vector<std::string> labels;
        std::vector<std::string> descriptions;
        for (size_t i = 0; i<nAnalogs; ++i){
            std::string name;
            if (data().nbFrames() == 0){
                if (i < parameters().group("ANALOG").parameter("LABELS").valuesAsString().size())
                    name = parameters().group("ANALOG").parameter("LABELS").valuesAsString()[i];
                else
                    name = newAnalogs[i-parameters().group("ANALOG").parameter("LABELS").valuesAsString().size()];
            } else {
                name = data().frame(0).analogs().subframe(0).channel(i).name();
            }
            labels.push_back(name);
            descriptions.push_back("");
        }
        grpAnalog.parameter_nonConst(idxLabels).set(labels);
        grpAnalog.parameter_nonConst(idxDescriptions).set(descriptions);

        size_t idxScale(grpAnalog.parameterIdx("SCALE"));
        std::vector<float> scales(grpAnalog.parameter(idxScale).valuesAsFloat());
        for (size_t i = grpAnalog.parameter(idxScale).valuesAsFloat().size(); i < nAnalogs; ++i)
            scales.push_back(1);
        grpAnalog.parameter_nonConst(idxScale).set(scales);

        size_t idxOffset(grpAnalog.parameterIdx("OFFSET"));
        std::vector<int> offset(grpAnalog.parameter(idxOffset).valuesAsInt());
        for (size_t i = grpAnalog.parameter(idxOffset).valuesAsInt().size(); i < nAnalogs; ++i)
            offset.push_back(0);
        grpAnalog.parameter_nonConst(idxOffset).set(offset);

        size_t idxUnits(grpAnalog.parameterIdx("UNITS"));
        std::vector<std::string> units(grpAnalog.parameter(idxUnits).valuesAsString());
        for (size_t i = grpAnalog.parameter(idxUnits).valuesAsString().size(); i < nAnalogs; ++i)
            units.push_back("V");
        grpAnalog.parameter_nonConst(idxUnits).set(units);

    }
    updateHeader();
}

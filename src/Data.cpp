#define EZC3D_API_EXPORTS
#include "Data.h"
// Implementation of data class


ezc3d::DataNS::Data::Data()
{

}

ezc3d::DataNS::Data::Data(ezc3d::c3d &file)
{
    // Firstly read a dummy value just prior to the data so it moves the pointer to the right place
    file.readInt(ezc3d::DATA_TYPE::BYTE, 256*ezc3d::DATA_TYPE::WORD*(file.header().parametersAddress()-1) + 256*ezc3d::DATA_TYPE::WORD*file.parameters().nbParamBlock() - ezc3d::DATA_TYPE::BYTE, std::ios::beg); // "- BYTE" so it is just prior

    // Initialize some variables
    if (file.header().nbFrames()>0)
        _frames.resize(static_cast<size_t>(file.header().nbFrames()));

    // Get names of the data
    std::vector<std::string> pointNames;
    if (file.header().nb3dPoints() > 0)
        pointNames = file.parameters().group("POINT").parameter("LABELS").valuesAsString();
    std::vector<std::string> analogNames;
    if (file.header().nbAnalogs() > 0)
        analogNames = file.parameters().group("ANALOG").parameter("LABELS").valuesAsString();

    // Read the actual data
    for (size_t j = 0; j < static_cast<size_t>(file.header().nbFrames()); ++j){
        ezc3d::DataNS::Frame f;
        if (file.header().scaleFactor() < 0){ // if it is float
            // Read point 3d
            ezc3d::DataNS::Points3dNS::Points ptsAtAFrame(file.header().nb3dPoints());
            std::vector<ezc3d::DataNS::Points3dNS::Point>& pts_tp(ptsAtAFrame.points_nonConst());
            for (size_t i = 0; i < static_cast<size_t>(file.header().nb3dPoints()); ++i){
                pts_tp[i].x(file.readFloat());
                pts_tp[i].y(file.readFloat());
                pts_tp[i].z(file.readFloat());
                pts_tp[i].residual(file.readFloat());
                if (i < pointNames.size())
                    pts_tp[i].name(pointNames[i]);
                else {
                    std::stringstream unlabel;
                    unlabel << "unlabeled_point_" << i;
                    pts_tp[i].name(unlabel.str());
                }

            }
            _frames[j].add(ptsAtAFrame); // modified by pts_tp which is an nonconst ref to internal points

            // Read analogs
            ezc3d::DataNS::AnalogsNS::Analogs analog(file.header().nbAnalogByFrame());
            for (int k = 0; k < file.header().nbAnalogByFrame(); ++k){
                ezc3d::DataNS::AnalogsNS::SubFrame sub(file.header().nbAnalogs());
                for (size_t i = 0; i < static_cast<size_t>(file.header().nbAnalogs()); ++i){
                    ezc3d::DataNS::AnalogsNS::Channel c;
                    c.value(file.readFloat());
                    if (i < analogNames.size())
                        c.name(analogNames[i]);
                    else {
                        std::stringstream unlabel;
                        unlabel << "unlabeled_analog_" << i;
                        c.name(unlabel.str());
                    }
                    sub.replaceChannel(static_cast<int>(i), c);
                }
                analog.replaceSubframe(k, sub);
            }
            _frames[j].add(analog);

        }
        else
            throw std::invalid_argument("Points were recorded using int number which is not implemented yet");
    }
}
void ezc3d::DataNS::Data::frame(const ezc3d::DataNS::Frame &f, int j)
{
    if (j < 0)
        _frames.push_back(f);
    else
        _frames[static_cast<size_t>(j)].add(f);
}
std::vector<ezc3d::DataNS::Frame> &ezc3d::DataNS::Data::frames_nonConst()
{
    return _frames;
}
const std::vector<ezc3d::DataNS::Frame>& ezc3d::DataNS::Data::frames() const
{
    return _frames;
}
const ezc3d::DataNS::Frame& ezc3d::DataNS::Data::frame(int idx) const
{
    if (idx < 0 || idx >= static_cast<int>(frames().size()))
        throw std::out_of_range("Wrong number of frames");
    return _frames[static_cast<size_t>(idx)];
}
void ezc3d::DataNS::Data::print() const
{
    for (int i = 0; i < static_cast<int>(frames().size()); ++i){
        std::cout << "Frame " << i << std::endl;
        frame(i).print();
        std::cout << std::endl;
    }
}

void ezc3d::DataNS::Data::write(std::fstream &f) const
{
    for (int i=0; i<static_cast<int>(frames().size()); ++i){
        frame(i).write(f);
    }
}



// Point3d data
ezc3d::DataNS::Points3dNS::Points::Points()
{

}

ezc3d::DataNS::Points3dNS::Points::Points(int nMarkers)
{
    if (nMarkers < 0)
        throw std::out_of_range("Wrong number of markers");
    _points.resize(static_cast<size_t>(nMarkers));
}

void ezc3d::DataNS::Points3dNS::Points::add(const ezc3d::DataNS::Points3dNS::Point &p)
{
    _points.push_back(p);
}

void ezc3d::DataNS::Points3dNS::Points::replace(int idx, const ezc3d::DataNS::Points3dNS::Point &p)
{
    if (idx < 0 || idx >= static_cast<int>(points().size()))
        throw std::out_of_range("Wrong number of idx");
    _points[static_cast<size_t>(idx)] = p;
}
void ezc3d::DataNS::Points3dNS::Points::print() const
{
    for (int i = 0; i < static_cast<int>(points().size()); ++i)
        point(i).print();
}

void ezc3d::DataNS::Points3dNS::Points::write(std::fstream &f) const
{
    for (int i=0; i<static_cast<int>(points().size()); ++i){
        point(i).write(f);
    }
}

const std::vector<ezc3d::DataNS::Points3dNS::Point>& ezc3d::DataNS::Points3dNS::Points::points() const
{
    return _points;
}
std::vector<ezc3d::DataNS::Points3dNS::Point> &ezc3d::DataNS::Points3dNS::Points::points_nonConst()
{
    return _points;
}
int ezc3d::DataNS::Points3dNS::Points::pointIdx(const std::string &pointName) const
{
    for (int i = 0; i<static_cast<int>(points().size()); ++i)
        if (!point(i).name().compare(pointName))
            return i;
    return -1;
}
const ezc3d::DataNS::Points3dNS::Point& ezc3d::DataNS::Points3dNS::Points::point(int idx) const
{
    if (idx < 0 || idx >= static_cast<int>(points().size()))
        throw std::out_of_range("Tried to access wrong index for points data");
    return _points[static_cast<size_t>(idx)];
}
const ezc3d::DataNS::Points3dNS::Point &ezc3d::DataNS::Points3dNS::Points::point(const std::string &pointName) const
{
    int idx(pointIdx(pointName));
    if (idx < 0)
        throw std::invalid_argument("Point name was not found within the points");
    return point(idx);
}
void ezc3d::DataNS::Points3dNS::Point::print() const
{
    std::cout << name() << " = [" << x() << ", " << y() << ", " << z() << "]; Residual = " << residual() << std::endl;
}



ezc3d::DataNS::Points3dNS::Point::Point() :
    _name("")
{
    _data.resize(4);
}
void ezc3d::DataNS::Points3dNS::Point::write(std::fstream &f) const
{
    f.write(reinterpret_cast<const char*>(&_data[0]), ezc3d::DATA_TYPE::FLOAT);
    f.write(reinterpret_cast<const char*>(&_data[1]), ezc3d::DATA_TYPE::FLOAT);
    f.write(reinterpret_cast<const char*>(&_data[2]), ezc3d::DATA_TYPE::FLOAT);
    f.write(reinterpret_cast<const char*>(&_data[3]), ezc3d::DATA_TYPE::FLOAT);
}
float ezc3d::DataNS::Points3dNS::Point::x() const
{
    return _data[0];
}
void ezc3d::DataNS::Points3dNS::Point::x(float x)
{
    _data[0] = x;
}

float ezc3d::DataNS::Points3dNS::Point::y() const
{
    return _data[1];
}
void ezc3d::DataNS::Points3dNS::Point::y(float y)
{
    _data[1] = y;
}

float ezc3d::DataNS::Points3dNS::Point::z() const
{
    return _data[2];
}
void ezc3d::DataNS::Points3dNS::Point::z(float z)
{
    _data[2] = z;
}

const std::vector<float> ezc3d::DataNS::Points3dNS::Point::data() const
{
    return _data;
}
float ezc3d::DataNS::Points3dNS::Point::residual() const
{
    return _data[3];
}
void ezc3d::DataNS::Points3dNS::Point::residual(float residual){
    _data[3] = residual;
}
const std::string& ezc3d::DataNS::Points3dNS::Point::name() const
{
    return _name;
}
void ezc3d::DataNS::Points3dNS::Point::name(const std::string &name)
{
    std::string name_copy = name;
    ezc3d::removeSpacesOfAString(name_copy);
    _name = name_copy;
}




// Analog data
ezc3d::DataNS::AnalogsNS::SubFrame::SubFrame()
{

}
ezc3d::DataNS::AnalogsNS::SubFrame::SubFrame(int nChannels)
{
    if (nChannels < 0)
        throw std::out_of_range("Number of channels can't be under 0");
    _channels.resize(static_cast<size_t>(nChannels));
}
void ezc3d::DataNS::AnalogsNS::SubFrame::print() const
{
    for (int i = 0; i < static_cast<int>(channels().size()); ++i){
        channel(i).print();
    }
}
void ezc3d::DataNS::AnalogsNS::SubFrame::write(std::fstream &f) const
{
    for (int i = 0; i < static_cast<int>(channels().size()); ++i){
        channel(i).write(f);
    }
}

void ezc3d::DataNS::AnalogsNS::SubFrame::addChannel(const ezc3d::DataNS::AnalogsNS::Channel& channel)
{
    _channels.push_back(channel);
}
void ezc3d::DataNS::AnalogsNS::SubFrame::replaceChannel(int idx, const ezc3d::DataNS::AnalogsNS::Channel &channel)
{
    if (idx < 0 || idx >= static_cast<int>(channels().size()))
        throw std::out_of_range("Tried to access wrong index for channel data");
    _channels[static_cast<size_t>(idx)] = channel;
}
void ezc3d::DataNS::AnalogsNS::SubFrame::addChannels(const std::vector<ezc3d::DataNS::AnalogsNS::Channel>& allChannelsData)
{
    _channels = allChannelsData;
}
std::vector<ezc3d::DataNS::AnalogsNS::Channel>& ezc3d::DataNS::AnalogsNS::SubFrame::channels_nonConst()
{
    return _channels;
}
const std::vector<ezc3d::DataNS::AnalogsNS::Channel>& ezc3d::DataNS::AnalogsNS::SubFrame::channels() const
{
    return _channels;
}
const ezc3d::DataNS::AnalogsNS::Channel &ezc3d::DataNS::AnalogsNS::SubFrame::channel(int idx) const
{
    if (idx < 0 || idx >= static_cast<int>(_channels.size()))
        throw std::out_of_range("Tried to access wrong index for analog data");
    return _channels[static_cast<size_t>(idx)];
}
const ezc3d::DataNS::AnalogsNS::Channel &ezc3d::DataNS::AnalogsNS::SubFrame::channel(std::string channelName) const
{
    for (int i = 0; i < static_cast<int>(channels().size()); ++i)
        if (!channel(i).name().compare(channelName))
            return channel(i);
    throw std::invalid_argument("Analog name was not found within the analogs");
}
float ezc3d::DataNS::AnalogsNS::Channel::value() const
{
    return _value;
}
void ezc3d::DataNS::AnalogsNS::Channel::value(float v)
{
    _value = v;
}
void ezc3d::DataNS::AnalogsNS::Channel::print() const
{
    std::cout << "Analog[" << name() << "] = " << value() << std::endl;
}
void ezc3d::DataNS::AnalogsNS::Channel::write(std::fstream &f) const
{
    f.write(reinterpret_cast<const char*>(&_value), ezc3d::DATA_TYPE::FLOAT);
}
const std::string& ezc3d::DataNS::AnalogsNS::Channel::name() const
{
    return _name;
}

void ezc3d::DataNS::AnalogsNS::Channel::name(const std::string &name)
{
    std::string name_copy = name;
    ezc3d::removeSpacesOfAString(name_copy);
    _name = name_copy;
}




// Frame data
ezc3d::DataNS::Frame::Frame()
{
    _points = std::shared_ptr<ezc3d::DataNS::Points3dNS::Points>(new ezc3d::DataNS::Points3dNS::Points());
    _analogs = std::shared_ptr<ezc3d::DataNS::AnalogsNS::Analogs>(new ezc3d::DataNS::AnalogsNS::Analogs());
}
void ezc3d::DataNS::Frame::print() const
{
    points().print();
    analogs().print();
}
void ezc3d::DataNS::Frame::write(std::fstream &f) const
{
    points().write(f);
    analogs().write(f);
}
ezc3d::DataNS::Points3dNS::Points &ezc3d::DataNS::Frame::points_nonConst() const
{
    return *_points;
}
const ezc3d::DataNS::Points3dNS::Points& ezc3d::DataNS::Frame::points() const
{
    return *_points;
}
ezc3d::DataNS::AnalogsNS::Analogs &ezc3d::DataNS::Frame::analogs_nonConst() const
{
    return *_analogs;
}
const ezc3d::DataNS::AnalogsNS::Analogs& ezc3d::DataNS::Frame::analogs() const
{
    return *_analogs;
}
void ezc3d::DataNS::Frame::add(const ezc3d::DataNS::AnalogsNS::Analogs &analogs_frame)
{
    _analogs = std::shared_ptr<ezc3d::DataNS::AnalogsNS::Analogs>(new ezc3d::DataNS::AnalogsNS::Analogs(analogs_frame));
}
void ezc3d::DataNS::Frame::add(const ezc3d::DataNS::Points3dNS::Points &point3d_frame)
{
    _points = std::shared_ptr<ezc3d::DataNS::Points3dNS::Points>(new ezc3d::DataNS::Points3dNS::Points(point3d_frame));
}
void ezc3d::DataNS::Frame::add(const ezc3d::DataNS::Frame &frame)
{
    add(frame.points(), frame.analogs());
}
void ezc3d::DataNS::Frame::add(const ezc3d::DataNS::Points3dNS::Points &point3d_frame, const ezc3d::DataNS::AnalogsNS::Analogs &analog_frame)
{
    add(point3d_frame);
    add(analog_frame);
}
const std::vector<ezc3d::DataNS::AnalogsNS::SubFrame>& ezc3d::DataNS::AnalogsNS::Analogs::subframes() const
{
    return _subframe;
}
std::vector<ezc3d::DataNS::AnalogsNS::SubFrame> &ezc3d::DataNS::AnalogsNS::Analogs::subframes_nonConst()
{
    return _subframe;
}
const ezc3d::DataNS::AnalogsNS::SubFrame& ezc3d::DataNS::AnalogsNS::Analogs::subframe(int idx) const
{
    if (idx < 0 || idx >= static_cast<int>(_subframe.size()))
        throw std::out_of_range("Tried to access wrong subframe index for analog data");
    return _subframe[static_cast<size_t>(idx)];
}


ezc3d::DataNS::AnalogsNS::Analogs::Analogs()
{

}
ezc3d::DataNS::AnalogsNS::Analogs::Analogs(int nSubframes)
{
    if (nSubframes < 0)
        throw std::out_of_range("Number of subframes can't be under 0");
    _subframe.resize(static_cast<size_t>(nSubframes));
}
void ezc3d::DataNS::AnalogsNS::Analogs::print() const
{
    for (int i = 0; i < static_cast<int>(subframes().size()); ++i){
        std::cout << "Subframe = " << i << std::endl;
        subframe(i).print();
        std::cout << std::endl;
    }
}
void ezc3d::DataNS::AnalogsNS::Analogs::write(std::fstream &f) const
{
    for (int i = 0; i < static_cast<int>(subframes().size()); ++i){
        subframe(i).write(f);
    }
}
void ezc3d::DataNS::AnalogsNS::Analogs::addSubframe(const ezc3d::DataNS::AnalogsNS::SubFrame& subframe)
{
    _subframe.push_back(subframe);
}
void ezc3d::DataNS::AnalogsNS::Analogs::replaceSubframe(int idx, const ezc3d::DataNS::AnalogsNS::SubFrame& subframe)
{
    if (idx < 0 || idx >= static_cast<int>(_subframe.size()))
        throw std::out_of_range("Tried to access wrong subframe index for analog data");
    _subframe[static_cast<size_t>(idx)] = subframe;
}

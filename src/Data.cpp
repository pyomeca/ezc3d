#include "Data.h"
// Implementation of data class


ezC3D::Data::Data::Data(ezC3D::C3D &file)
{
    // Firstly read a dummy value just prior to the data so it moves the pointer to the right place
    file.readInt(ezC3D::READ_SIZE::BYTE, 256*ezC3D::READ_SIZE::WORD*(file.header().parametersAddress()-1) + 256*ezC3D::READ_SIZE::WORD*file.parameters().nbParamBlock() - ezC3D::READ_SIZE::BYTE, std::ios::beg); // "- BYTE" so it is just prior

    // Read the actual data
    for (int j = 0; j < file.header().nbFrames(); ++j){
        ezC3D::Data::Data::Frame frame;

        // Get names of the data
        std::vector<std::string> pointNames(file.parameters().group("POINT").parameter("LABELS").stringValues());
        std::vector<std::string> analogNames(file.parameters().group("ANALOG").parameter("LABELS").stringValues());

        // Read point 3d
        if (file.header().scaleFactor() < 0){
            ezC3D::Data::Data::Points3d pts;
            for (int i = 0; i < file.header().nb3dPoints(); ++i){
                ezC3D::Data::Data::Points3d::Point p;
                p.x(file.readFloat());
                p.y(file.readFloat());
                p.z(file.readFloat());
                p.residual(file.readFloat());
                if (i < pointNames.size())
                    p.name(pointNames[i]);
                else
                    p.name("unlabeled_point_" + i);
                pts.add(p);
            }
            frame.add(pts);

            ezC3D::Data::Data::Analogs analog;
            for (int k = 0; k < file.header().nbAnalogByFrame(); ++k){
                ezC3D::Data::Data::Analogs::SubFrame sub;
                for (int i = 0; i < file.header().nbAnalogs(); ++i){
                    ezC3D::Data::Data::Analogs::SubFrame::Channel c;
                    c.value(file.readFloat());
                    if (i < pointNames.size())
                        c.name(analogNames[i]);
                    else
                        c.name("unlabeled_analog_" + i);
                    sub.addChannel(c);
                }
                analog.addSubframe(sub);
            }
            frame.add(analog);
        }
        else
            throw std::invalid_argument("Points were recorded using int number which is not implemented yet");

        _frames.push_back(frame);
    }

}
const std::vector<ezC3D::Data::Data::Frame>& ezC3D::Data::Data::frames() const
{
    return _frames;
}
const ezC3D::Data::Data::Frame& ezC3D::Data::Data::frame(int idx) const
{
    if (idx < 0 || idx >= frames().size())
        throw std::out_of_range("Wrong number of frames");
    return _frames[idx];
}
void ezC3D::Data::Data::print() const
{
    for (int i = 0; i < frames().size(); ++i){
        std::cout << "Frame " << i << std::endl;
        frame(i).print();
        std::cout << std::endl;
    }
}



// Point3d data
void ezC3D::Data::Data::Points3d::add(ezC3D::Data::Data::Points3d::Point p)
{
    _points.push_back(p);
}
void ezC3D::Data::Data::Points3d::print() const
{
    for (int i = 0; i < points().size(); ++i)
        point(i).print();
}

const std::vector<ezC3D::Data::Data::Points3d::Point>& ezC3D::Data::Data::Points3d::points() const
{
    return _points;
}

const ezC3D::Data::Data::Points3d::Point& ezC3D::Data::Data::Points3d::point(int idx) const
{
    if (idx < 0 || idx >= points().size())
        throw std::out_of_range("Tried to access wrong index for points data");
    return _points[idx];
}
const ezC3D::Data::Data::Points3d::Point &ezC3D::Data::Data::Points3d::point(const std::string &pointName) const
{
    for (int i = 0; i<points().size(); ++i)
        if (!point(i).name().compare(pointName))
            return point(i);
    throw std::invalid_argument("Point name was not found within the points");
}
void ezC3D::Data::Data::Points3d::Point::print() const
{
    std::cout << name() << " = [" << x() << ", " << y() << ", " << z() << "]; Residual = " << residual() << std::endl;
}



ezC3D::Data::Data::Points3d::Point::Point() :
    _name(""),
    _idxInData(-1)
{

}
float ezC3D::Data::Data::Points3d::Point::x() const
{
    return _x;
}
void ezC3D::Data::Data::Points3d::Point::x(float x)
{
    _x = x;
}

float ezC3D::Data::Data::Points3d::Point::y() const
{
    return _y;
}
void ezC3D::Data::Data::Points3d::Point::y(float y)
{
    _y = y;
}

float ezC3D::Data::Data::Points3d::Point::z() const
{
    return _z;
}
void ezC3D::Data::Data::Points3d::Point::z(float z)
{
    _z = z;
}
float ezC3D::Data::Data::Points3d::Point::residual() const
{
    return _residual;
}
void ezC3D::Data::Data::Points3d::Point::residual(float residual){
    _residual = residual;
}
const std::string& ezC3D::Data::Data::Points3d::Point::name() const
{
    return _name;
}
void ezC3D::Data::Data::Points3d::Point::name(const std::string &name)
{
    _name = name;
}
int ezC3D::Data::Data::Points3d::Point::idxInData() const
{
    return _idxInData;
}
void ezC3D::Data::Data::Points3d::Point::idxInData(int idxInData)
{
    _idxInData = idxInData;
}




// Analog data
void ezC3D::Data::Data::Analogs::SubFrame::print() const
{
    for (int i = 0; i < channels().size(); ++i){
        channel(i).print();
    }
}

void ezC3D::Data::Data::Analogs::SubFrame::addChannel(ezC3D::Data::Data::Analogs::SubFrame::Channel channel)
{
    _channels.push_back(channel);
}
void ezC3D::Data::Data::Analogs::SubFrame::addChannels(const std::vector<ezC3D::Data::Data::Analogs::SubFrame::Channel>& allChannelsData)
{
    _channels = allChannelsData;
}
const std::vector<ezC3D::Data::Data::Analogs::SubFrame::Channel>& ezC3D::Data::Data::Analogs::SubFrame::channels() const
{
    return _channels;
}
const ezC3D::Data::Data::Analogs::SubFrame::Channel &ezC3D::Data::Data::Analogs::SubFrame::channel(int idx) const
{
    if (idx < 0 || idx >= _channels.size())
        throw std::out_of_range("Tried to access wrong index for analog data");
    return _channels[idx];
}
const ezC3D::Data::Data::Analogs::SubFrame::Channel &ezC3D::Data::Data::Analogs::SubFrame::channel(std::string channelName) const
{
    for (int i = 0; i < channels().size(); ++i)
        if (!channel(i).name().compare(channelName))
            return channel(i);
    throw std::invalid_argument("Analog name was not found within the analogs");
}
float ezC3D::Data::Data::Analogs::SubFrame::Channel::value() const
{
    return _value;
}
void ezC3D::Data::Data::Analogs::SubFrame::Channel::value(float v)
{
    _value = v;
}
void ezC3D::Data::Data::Analogs::SubFrame::Channel::print() const
{
    std::cout << "Analog[" << name() << "] = " << value() << std::endl;
}

const std::string& ezC3D::Data::Data::Analogs::SubFrame::Channel::name() const
{
    return _name;
}

void ezC3D::Data::Data::Analogs::SubFrame::Channel::name(const std::string &name)
{
    _name = name;
}




// Frame data
void ezC3D::Data::Data::Frame::print() const
{
    points()->print();
    analogs()->print();
}
const std::shared_ptr<ezC3D::Data::Data::Points3d>& ezC3D::Data::Data::Frame::points() const
{
    return _points;
}
const std::shared_ptr<ezC3D::Data::Data::Analogs>& ezC3D::Data::Data::Frame::analogs() const
{
    return _analogs;
}
void ezC3D::Data::Data::Frame::add(ezC3D::Data::Data::Analogs analogs_frame)
{
    _analogs = std::shared_ptr<ezC3D::Data::Data::Analogs>(new ezC3D::Data::Data::Analogs(analogs_frame));
}
void ezC3D::Data::Data::Frame::add(ezC3D::Data::Data::Points3d point3d_frame)
{
    _points = std::shared_ptr<ezC3D::Data::Data::Points3d>(new ezC3D::Data::Data::Points3d(point3d_frame));
}
void ezC3D::Data::Data::Frame::add(ezC3D::Data::Data::Points3d point3d_frame, ezC3D::Data::Data::Analogs analog_frame)
{
    add(point3d_frame);
    add(analog_frame);
}
const std::vector<ezC3D::Data::Data::Analogs::SubFrame>& ezC3D::Data::Data::Analogs::subframes() const
{
    return _subframe;
}
const ezC3D::Data::Data::Analogs::SubFrame& ezC3D::Data::Data::Analogs::subframe(int idx) const
{
    if (idx < 0 || idx >= _subframe.size())
        throw std::out_of_range("Tried to access wrong subframe index for analog data");
    return _subframe[idx];
}
void ezC3D::Data::Data::Analogs::print()
{
    for (int i = 0; i < subframes().size(); ++i){
        std::cout << "Subframe = " << i << std::endl;
        subframe(i).print();
        std::cout << std::endl;
    }
}
void ezC3D::Data::Data::Analogs::addSubframe(const ezC3D::Data::Data::Analogs::SubFrame& subframe)
{
    _subframe.push_back(subframe);
}

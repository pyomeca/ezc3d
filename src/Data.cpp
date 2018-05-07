#include "Data.h"
// Implementation of data class


ezC3D::Data::Data(ezC3D &file)
{
    // Firstly read a dummy value just prior to the data so it moves the pointer to the right place
    file.readInt(ezC3D::READ_SIZE::BYTE, 256*ezC3D::READ_SIZE::WORD*(file.header()->parametersAddress()-1) + 256*ezC3D::READ_SIZE::WORD*file.parameters()->nbParamBlock() - ezC3D::READ_SIZE::BYTE, std::ios::beg); // "- BYTE" so it is just prior

    // Read the actual data
    for (int j = 0; j < file.header()->nbFrames(); ++j){
        ezC3D::Data::Frame frame;

        // Read point 3d
        if (file.header()->scaleFactor() < 0){
            ezC3D::Data::Points3d pts;
            for (int i = 0; i < file.header()->nb3dPoints(); ++i){
                ezC3D::Data::Points3d::Point p;
                p.x(file.readFloat());
                p.y(file.readFloat());
                p.z(file.readFloat());
                p.residual(file.readFloat());
                pts.add(p);
            }
            frame.add(pts);

            ezC3D::Data::Analogs a;
            for (int k = 0; k < file.header()->nbAnalogByFrame(); ++k){
                for (int i = 0; i < file.header()->nbAnalogs(); ++i){
                    ezC3D::Data::Analogs::Channel c;
                    c.value(file.readFloat());
                    a.addChannel(c);
                }
                frame.add(a);
            }
        }
        else
            throw std::invalid_argument("Points were recorded using int number which is not implemented yet");

        _frames.push_back(frame);
        std::cout << std::endl;
    }

}

const std::vector<ezC3D::Data::Frame>& ezC3D::Data::frames() const
{
    return _frames;
}
const ezC3D::Data::Frame& ezC3D::Data::frame(int idx) const
{
    if (idx < 0 || idx >= frames().size())
        throw std::out_of_range("Wrong number of frames");
    return _frames[idx];
}
void ezC3D::Data::print() const
{
    for (int i = 0; i < frames().size(); ++i){
        frame(i).print();
    }
}



// Point3d data
void ezC3D::Data::Points3d::add(ezC3D::Data::Points3d::Point p)
{
    _points.push_back(p);
}
void ezC3D::Data::Points3d::print() const
{
    for (int i = 0; i < points().size(); ++i)
        point(i).print();
}

const std::vector<ezC3D::Data::Points3d::Point>& ezC3D::Data::Points3d::points() const
{
    return _points;
}

const ezC3D::Data::Points3d::Point& ezC3D::Data::Points3d::point(int idx) const
{
    if (idx < 0 || idx >= points().size())
        throw std::out_of_range("Tried to access wrong index for points data");
    return _points[idx];
}

void ezC3D::Data::Points3d::Point::print() const
{
    std::cout << x() << ", " << y() << ", " << z() << "]; Residual = " << residual() << std::endl;
}

float ezC3D::Data::Points3d::Point::x() const
{
    return _x;
}
void ezC3D::Data::Points3d::Point::x(float x)
{
    _x = x;
}

float ezC3D::Data::Points3d::Point::y() const
{
    return _y;
}
void ezC3D::Data::Points3d::Point::y(float y)
{
    _y = y;
}

float ezC3D::Data::Points3d::Point::z() const
{
    return _z;
}
void ezC3D::Data::Points3d::Point::z(float z)
{
    _z = z;
}

float ezC3D::Data::Points3d::Point::residual() const
{
    return _residual;
}
void ezC3D::Data::Points3d::Point::residual(float residual){
    _residual = residual;
}


// Analog data
void ezC3D::Data::Analogs::print() const
{
    for (int i = 0; i < channels().size(); ++i){
        channel(i).print();
    }
}

void ezC3D::Data::Analogs::addChannel(ezC3D::Data::Analogs::Channel channel)
{
    _channels.push_back(channel);
}
void ezC3D::Data::Analogs::addChannels(const std::vector<ezC3D::Data::Analogs::Channel>& allChannelsData)
{
    _channels = allChannelsData;
}
const std::vector<ezC3D::Data::Analogs::Channel>& ezC3D::Data::Analogs::channels() const
{
    return _channels;
}
ezC3D::Data::Analogs::Channel ezC3D::Data::Analogs::channel(int channel) const
{
    if (channel < 0 || channel >= _channels.size())
        throw std::out_of_range("Tried to access wrong index for analog data");
    return _channels[channel];
}
float ezC3D::Data::Analogs::Channel::value() const
{
    return _value;
}
void ezC3D::Data::Analogs::Channel::value(float v)
{
    _value = v;
}
void ezC3D::Data::Analogs::Channel::print() const
{
    std::cout << value() << std::endl;
}




// Frame data
void ezC3D::Data::Frame::print() const
{
    _points->print();
    _analogs->print();
}
void ezC3D::Data::Frame::add(ezC3D::Data::Analogs analogs_frame)
{
    _analogs = std::shared_ptr<ezC3D::Data::Analogs>(new ezC3D::Data::Analogs(analogs_frame));
}
void ezC3D::Data::Frame::add(ezC3D::Data::Points3d point3d_frame)
{
    _points = std::shared_ptr<ezC3D::Data::Points3d>(new ezC3D::Data::Points3d(point3d_frame));
}
void ezC3D::Data::Frame::add(ezC3D::Data::Points3d point3d_frame, ezC3D::Data::Analogs analog_frame)
{
    add(point3d_frame);
    add(analog_frame);
}

//// Frame data
//void ezC3D::Frame::print()
//{
//    _points.print();
//}
//void ezC3D::Frame::add(ezC3D::Analog analog_frame){
//    if (_points.size() != 0)
//        throw std::range_error("Analogs and Points can't be added separately, unless they are the only available data");
//    _analogs.push_back(analog_frame);
//}

//void ezC3D::Frame::add(ezC3D::Point3d point3d_frame)
//{
//    if (_analogs.size() != 0)
//        throw std::range_error("Analogs and Points can't be added separately, unless they are the only available data");
//    _points.push_back(point3d_frame);
//}

//void ezC3D::Frame::add(ezC3D::Point3d point3d_frame, ezC3D::Analog analog_frame)
//{
//    _points.push_back(point3d_frame);
//    _analogs.push_back(analog_frame);
//}



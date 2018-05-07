#include "DataHolder.h"
// Implementation of data class

// Point3d data
void ezC3D::Point3d::print()
{
    std::cout << x() << ", " << y() << ", " << z() << "]; Residual = " << residual() << std::endl;
}

float ezC3D::Point3d::x(){
    return _x;
}
void ezC3D::Point3d::x(float x){
    _x = x;
}

float ezC3D::Point3d::y(){
    return _y;
}
void ezC3D::Point3d::y(float y){
    _y = y;
}

float ezC3D::Point3d::z(){
    return _z;
}
void ezC3D::Point3d::z(float z){
    _z = z;
}

float ezC3D::Point3d::residual(){
    return _residual;
}
void ezC3D::Point3d::residual(float residual){
    _residual = residual;
}


// Analog data
void ezC3D::Analog::print()
{
    for (int i = 0; i < _channels.size(); ++i){
        std::cout << "Analog [" << i << "] = " << data(i) << std::endl;
    }
}

void ezC3D::Analog::addChannel(float oneChannelData)
{
    _channels.push_back(oneChannelData);
}
void ezC3D::Analog::addChannels(const std::vector<float>& allChannelsData)
{
    _channels = allChannelsData;
}
const std::vector<float>& ezC3D::Analog::data() const
{
    return _channels;
}
float ezC3D::Analog::data(int channel) const
{
    if (channel < 0 || channel >= _channels.size())
        throw std::out_of_range("Tried to access wrong index for analog data");
    return _channels[channel];
}





// Frame data
void ezC3D::Frame::print()
{
    _points.print();
    _analogs.print();
}
void ezC3D::Frame::add(ezC3D::Analog analog_frame){
    _analogs = analog_frame;
}
void ezC3D::Frame::add(ezC3D::Point3d point3d_frame)
{
    _points = point3d_frame;
}
void ezC3D::Frame::add(ezC3D::Point3d point3d_frame, ezC3D::Analog analog_frame)
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

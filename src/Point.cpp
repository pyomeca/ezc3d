#define EZC3D_API_EXPORTS
#include "Point.h"
// Implementation of Point class

ezc3d::DataNS::Points3dNS::Point::Point() :
    _name("")
{
    _data.resize(4);
}
ezc3d::DataNS::Points3dNS::Point::Point(const ezc3d::DataNS::Points3dNS::Point &p) :
    _name(p.name())
{
    _data.resize(4);
    x(p.x());
    y(p.y());
    z(p.z());
    residual(0);
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

std::vector<float> ezc3d::DataNS::Points3dNS::Point::data_nonConst()
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
    ezc3d::removeTrailingSpaces(name_copy);
    _name = name_copy;
}

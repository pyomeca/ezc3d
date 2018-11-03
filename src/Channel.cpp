#define EZC3D_API_EXPORTS
#include "Channel.h"
// Implementation of Channel class

ezc3d::DataNS::AnalogsNS::Channel::Channel(const std::string &name) :
    _name(name)
{

}

ezc3d::DataNS::AnalogsNS::Channel::Channel(const ezc3d::DataNS::AnalogsNS::Channel &channel) :
    _name(channel._name),
    _data(channel._data)
{

}

void ezc3d::DataNS::AnalogsNS::Channel::print() const
{
    std::cout << "Analog[" << name() << "] = " << data() << std::endl;
}

void ezc3d::DataNS::AnalogsNS::Channel::write(std::fstream &f) const
{
    f.write(reinterpret_cast<const char*>(&_data), ezc3d::DATA_TYPE::FLOAT);
}

const std::string& ezc3d::DataNS::AnalogsNS::Channel::name() const
{
    return _name;
}

void ezc3d::DataNS::AnalogsNS::Channel::name(const std::string &name)
{
    std::string name_copy = name;
    ezc3d::removeTrailingSpaces(name_copy);
    _name = name_copy;
}

float ezc3d::DataNS::AnalogsNS::Channel::data() const
{
    return _data;
}

void ezc3d::DataNS::AnalogsNS::Channel::data(float v)
{
    _data = v;
}

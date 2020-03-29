#define EZC3D_API_EXPORTS
///
/// \file Channel.cpp
/// \brief Implementation of Channel class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "Channel.h"

ezc3d::DataNS::AnalogsNS::Channel::Channel() {

}

ezc3d::DataNS::AnalogsNS::Channel::Channel(
        const ezc3d::DataNS::AnalogsNS::Channel &channel) :
    _data(channel._data) {
}

void ezc3d::DataNS::AnalogsNS::Channel::print() const {
    std::cout << "Analog = " << data() << std::endl;
}

void ezc3d::DataNS::AnalogsNS::Channel::write(
        std::fstream &f,
        double scaleFactor) const {
    float data(static_cast<float>(_data / fabs(scaleFactor)));
    f.write(reinterpret_cast<const char*>(&data), ezc3d::DATA_TYPE::FLOAT);
}

double ezc3d::DataNS::AnalogsNS::Channel::data() const {
    return _data;
}

void ezc3d::DataNS::AnalogsNS::Channel::data(
        double v) {
    _data = v;
}

bool ezc3d::DataNS::AnalogsNS::Channel::isEmpty() const {
    if (static_cast<double>(data()) == 0.0) {
        return true;
    }
    else {
        return false;
    }
}

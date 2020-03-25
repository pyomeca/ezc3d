#define EZC3D_API_EXPORTS
///
/// \file Vector3d.cpp
/// \brief Implementation of Vector3d class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "Vector3d.h"

#include <bitset>

ezc3d::DataNS::Vector3d::Vector3d()
{
    _data.resize(3, 0);
}

ezc3d::DataNS::Vector3d::Vector3d(
        const ezc3d::DataNS::Vector3d &p) {
    _data.resize(3);
    x(p.x());
    y(p.y());
    z(p.z());
}

void ezc3d::DataNS::Vector3d::print() const {
    std::cout << " Position = ["
              << x() << ", "
              << y() << ", "
              << z() << "];"
              << std::endl;
}


bool ezc3d::DataNS::Vector3d::isValid() const
{
    if (std::isnan(_data[0])
            || std::isnan(_data[1])
            || std::isnan(_data[2])) {
        return false;
    }
    else {
        return true;
    }
}

const std::vector<float> ezc3d::DataNS::Vector3d::data() const {
    return _data;
}

std::vector<float> ezc3d::DataNS::Vector3d::data() {
    return _data;
}

void ezc3d::DataNS::Vector3d::set(
        float x,
        float y,
        float z)
{
    _data[0] = x;
    _data[1] = y;
    _data[2] = z;
}

float ezc3d::DataNS::Vector3d::x() const {
    return _data[0];
}

void ezc3d::DataNS::Vector3d::x(
        float x) {
    _data[0] = x;
}

float ezc3d::DataNS::Vector3d::y() const {
    return _data[1];
}

void ezc3d::DataNS::Vector3d::y(
        float y) {
    _data[1] = y;
}

float ezc3d::DataNS::Vector3d::z() const {
    return _data[2];
}

void ezc3d::DataNS::Vector3d::z(
        float z) {
    _data[2] = z;
}

#define EZC3D_API_EXPORTS
///
/// \file Vector3d.cpp
/// \brief Implementation of Vector3d class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "Vector3d.h"

ezc3d::Vector3d::Vector3d() :
    ezc3d::Matrix(3, 1)
{

}

ezc3d::Vector3d::Vector3d(
        const ezc3d::Vector3d &p) :
    ezc3d::Matrix(p)
{

}

void ezc3d::Vector3d::print() const {
    std::cout << " Vector = ["
              << x() << ", "
              << y() << ", "
              << z() << "];"
              << std::endl;
}


bool ezc3d::Vector3d::isValid() const
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

void ezc3d::Vector3d::set(
        float x,
        float y,
        float z)
{
    _data[0] = x;
    _data[1] = y;
    _data[2] = z;
}

float ezc3d::Vector3d::x() const {
    return _data[0];
}

void ezc3d::Vector3d::x(
        float x) {
    _data[0] = x;
}

float ezc3d::Vector3d::y() const {
    return _data[1];
}

void ezc3d::Vector3d::y(
        float y) {
    _data[1] = y;
}

float ezc3d::Vector3d::z() const {
    return _data[2];
}

void ezc3d::Vector3d::z(
        float z) {
    _data[2] = z;
}

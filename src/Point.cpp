#define EZC3D_API_EXPORTS
///
/// \file Point.cpp
/// \brief Implementation of Point class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "Point.h"

ezc3d::DataNS::Points3dNS::Point::Point()
{
    _data.resize(4);
}

ezc3d::DataNS::Points3dNS::Point::Point(
        const ezc3d::DataNS::Points3dNS::Point &p) {
    _data.resize(4);
    x(p.x());
    y(p.y());
    z(p.z());
    residual(p.residual());
}

void ezc3d::DataNS::Points3dNS::Point::print() const {
    std::cout << " Position = [" << x() << ", " << y() << ", " << z()
              << "]; Residual = " << residual() << std::endl;
}

void ezc3d::DataNS::Points3dNS::Point::write(std::fstream &f) const {
    if (residual() >= 0){
        f.write(reinterpret_cast<const char*>(&_data[0]), ezc3d::DATA_TYPE::FLOAT);
        f.write(reinterpret_cast<const char*>(&_data[1]), ezc3d::DATA_TYPE::FLOAT);
        f.write(reinterpret_cast<const char*>(&_data[2]), ezc3d::DATA_TYPE::FLOAT);
        f.write(reinterpret_cast<const char*>(&_data[3]), ezc3d::DATA_TYPE::FLOAT);
    }
    else {
        float zero(0);
        float minusOne(-1);
        f.write(reinterpret_cast<const char*>(&zero), ezc3d::DATA_TYPE::FLOAT);
        f.write(reinterpret_cast<const char*>(&zero), ezc3d::DATA_TYPE::FLOAT);
        f.write(reinterpret_cast<const char*>(&zero), ezc3d::DATA_TYPE::FLOAT);
        f.write(reinterpret_cast<const char*>(&minusOne), ezc3d::DATA_TYPE::FLOAT);
    }
}

const std::vector<float> ezc3d::DataNS::Points3dNS::Point::data() const {
    return _data;
}

std::vector<float> ezc3d::DataNS::Points3dNS::Point::data() {
    return _data;
}

void ezc3d::DataNS::Points3dNS::Point::set(
        float x,
        float y,
        float z,
        float residual)
{
    _data[0] = x;
    _data[1] = y;
    _data[2] = z;
    _data[3] = residual;
}

void ezc3d::DataNS::Points3dNS::Point::set(
        float x,
        float y,
        float z)
{
    _data[0] = x;
    _data[1] = y;
    _data[2] = z;
}

float ezc3d::DataNS::Points3dNS::Point::x() const {
    return _data[0];
}

void ezc3d::DataNS::Points3dNS::Point::x(
        float x) {
    _data[0] = x;
}

float ezc3d::DataNS::Points3dNS::Point::y() const {
    return _data[1];
}

void ezc3d::DataNS::Points3dNS::Point::y(
        float y) {
    _data[1] = y;
}

float ezc3d::DataNS::Points3dNS::Point::z() const {
    return _data[2];
}

void ezc3d::DataNS::Points3dNS::Point::z(
        float z) {
    _data[2] = z;
}


float ezc3d::DataNS::Points3dNS::Point::residual() const {
    return _data[3];
}

void ezc3d::DataNS::Points3dNS::Point::residual(
        float residual) {
    _data[3] = residual;
}

bool ezc3d::DataNS::Points3dNS::Point::isempty() const {
    if (static_cast<double>(x()) == 0.0 &&
            static_cast<double>(y()) == 0.0 &&
            static_cast<double>(z()) == 0.0 &&
            static_cast<double>(residual()) == 0.0)
        return true;
    else {
        return false;
    }
}

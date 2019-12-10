#define EZC3D_API_EXPORTS
///
/// \file Point.cpp
/// \brief Implementation of Point class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "Point.h"

ezc3d::DataNS::Points3dNS::Point::Point() :
    _residual(-1)
{
    _data.resize(3, 0);
    _cameraMasks.resize(7, false);
}

ezc3d::DataNS::Points3dNS::Point::Point(
        const ezc3d::DataNS::Points3dNS::Point &p) {
    _data.resize(3);
    x(p.x());
    y(p.y());
    z(p.z());
    residual(p.residual());
    _cameraMasks = p._cameraMasks;
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
        int residual(static_cast<int>(_residual));
f.write(reinterpret_cast<const char*>(&residual), ezc3d::DATA_TYPE::BYTE);
        f.write(reinterpret_cast<const char*>(&residual), ezc3d::DATA_TYPE::BYTE);
    }
    else {
        float zero(0);
        int minusOne(-1);
        f.write(reinterpret_cast<const char*>(&zero), ezc3d::DATA_TYPE::FLOAT);
        f.write(reinterpret_cast<const char*>(&zero), ezc3d::DATA_TYPE::FLOAT);
        f.write(reinterpret_cast<const char*>(&zero), ezc3d::DATA_TYPE::FLOAT);
        f.write(reinterpret_cast<const char*>(&zero), ezc3d::DATA_TYPE::BYTE);
        f.write(reinterpret_cast<const char*>(&minusOne), ezc3d::DATA_TYPE::BYTE);
    }
}

bool ezc3d::DataNS::Points3dNS::Point::isPointValid() const
{
    if (std::isnan(_data[0]) || std::isnan(_data[1]) || std::isnan(_data[2])) {
        return false;
    }
    else {
        return true;
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
    _residual = residual;
}

void ezc3d::DataNS::Points3dNS::Point::set(
        float x,
        float y,
        float z)
{
    _data[0] = x;
    _data[1] = y;
    _data[2] = z;
    if (isPointValid()) {
        residual(0);
    }
    else {
        residual(-1);
    }
}

float ezc3d::DataNS::Points3dNS::Point::x() const {
    return _data[0];
}

void ezc3d::DataNS::Points3dNS::Point::x(
        float x) {
    _data[0] = x;
    if (isPointValid()) {
        residual(0);
    }
    else {
        residual(-1);
    }
}

float ezc3d::DataNS::Points3dNS::Point::y() const {
    return _data[1];
}

void ezc3d::DataNS::Points3dNS::Point::y(
        float y) {
    _data[1] = y;
    if (isPointValid()) {
        residual(0);
    }
    else {
        residual(-1);
    }
}

float ezc3d::DataNS::Points3dNS::Point::z() const {
    return _data[2];
}

void ezc3d::DataNS::Points3dNS::Point::z(
        float z) {
    _data[2] = z;
    if (isPointValid()) {
        residual(0);
    }
    else {
        residual(-1);
    }
}


float ezc3d::DataNS::Points3dNS::Point::residual() const {
    return _residual;
}

void ezc3d::DataNS::Points3dNS::Point::residual(
        float residual) {
    _residual = residual;
}

const std::vector<bool>&
ezc3d::DataNS::Points3dNS::Point::cameraMask() const
{
    return _cameraMasks;
}

void ezc3d::DataNS::Points3dNS::Point::cameraMask(
        const std::vector<bool> &masks)
{
    _cameraMasks = masks;
}

void ezc3d::DataNS::Points3dNS::Point::cameraMask(int byte)
{
    if (byte != 0){
        for (size_t i=0; i<7; ++i) {
            _cameraMasks[i] = ((byte & ( 1 << i )) >> i);
        }
    }
}

bool ezc3d::DataNS::Points3dNS::Point::isempty() const {
    if (static_cast<double>(x()) == 0.0 &&
            static_cast<double>(y()) == 0.0 &&
            static_cast<double>(z()) == 0.0 &&
            static_cast<double>(residual()) < 0)
        return true;
    else {
        return false;
    }
}

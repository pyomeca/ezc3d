#define EZC3D_API_EXPORTS
///
/// \file Point.cpp
/// \brief Implementation of Point class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "Point.h"

#include <bitset>

ezc3d::DataNS::Points3dNS::Point::Point() :
    ezc3d::Vector3d(),
    _residual(-1)
{
    _cameraMasks.resize(7, false);
}

ezc3d::DataNS::Points3dNS::Point::Point(
        const ezc3d::DataNS::Points3dNS::Point &p) :
    ezc3d::Vector3d(p)
{
    residual(p.residual());
    _cameraMasks = p._cameraMasks;
}

void ezc3d::DataNS::Points3dNS::Point::print() const {
    ezc3d::Vector3d::print();
    std::cout << "Residual = " << residual() << "; Masks = [";
    for (size_t i = 0; i<_cameraMasks.size()-1; ++i){
        std::cout << _cameraMasks[i] << ", ";
    }
    if (_cameraMasks.size() > 0){
        std::cout << _cameraMasks[_cameraMasks.size()-1] << "]";
    }
    std::cout << std::endl;
}

void ezc3d::DataNS::Points3dNS::Point::write(
        std::fstream &f,
        float scaleFactor) const {
    if (residual() >= 0){
        for (size_t i = 0; i<size(); ++i) {
            float data(static_cast<float>(_data[i]));
            f.write(reinterpret_cast<const char*>(&data), ezc3d::DATA_TYPE::FLOAT);
        }
        std::bitset<8> cameraMasksBits;
        for (size_t i = 0; i < _cameraMasks.size(); ++i){
            if (_cameraMasks[i]){
                cameraMasksBits[i] = 1;
            }
            else {
                cameraMasksBits[i] = 0;
            }
        }
        cameraMasksBits[7] = 0;
        size_t cameraMasks(cameraMasksBits.to_ulong());
        f.write(reinterpret_cast<const char*>(&cameraMasks), ezc3d::DATA_TYPE::WORD);
        int residual(static_cast<int>(_residual / fabsf(scaleFactor)));
        f.write(reinterpret_cast<const char*>(&residual), ezc3d::DATA_TYPE::WORD);
    }
    else {
        float zero(0);
        int minusOne(-1);
        f.write(reinterpret_cast<const char*>(&zero), ezc3d::DATA_TYPE::FLOAT);
        f.write(reinterpret_cast<const char*>(&zero), ezc3d::DATA_TYPE::FLOAT);
        f.write(reinterpret_cast<const char*>(&zero), ezc3d::DATA_TYPE::FLOAT);
        f.write(reinterpret_cast<const char*>(&zero), ezc3d::DATA_TYPE::WORD);
        f.write(reinterpret_cast<const char*>(&minusOne), ezc3d::DATA_TYPE::WORD);
    }
}

void ezc3d::DataNS::Points3dNS::Point::set(
        double x,
        double y,
        double z,
        double residual)
{
    ezc3d::Vector3d::set(x, y, z);
    _residual = residual;
}

void ezc3d::DataNS::Points3dNS::Point::set(
        double x,
        double y,
        double z)
{
    ezc3d::Vector3d::set(x, y, z);
    if (!isValid() || (_data[0] == 0.0 && _data[1] == 0.0 && _data[2] == 0.0)) {
        residual(-1);
    }
    else {
        residual(0);
    }
}

double ezc3d::DataNS::Points3dNS::Point::x() const
{
    return ezc3d::Vector3d::x();
}

void ezc3d::DataNS::Points3dNS::Point::x(
        double x) {
    ezc3d::Vector3d::x(x);
    if (!isValid() || (_data[0] == 0.0 && _data[1] == 0.0 && _data[2] == 0.0)) {
        residual(-1);
    }
    else {
        residual(0);
    }
}

double ezc3d::DataNS::Points3dNS::Point::y() const
{
    return ezc3d::Vector3d::y();
}

void ezc3d::DataNS::Points3dNS::Point::y(
        double y) {
    ezc3d::Vector3d::y(y);
    if (!isValid() || (_data[0] == 0.0 && _data[1] == 0.0 && _data[2] == 0.0)) {
        residual(-1);
    }
    else {
        residual(0);
    }
}

double ezc3d::DataNS::Points3dNS::Point::z() const
{
    return ezc3d::Vector3d::z();
}

void ezc3d::DataNS::Points3dNS::Point::z(
        double z) {
    ezc3d::Vector3d::z(z);
    if (!isValid() || (_data[0] == 0.0 && _data[1] == 0.0 && _data[2] == 0.0)) {
        residual(-1);
    }
    else {
        residual(0);
    }
}

double ezc3d::DataNS::Points3dNS::Point::residual() const {
    return _residual;
}

void ezc3d::DataNS::Points3dNS::Point::residual(
        double residual) {
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
    for (size_t i=0; i<7; ++i) {
        _cameraMasks[i] = ((byte & ( 1 << i )) >> i);
    }
}

bool ezc3d::DataNS::Points3dNS::Point::isEmpty() const {
    if (!isValid() || (x() == 0.0 && y() == 0.0 && z() == 0.0
            && residual() < 0))
        return true;
    else {
        return false;
    }
}

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
        double x,
        double y,
        double z) :
    ezc3d::Matrix(3, 1)
{
    set(x, y, z);
}

ezc3d::Vector3d::Vector3d(
        const ezc3d::Matrix &p) :
    ezc3d::Matrix(p)
{
    if (nbRows() != 3 || nbCols() != 1){
        throw std::runtime_error("Size of the matrix must be 3x1 to be casted"
                                 "as a vector3d");
    }
}

void ezc3d::Vector3d::print() const
{
    std::cout << " Vector = ["
              << x() << ", "
              << y() << ", "
              << z() << "];"
              << std::endl;
}

void ezc3d::Vector3d::set(
        double x,
        double y,
        double z)
{
    _data[0] = x;
    _data[1] = y;
    _data[2] = z;
}

double ezc3d::Vector3d::x() const
{
    return _data[0];
}

void ezc3d::Vector3d::x(
        double x)
{
    _data[0] = x;
}

double ezc3d::Vector3d::y() const
{
    return _data[1];
}

void ezc3d::Vector3d::y(
        double y)
{
    _data[1] = y;
}

double ezc3d::Vector3d::z() const
{
    return _data[2];
}

void ezc3d::Vector3d::z(
        double z)
{
    _data[2] = z;
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

double ezc3d::Vector3d::operator()(
        size_t idx) const
{
    return this->ezc3d::Matrix::operator ()(idx, 0);
}

double& ezc3d::Vector3d::operator()(
        size_t idx)
{
    return this->ezc3d::Matrix::operator ()(idx, 0);
}

double ezc3d::Vector3d::dot(
        const ezc3d::Vector3d &other)
{
    return x()*other.x() + y()*other.y() + z()*other.z();
}

ezc3d::Vector3d ezc3d::Vector3d::cross(
        const ezc3d::Vector3d &other)
{
    return ezc3d::Vector3d(
                y()*other.z() - z()*other.y(),
                z()*other.x() - x()*other.z(),
                x()*other.y() - y()*other.x());
}

double ezc3d::Vector3d::norm()
{
    return sqrt(dot(*this));
}

void ezc3d::Vector3d::normalize()
{
    *this /= norm();
}

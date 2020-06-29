#define EZC3D_API_EXPORTS
///
/// \file Vector3d.cpp
/// \brief Implementation of Vector3d class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "math/Vector3d.h"

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
#ifndef USE_MATRIX_FAST_ACCESSOR
    if (nbRows() != 3 || nbCols() != 1){
        throw std::runtime_error("Size of the matrix must be 3x1 to be casted"
                                 "as a vector3d");
    }
#endif
}

void ezc3d::Vector3d::print() const
{
    std::cout << " Vector = ["
              << x() << ", "
              << y() << ", "
              << z() << "];"
              << std::endl;
}

void ezc3d::Vector3d::resize(
        size_t,
        size_t)
{
    throw std::runtime_error("Vector3d cannot be resized");
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

ezc3d::Vector3d& ezc3d::Vector3d::operator=(
        const ezc3d::Matrix& other)
{
    if (this != &other){
#ifndef USE_MATRIX_FAST_ACCESSOR
        if (other.nbRows() != 3 || other.nbCols() != 1){
            throw std::runtime_error("Size of the matrix must be 3x1 to be casted"
                                     "as a vector3d");
        }
#endif

        _data[0] = other._data[0];
        _data[1] = other._data[1];
        _data[2] = other._data[2];
    }
    return *this;
}

double ezc3d::Vector3d::operator()(
        size_t row) const
{
#ifndef USE_MATRIX_FAST_ACCESSOR
    if (row > 2){
        throw std::runtime_error("Maximal index for a vector3d is 2");
    }
#endif
    return _data[row];
}

double& ezc3d::Vector3d::operator()(
        size_t row)
{
#ifndef USE_MATRIX_FAST_ACCESSOR
    if (row > 2){
        throw std::runtime_error("Maximal index for a vector3d is 2");
    }
#endif
    return _data[row];
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

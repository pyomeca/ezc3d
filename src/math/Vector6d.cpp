#define EZC3D_API_EXPORTS
///
/// \file Vector6d.cpp
/// \brief Implementation of Vector6d class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "math/Vector6d.h"

ezc3d::Vector6d::Vector6d() :
    ezc3d::Matrix(6, 1)
{

}

ezc3d::Vector6d::Vector6d(
        double e0,
        double e1,
        double e2,
        double e3,
        double e4,
        double e5) :
    ezc3d::Matrix(6, 1)
{
    _data[0] = e0;
    _data[1] = e1;
    _data[2] = e2;
    _data[3] = e3;
    _data[4] = e4;
    _data[5] = e5;
}

ezc3d::Vector6d::Vector6d(
        const ezc3d::Matrix &p) :
    ezc3d::Matrix(p)
{
#ifndef USE_MATRIX_FAST_ACCESSOR
    if (nbRows() != 6 || nbCols() != 1){
        throw std::runtime_error("Size of the matrix must be 6x1 to be casted"
                                 "as a Vector6d");
    }
#endif
}

void ezc3d::Vector6d::print() const
{
    std::cout << " Vector = ["
              << _data[0] << ", "
              << _data[1] << ", "
              << _data[2] << ", "
              << _data[3] << ", "
              << _data[4] << ", "
              << _data[5] << "];"
              << std::endl;
}

ezc3d::Vector6d& ezc3d::Vector6d::operator=(
        const ezc3d::Matrix& other)
{
    if (this != &other){
#ifndef USE_MATRIX_FAST_ACCESSOR
        if (other.nbRows() != 6 || other.nbCols() != 1){
            throw std::runtime_error("Size of the matrix must be 6x1 to be casted"
                                     "as a Vector6d");
        }
#endif

        _data[0] = other._data[0];
        _data[1] = other._data[1];
        _data[2] = other._data[2];
        _data[3] = other._data[3];
        _data[4] = other._data[4];
        _data[5] = other._data[5];
    }
    return *this;
}

void ezc3d::Vector6d::resize(
        size_t,
        size_t)
{
    throw std::runtime_error("Vector6d cannot be resized");
}

double ezc3d::Vector6d::operator()(
        size_t row) const
{
#ifndef USE_MATRIX_FAST_ACCESSOR
    if (row > 5){
        throw std::runtime_error("Maximal index for a Vector6d is 5");
    }
#endif
    return _data[row];
}

double& ezc3d::Vector6d::operator()(
        size_t row)
{
#ifndef USE_MATRIX_FAST_ACCESSOR
    if (row > 5){
        throw std::runtime_error("Maximal index for a Vector6d is 5");
    }
#endif
    return _data[row];
}

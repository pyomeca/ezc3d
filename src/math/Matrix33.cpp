#define EZC3D_API_EXPORTS
///
/// \file Matrix33.cpp
/// \brief Implementation of Matrix33 class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "math/Matrix33.h"

#include "math/Vector3d.h"

ezc3d::Matrix33::Matrix33() :
    ezc3d::Matrix(3, 3)
{

}

ezc3d::Matrix33::Matrix33(
        double elem00, double elem01, double elem02,
        double elem10, double elem11, double elem12,
        double elem20, double elem21, double elem22) :
    ezc3d::Matrix(3, 3)
{
    _data[0] = elem00;
    _data[1] = elem10;
    _data[2] = elem20;
    _data[3] = elem01;
    _data[4] = elem11;
    _data[5] = elem21;
    _data[6] = elem02;
    _data[7] = elem12;
    _data[8] = elem22;
}

ezc3d::Matrix33::Matrix33(
        const ezc3d::Matrix &other) :
    ezc3d::Matrix(other)
{
#ifndef USE_MATRIX_FAST_ACCESSOR
    if (_nbRows != 3 || _nbCols != 3){
        throw std::runtime_error("Size of the matrix must be 3x3 to be casted"
                                 "as a Matrix33");
    }
#endif
}

size_t ezc3d::Matrix33::size() const
{
    return 9;
}

size_t ezc3d::Matrix33::nbRows() const
{
    return 3;
}

size_t ezc3d::Matrix33::nbCols() const
{
    return 3;
}

void ezc3d::Matrix33::resize(
        size_t,
        size_t)
{
    throw std::runtime_error("Matrix33 cannot be resized");
}

ezc3d::Vector3d ezc3d::Matrix33::operator*(
        const ezc3d::Vector3d& other)
{
    return ezc3d::Vector3d(
            _data[0] * other._data[0] + _data[3] * other._data[1] + _data[6] * other._data[2],
            _data[1] * other._data[0] + _data[4] * other._data[1] + _data[7] * other._data[2],
            _data[2] * other._data[0] + _data[5] * other._data[1] + _data[8] * other._data[2]
            );
}

ezc3d::Matrix33 ezc3d::Matrix33::operator*(
        const ezc3d::Matrix33& other)
{
    return ezc3d::Matrix33(
            _data[0] * other._data[0] + _data[3] * other._data[1] + _data[6] * other._data[2],
            _data[0] * other._data[3] + _data[3] * other._data[4] + _data[6] * other._data[5],
            _data[0] * other._data[6] + _data[3] * other._data[7] + _data[6] * other._data[8],
            _data[1] * other._data[0] + _data[4] * other._data[1] + _data[7] * other._data[2],
            _data[1] * other._data[3] + _data[4] * other._data[4] + _data[7] * other._data[5],
            _data[1] * other._data[6] + _data[4] * other._data[7] + _data[7] * other._data[8],
            _data[2] * other._data[0] + _data[5] * other._data[1] + _data[8] * other._data[2],
            _data[2] * other._data[3] + _data[5] * other._data[4] + _data[8] * other._data[5],
            _data[2] * other._data[6] + _data[5] * other._data[7] + _data[8] * other._data[8]
            );
}

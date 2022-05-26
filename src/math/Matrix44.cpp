#define EZC3D_API_EXPORTS
///
/// \file Matrix44.cpp
/// \brief Implementation of Matrix44 class
/// \author Pariterre
/// \version 1.0
/// \date April 30th, 2022
///

#include "ezc3d/math/Matrix44.h"
#include "ezc3d/math/Vector3d.h"
#include <stdexcept>

ezc3d::Matrix44::Matrix44() :
    ezc3d::Matrix(4, 4)
{

}

ezc3d::Matrix44::Matrix44(
        double elem00, double elem01, double elem02, double elem03,
        double elem10, double elem11, double elem12, double elem13,
        double elem20, double elem21, double elem22, double elem23,
        double elem30, double elem31, double elem32, double elem33) :
    ezc3d::Matrix(4, 4)
{
    _data[0]  = elem00;
    _data[1]  = elem10;
    _data[2]  = elem20;
    _data[3]  = elem30;
    _data[4]  = elem01;
    _data[5]  = elem11;
    _data[6]  = elem21;
    _data[7]  = elem31;
    _data[8]  = elem02;
    _data[9]  = elem12;
    _data[10] = elem22;
    _data[11] = elem32;
    _data[12] = elem03;
    _data[13] = elem13;
    _data[14] = elem23;
    _data[15] = elem33;
}

ezc3d::Matrix44::Matrix44(
        const ezc3d::Matrix &other) :
    ezc3d::Matrix(other)
{
#ifndef USE_MATRIX_FAST_ACCESSOR
    if (_nbRows != 4 || _nbCols != 4){
        throw std::runtime_error("Size of the matrix must be 4x4 to be casted"
                                 "as a Matrix44");
    }
#endif
}

size_t ezc3d::Matrix44::size() const
{
    return 16;
}

size_t ezc3d::Matrix44::nbRows() const
{
    return 4;
}

size_t ezc3d::Matrix44::nbCols() const
{
    return 4;
}

void ezc3d::Matrix44::resize(
        size_t,
        size_t)
{
    throw std::runtime_error("Matrix44 cannot be resized");
}

void ezc3d::Matrix44::set(
        double elem00, double elem01, double elem02, double elem03,
        double elem10, double elem11, double elem12, double elem13,
        double elem20, double elem21, double elem22, double elem23,
        double elem30, double elem31, double elem32, double elem33)
{
    _data[0]  = elem00;
    _data[1]  = elem10;
    _data[2]  = elem20;
    _data[3]  = elem30;
    _data[4]  = elem01;
    _data[5]  = elem11;
    _data[6]  = elem21;
    _data[7]  = elem31;
    _data[8]  = elem02;
    _data[9]  = elem12;
    _data[10] = elem22;
    _data[11] = elem32;
    _data[12] = elem03;
    _data[13] = elem13;
    _data[14] = elem23;
    _data[15] = elem33;
}

ezc3d::Vector3d ezc3d::Matrix44::operator*(
        const ezc3d::Vector3d& other)
{
    return ezc3d::Vector3d(
            _data[0] * other._data[0] + _data[4] * other._data[1] + _data[8]  * other._data[2] + _data[12],
            _data[1] * other._data[0] + _data[5] * other._data[1] + _data[9]  * other._data[2] + _data[13],
            _data[2] * other._data[0] + _data[6] * other._data[1] + _data[10] * other._data[2] + _data[14]
        );
}

ezc3d::Matrix44 ezc3d::Matrix44::operator*(
        const ezc3d::Matrix44& other)
{
    return ezc3d::Matrix44(
            _data[0] * other._data[0]  + _data[4] * other._data[1]  + _data[8]  * other._data[2]  + _data[12] * other._data[3],
            _data[0] * other._data[4]  + _data[4] * other._data[5]  + _data[8]  * other._data[6]  + _data[12] * other._data[7],
            _data[0] * other._data[8]  + _data[4] * other._data[9]  + _data[8]  * other._data[10] + _data[12] * other._data[11],
            _data[0] * other._data[12] + _data[4] * other._data[13] + _data[8]  * other._data[14] + _data[12] * other._data[15],

            _data[1] * other._data[0]  + _data[5] * other._data[1]  + _data[9]  * other._data[2]  + _data[13] * other._data[3],
            _data[1] * other._data[4]  + _data[5] * other._data[5]  + _data[9]  * other._data[6]  + _data[13] * other._data[7],
            _data[1] * other._data[8]  + _data[5] * other._data[9]  + _data[9]  * other._data[10] + _data[13] * other._data[11],
            _data[1] * other._data[12] + _data[5] * other._data[13] + _data[9]  * other._data[14] + _data[13] * other._data[15],

            _data[2] * other._data[0]  + _data[6] * other._data[1]  + _data[10] * other._data[2]  + _data[14] * other._data[3],
            _data[2] * other._data[4]  + _data[6] * other._data[5]  + _data[10] * other._data[6]  + _data[14] * other._data[7],
            _data[2] * other._data[8]  + _data[6] * other._data[9]  + _data[10] * other._data[10] + _data[14] * other._data[11],
            _data[2] * other._data[12] + _data[6] * other._data[13] + _data[10] * other._data[14] + _data[14] * other._data[15],

            _data[3] * other._data[0]  + _data[7] * other._data[1]  + _data[11] * other._data[2]  + _data[15] * other._data[3],
            _data[3] * other._data[4]  + _data[7] * other._data[5]  + _data[11] * other._data[6]  + _data[15] * other._data[7],
            _data[3] * other._data[8]  + _data[7] * other._data[9]  + _data[11] * other._data[10] + _data[15] * other._data[11],
            _data[3] * other._data[12] + _data[7] * other._data[13] + _data[11] * other._data[14] + _data[15] * other._data[15]
            );
}

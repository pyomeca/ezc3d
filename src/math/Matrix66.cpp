#define EZC3D_API_EXPORTS
///
/// \file Matrix66.cpp
/// \brief Implementation of Matrix66 class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "math/Matrix66.h"

#include "math/Vector6d.h"

ezc3d::Matrix66::Matrix66() :
    ezc3d::Matrix(6, 6)
{

}

ezc3d::Matrix66::Matrix66(
        const ezc3d::Matrix &other) :
    ezc3d::Matrix(other)
{
#ifndef USE_MATRIX_FAST_ACCESSOR
    if (_nbRows != 6 || _nbCols != 6){
        throw std::runtime_error("Size of the matrix must be 6x6 to be casted"
                                 "as a Matrix66");
    }
#endif
}

size_t ezc3d::Matrix66::size() const
{
    return 36;
}

size_t ezc3d::Matrix66::nbRows() const
{
    return 6;
}

size_t ezc3d::Matrix66::nbCols() const
{
    return 6;
}

void ezc3d::Matrix66::resize(
        size_t,
        size_t)
{
    throw std::runtime_error("Matrix66 cannot be resized");
}

ezc3d::Vector6d ezc3d::Matrix66::operator*(
        const ezc3d::Vector6d& other)
{
    return ezc3d::Vector6d(
            _data[0] * other._data[0] + _data[6] * other._data[1] + _data[12] * other._data[2] + _data[18] * other._data[3] + _data[24] * other._data[4] + _data[30] * other._data[5],
            _data[1] * other._data[0] + _data[7] * other._data[1] + _data[13] * other._data[2] + _data[19] * other._data[3] + _data[25] * other._data[4] + _data[31] * other._data[5],
            _data[2] * other._data[0] + _data[8] * other._data[1] + _data[14] * other._data[2] + _data[20] * other._data[3] + _data[26] * other._data[4] + _data[32] * other._data[5],
            _data[3] * other._data[0] + _data[9] * other._data[1] + _data[15] * other._data[2] + _data[21] * other._data[3] + _data[27] * other._data[4] + _data[33] * other._data[5],
            _data[4] * other._data[0] + _data[10]* other._data[1] + _data[16] * other._data[2] + _data[22] * other._data[3] + _data[28] * other._data[4] + _data[34] * other._data[5],
            _data[5] * other._data[0] + _data[11]* other._data[1] + _data[17] * other._data[2] + _data[23] * other._data[3] + _data[29] * other._data[4] + _data[35] * other._data[5]
            );
}

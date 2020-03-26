#define EZC3D_API_EXPORTS
///
/// \file Matrix.cpp
/// \brief Implementation of Vector3d class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "Matrix.h"

ezc3d::Matrix::Matrix(
        size_t nbRows,
        size_t nbCols) :
    _nbRows(nbRows),
    _nbCols(nbCols),
    _data(std::vector<double>(_nbRows * _nbCols))
{

}

ezc3d::Matrix::Matrix(
        const ezc3d::Matrix &matrix) :
    _nbRows(matrix._nbRows),
    _nbCols(matrix._nbCols),
    _data(matrix._data)
{

}

void ezc3d::Matrix::print() const
{
    std::cout << " Matrix = [" << std::endl;
    for (size_t i=0; i<_nbRows; ++i){
        for (size_t j=0; j<_nbCols; ++j){
            std::cout << operator ()(i, j);
            if (j != _nbCols-1){
                std::cout << ", ";
            }
        }
        if (i == _nbRows-1){
            std::cout << "]";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

const std::vector<double>& ezc3d::Matrix::data() const
{
    return _data;
}

std::vector<double>& ezc3d::Matrix::data()
{
    return _data;
}

size_t ezc3d::Matrix::nbRows() const
{
    return _nbRows;
}

size_t ezc3d::Matrix::nbCols() const
{
    return _nbCols;
}

ezc3d::Matrix ezc3d::Matrix::T()
{
    ezc3d::Matrix result(nbCols(), nbRows());
    for (size_t i=0; i<nbRows(); ++i){
        for (size_t j=0; j<nbCols(); ++j){
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

ezc3d::Matrix ezc3d::Matrix::operator+(
        const ezc3d::Matrix &other)
{
    ezc3d::Matrix result(nbRows(), nbCols());
    for (size_t i=0; i<nbRows(); ++i){
        for (size_t j=0; j<nbCols(); ++j){
            result(i, j) = (*this)(i, j) + other(i, j);
        }
    }
    return result;
}

ezc3d::Matrix ezc3d::Matrix::operator*(
        const ezc3d::Matrix &other)
{
    ezc3d::Matrix result(nbRows(), other.nbCols());
    for (size_t i=0; i<nbRows(); ++i){
        for (size_t j=0; j<other.nbCols(); ++j){
            for (size_t k=0; k<other.nbRows(); ++k){
                result(i, j) += (*this)(i, k) * other(k, j);
            }
        }
    }
    return result;
}

double ezc3d::Matrix::operator()(
        size_t row,
        size_t col) const
{
    // The data are arrange column majors
    return _data[col*_nbRows + row];
}

double &ezc3d::Matrix::operator()(
        size_t row,
        size_t col)
{
    // The data are arrange column majors
    return _data[col*_nbRows + row];
}


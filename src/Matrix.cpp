#define EZC3D_API_EXPORTS
///
/// \file Matrix.cpp
/// \brief Implementation of Vector3d class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "Matrix.h"

ezc3d::Matrix::Matrix():
    _nbRows(0),
    _nbCols(0),
    _data(std::vector<double>(_nbRows * _nbCols))
{

}

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

void ezc3d::Matrix::resize(
        size_t nbRows,
        size_t nbCols)
{
    _nbRows = nbRows;
    _nbCols = nbCols;
    _data.resize(nbRows * nbCols);
}

double ezc3d::Matrix::operator()(
        size_t row,
        size_t col) const
{
    // The data are arrange column majors
    return _data.at(col*_nbRows + row);
}

double& ezc3d::Matrix::operator()(
        size_t row,
        size_t col)
{
    // The data are arrange column majors
    return _data.at(col*_nbRows + row);
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
        double scalar)
{
    ezc3d::Matrix result(*this);
    return result += scalar;
}

ezc3d::Matrix& ezc3d::Matrix::operator+=(
        double scalar)
{
    for (size_t i=0; i<nbRows(); ++i){
        for (size_t j=0; j<nbCols(); ++j){
            (*this)(i, j) += scalar;
        }
    }
    return *this;
}

ezc3d::Matrix ezc3d::Matrix::operator+(
        const ezc3d::Matrix &other)
{
    ezc3d::Matrix result(*this);
    return result += other;
}

ezc3d::Matrix& ezc3d::Matrix::operator+=(
        const ezc3d::Matrix &other)
{
    if (nbRows() != other.nbRows() || nbCols() != other.nbCols()){
        throw std::runtime_error(
            "Dimensions of matrices don't agree: \nFirst matrix dimensions = "
            + std::to_string(nbRows()) + "x" + std::to_string(nbRows()) + "\n"
            "Second matrix dimensions = "
            + std::to_string(other.nbRows()) + "x" + std::to_string(other.nbRows()));
    }

    for (size_t i=0; i<nbRows(); ++i){
        for (size_t j=0; j<nbCols(); ++j){
            (*this)(i, j) += other(i, j);
        }
    }
    return *this;
}

ezc3d::Matrix ezc3d::Matrix::operator-(
        double scalar)
{
    ezc3d::Matrix result(*this);
    return result -= scalar;
}

ezc3d::Matrix& ezc3d::Matrix::operator-=(
        double scalar)
{
    for (size_t i=0; i<nbRows(); ++i){
        for (size_t j=0; j<nbCols(); ++j){
            (*this)(i, j) -= scalar;
        }
    }
    return *this;
}

ezc3d::Matrix ezc3d::Matrix::operator-(
        const ezc3d::Matrix &other)
{
    ezc3d::Matrix result(*this);
    return result -= other;
}

ezc3d::Matrix& ezc3d::Matrix::operator-=(
        const ezc3d::Matrix &other)
{
    if (nbRows() != other.nbRows() || nbCols() != other.nbCols()){
        throw std::runtime_error(
            "Dimensions of matrices don't agree: \nFirst matrix dimensions = "
            + std::to_string(nbRows()) + "x" + std::to_string(nbRows()) + "\n"
            "Second matrix dimensions = "
            + std::to_string(other.nbRows()) + "x" + std::to_string(other.nbRows()));
    }

    for (size_t i=0; i<nbRows(); ++i){
        for (size_t j=0; j<nbCols(); ++j){
            (*this)(i, j) -= other(i, j);
        }
    }
    return *this;
}

ezc3d::Matrix ezc3d::Matrix::operator*(
        double scalar)
{
    ezc3d::Matrix result(*this);
    return result *= scalar;
}

ezc3d::Matrix& ezc3d::Matrix::operator*=(
        double scalar)
{
    for (size_t i=0; i<nbRows(); ++i){
        for (size_t j=0; j<nbCols(); ++j){
            (*this)(i, j) *= scalar;
        }
    }
    return *this;
}

ezc3d::Matrix ezc3d::Matrix::operator*(
        const ezc3d::Matrix &other)
{
    if (nbCols() != other.nbRows()){
        throw std::runtime_error(
            "Dimensions of matrices don't agree: \nFirst matrix dimensions = "
            + std::to_string(nbRows()) + "x" + std::to_string(nbRows()) + "\n"
            "Second matrix dimensions = "
            + std::to_string(other.nbRows()) + "x" + std::to_string(other.nbRows()));
    }

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

ezc3d::Matrix ezc3d::Matrix::operator/(
        double scalar)
{
    return *this * (1./scalar);
}

ezc3d::Matrix& ezc3d::Matrix::operator/=(
        double scalar)
{
    *this *= 1./scalar;
    return *this;
}

ezc3d::Matrix operator+(
        double scalar,
        ezc3d::Matrix mat)
{
    return mat + scalar;
}

ezc3d::Matrix operator-(
        double scalar,
        ezc3d::Matrix mat)
{
    return -1.0*mat + scalar;
}

ezc3d::Matrix operator*(
        double scalar,
        ezc3d::Matrix mat)
{
    return mat * scalar;
}


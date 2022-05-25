#define EZC3D_API_EXPORTS
///
/// \file Matrix.cpp
/// \brief Implementation of Matrix class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "ezc3d/math/Matrix.h"
#include "ezc3d/math/Vector3d.h"
#include "ezc3d/math/Vector6d.h"
#include <iostream>
#include <stdexcept>

ezc3d::Matrix::Matrix():
    _nbRows(0),
    _nbCols(0),
    _data(std::vector<double>())
{

}

ezc3d::Matrix::Matrix(
        size_t nbRows,
        size_t nbCols) :
    _nbRows(nbRows),
    _nbCols(nbCols),
    _data(std::vector<double>(nbRows * nbCols))
{

}

ezc3d::Matrix::Matrix(
        const ezc3d::Matrix &matrix) :
    _nbRows(matrix._nbRows),
    _nbCols(matrix._nbCols),
    _data(matrix._data)
{

}

ezc3d::Matrix::Matrix(
        const std::vector<Vector3d>& matrix) :
    _nbRows(3),
    _nbCols(matrix.size()),
    _data(std::vector<double>(_nbRows * _nbCols))
{
    for (size_t i=0; i<_nbCols; ++i){
        for (size_t j=0; j<_nbRows; ++j){
            _data[i*_nbRows + j] = matrix[i](j);
        }
    }
}

ezc3d::Matrix::Matrix(
        const std::vector<Vector6d>& matrix) :
    _nbRows(6),
    _nbCols(matrix.size()),
    _data(std::vector<double>(_nbRows * _nbCols))
{
    for (size_t i=0; i<_nbCols; ++i){
        for (size_t j=0; j<_nbRows; ++j){
            _data[i*_nbRows + j] = matrix[i](j);
        }
    }
}

void ezc3d::Matrix::print() const
{
    std::cout << " Matrix = [" << "\n";
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
        std::cout << "\n";
    }
    std::cout << "\n";
}

double ezc3d::Matrix::sum() const
{
    double sum(0);
    for (size_t i = 0; i < _data.size(); ++i)
        sum += _data[i];
    return sum;
}

void ezc3d::Matrix::setZeros()
{
    for (size_t i=0; i<nbRows(); ++i){
        for (size_t j=0; j<nbCols(); ++j){
            (*this)(i, j) = 0.0;
        }
    }
}

void ezc3d::Matrix::setOnes()
{
    for (size_t i=0; i<nbRows(); ++i){
        for (size_t j=0; j<nbCols(); ++j){
            (*this)(i, j) = 1.0;
        }
    }
}

void ezc3d::Matrix::setIdentity()
{
    for (size_t i=0; i<nbRows(); ++i){
        for (size_t j=0; j<nbCols(); ++j){
            if (i == j){
                (*this)(i, j) = 1.0;
            }
            else {
                (*this)(i, j) = 0.0;
            }
        }
    }
}

size_t ezc3d::Matrix::size() const
{
    return _data.size();
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
    _data.resize(_nbRows * _nbCols);
}

double ezc3d::Matrix::operator()(
        size_t row,
        size_t col) const
{
#ifndef USE_MATRIX_FAST_ACCESSOR
    // The data are arrange column majors
    if (nbRows() <= row || nbCols() <= col){
        throw std::runtime_error("Element ouside of the matrix bounds.\n"
                                 "Elements ask = "
                                 + std::to_string(row) + "x" + std::to_string(col)
                                 + "\nMatrix dimension = "
                                 + std::to_string(nbRows()) + "x"
                                 + std::to_string(nbCols()));
    }
#endif
    return _data[col*_nbRows + row];
}

double& ezc3d::Matrix::operator()(
        size_t row,
        size_t col)
{
#ifndef USE_MATRIX_FAST_ACCESSOR
    // The data are arrange column majors
    if (nbRows() <= row || nbCols() <= col){
        throw std::runtime_error("Element ouside of the matrix bounds.\n"
                                 "Elements ask = "
                                 + std::to_string(row) + "x" + std::to_string(col)
                                 + "\nMatrix dimension = "
                                 + std::to_string(nbRows()) + "x"
                                 + std::to_string(nbCols()));
    }
#endif
    return _data[col*_nbRows + row];
}

ezc3d::Matrix ezc3d::Matrix::T() const
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
#ifndef USE_MATRIX_FAST_ACCESSOR
    if (nbRows() != other.nbRows() || nbCols() != other.nbCols()){
        throw std::runtime_error(
            "Dimensions of matrices don't agree: \nFirst matrix dimensions = "
            + std::to_string(nbRows()) + "x" + std::to_string(nbRows()) + "\n"
            "Second matrix dimensions = "
            + std::to_string(other.nbRows()) + "x" + std::to_string(other.nbRows()));
    }
#endif

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
#ifndef USE_MATRIX_FAST_ACCESSOR
    if (nbRows() != other.nbRows() || nbCols() != other.nbCols()){
        throw std::runtime_error(
            "Dimensions of matrices don't agree: \nFirst matrix dimensions = "
            + std::to_string(nbRows()) + "x" + std::to_string(nbRows()) + "\n"
            "Second matrix dimensions = "
            + std::to_string(other.nbRows()) + "x" + std::to_string(other.nbRows()));
    }
#endif

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
#ifndef USE_MATRIX_FAST_ACCESSOR
    if (nbCols() != other.nbRows()){
        throw std::runtime_error(
            "Dimensions of matrices don't agree: \nFirst matrix dimensions = "
            + std::to_string(nbRows()) + "x" + std::to_string(nbRows()) + "\n"
            "Second matrix dimensions = "
            + std::to_string(other.nbRows()) + "x" + std::to_string(other.nbRows()));
    }
#endif

    ezc3d::Matrix result(nbRows(), other.nbCols());
    result.setZeros();
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

std::ostream &operator<<(
        std::ostream &out,
        const ezc3d::Matrix &m)
{
    out << "[";
    for(size_t i = 0; i < m.nbRows(); ++i) {
        for(size_t j = 0; j < m.nbCols(); ++j) {
            if (i !=0 && j ==0){
                out << " ";
            }
            out << m(i, j);
            if (j < m.nbCols() - 1){
                out << ", ";
            }
        }
        if (i < m.nbRows() - 1){
            out << "\n";
        }
    }
    out << "]";
    return out;
}

#ifndef EZC3D_MATH_MATRIX66_H
#define EZC3D_MATH_MATRIX66_H
///
/// \file Matrix66.h
/// \brief Declaration of a Matrix66 class
/// \author Pariterre
/// \version 1.0
/// \date March 25th, 2020
///

#include "Matrix.h"

///
/// \brief Matrix of unknown dimension
///
class EZC3D_API ezc3d::Matrix66 : public ezc3d::Matrix {
    //---- CONSTRUCTORS ----//
public:
    ///
    /// \brief Create an 6x6 Matrix with the memory allocated
    ///
    Matrix66();

    ///
    /// \brief Copy a matrix into a Matrix66
    /// \param other The matrix to copy
    ///
    Matrix66(
            const ezc3d::Matrix &other);

    //---- DATA ----//
public:
    ///
    /// \brief Return the number of element in the matrix (nbRows * nbCols = 36)
    /// \return The number of element in the matrix (nbRows * nbCols = 36)
    ///
    size_t size() const override;

    ///
    /// \brief Return the number of rows (6)
    /// \return The number of rows (6)
    ///
    size_t nbRows() const override;

    ///
    /// \brief Return the number of columns (6)
    /// \return The number of columns (6)
    ///
    size_t nbCols() const override;

    ///
    /// \brief Do nothing, it is not possible to resize a 3x3 matrix
    ///
    void resize(
            size_t,
            size_t) override;

    //---- OPERATIONS ----//
public:

    ///
    /// \brief Defining matrix multiplication with a Vector6d
    /// \param other The vector to multiply with
    /// \return The vector multiplied
    ///
    virtual ezc3d::Vector6d operator*(
            const ezc3d::Vector6d& other);

};

#endif

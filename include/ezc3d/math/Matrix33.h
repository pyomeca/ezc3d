#ifndef EZC3D_MATH_MATRIX33_H
#define EZC3D_MATH_MATRIX33_H
///
/// \file Matrix33.h
/// \brief Declaration of a Matrix33 class
/// \author Pariterre
/// \version 1.0
/// \date March 25th, 2020
///

#include "Matrix.h"

///
/// \brief Matrix of unknown dimension
///
class EZC3D_API ezc3d::Matrix33 : public ezc3d::Matrix {
    //---- CONSTRUCTORS ----//
public:
    ///
    /// \brief Create an 3x3 Matrix with the memory allocated
    ///
    Matrix33();

    ///
    /// \brief Create an 3x3 Matrix initalized with the 9 elements
    /// \param elem00 first col, first row
    /// \param elem01 first col, second row
    /// \param elem02 first col, third row
    /// \param elem10 second col, first row
    /// \param elem11 second col, second row
    /// \param elem12 second col, third row
    /// \param elem20 third col, first row
    /// \param elem21 third col, second row
    /// \param elem22 third col, third row
    ///
    Matrix33(
            double elem00, double elem01, double elem02,
            double elem10, double elem11, double elem12,
            double elem20, double elem21, double elem22);

    ///
    /// \brief Copy a matrix into a Matrix33
    /// \param other The matrix to copy
    ///
    Matrix33(
            const ezc3d::Matrix &other);

    //---- DATA ----//
public:
    ///
    /// \brief Return the number of element in the matrix (nbRows * nbCols = 9)
    /// \return The number of element in the matrix (nbRows * nbCols = 9)
    ///
    size_t size() const override;

    ///
    /// \brief Return the number of rows (3)
    /// \return The number of rows (3)
    ///
    size_t nbRows() const override;

    ///
    /// \brief Return the number of columns (3)
    /// \return The number of columns (3)
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
    /// \brief Defining matrix multiplication with a Vector3d
    /// \param other The vector to multiply with
    /// \return The vector multiplied
    ///
    virtual ezc3d::Vector3d operator*(
            const ezc3d::Vector3d& other);

    ///
    /// \brief Defining matrix multiplication with a Matrix33
    /// \param other The matrix to multiply with
    /// \return The matrix multiplied
    ///
    virtual ezc3d::Matrix33 operator*(
            const ezc3d::Matrix33& other);

};

#endif

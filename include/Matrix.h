#ifndef MATRIX_H
#define MATRIX_H
///
/// \file Matrix.h
/// \brief Declaration of a Matrix class
/// \author Pariterre
/// \version 1.0
/// \date March 25th, 2020
///

#include <ezc3d.h>

///
/// \brief Matrix of unknown dimension
///
class EZC3D_API ezc3d::Matrix {
    //---- CONSTRUCTORS ----//
public:
    ///
    /// \brief Create an empty Matrix with the memory allocated
    /// \param nbRows The number of rows
    /// \param nbCols The number of columns
    ///
    Matrix();

    ///
    /// \brief Create an empty Matrix with the memory allocated
    /// \param nbRows The number of rows
    /// \param nbCols The number of columns
    ///
    Matrix(
            size_t nbRows,
            size_t nbCols);

    ///
    /// \brief Copy a Matrix
    /// \param vector The matrix to copy
    ///
    Matrix(
            const ezc3d::Matrix& matrix);

    //---- STREAM ----//
public:
    ///
    ///
    /// \brief Print the Matrix
    ///
    /// Print the values of the matrix to the console
    ///
    virtual void print() const;


    //---- DATA ----//
protected:
    size_t _nbRows; ///< Number of rows
    size_t _nbCols; ///< Number of columns
    std::vector<double> _data; ///< Value of the Matrix

public:
    ///
    /// \brief Get a reference to the STL vector where the matrix is store
    /// \return The matrix
    ///
    virtual const std::vector<double>& data() const;

    ///
    /// \brief Get a reference to the STL vector where the matrix is store in order to be modified by the caller
    /// \return The matrix
    ///
    /// Get a reference to the STL vector where the matrix is store in the form of a non-const reference.
    /// The user can thereafter modify the coordinates at will, but with the caution it requires.
    ///
    virtual std::vector<double>& data();

    ///
    /// \brief Set all values to zero
    ///
    void setZeros();

    ///
    /// \brief Set all values to one
    ///
    void setOnes();

    ///
    /// \brief Set the matrix to an identity matrix
    ///
    void setIdentity();

    ///
    /// \brief Return the number of rows
    /// \return The number of rows
    ///
    size_t nbRows() const;

    ///
    /// \brief Return the number of columns
    /// \return The number of columns
    ///
    size_t nbCols() const;

    ///
    /// \brief Change the size of the matrix
    /// \param nbRows The new number of rows
    /// \param nbCols The new number of columns
    ///
    /// Changes the size of the matrix.
    /// Warning, there is no garanty for the position of the data in the
    /// matrix after the resize
    ///
    void resize(
            size_t nbRows,
            size_t nbCols);

#ifndef SWIG
    ///
    /// \brief Get a specific value of the matrix
    /// \param row The row index
    /// \param col The column index
    ///
    double operator()(
            size_t row,
            size_t col) const;
#endif

    ///
    /// \brief Get a reference to a specific value of the matrix
    /// \param row The row index
    /// \param col The column index
    ///
    double& operator()(
            size_t row,
            size_t col);

    //---- OPERATIONS ----//
public:
    ///
    /// \brief Defining matrix transpose
    /// \return The matrix transposed
    ///
    ezc3d::Matrix T();

    ///
    /// \brief Defining the addition with a scalar
    /// \param scalar The scalar to add with
    /// \return The matrix added
    ///
    virtual ezc3d::Matrix operator+(
            double scalar);

    ///
    /// \brief Defining the addition with a scalar
    /// \param scalar The scalar to add with
    /// \return The matrix added
    ///
    virtual ezc3d::Matrix& operator+=(
            double scalar);

    ///
    /// \brief Defining matrix addition
    /// \param other The matrix to add with
    /// \return The matrix added
    ///
    virtual ezc3d::Matrix operator+(
            const ezc3d::Matrix& other);

    ///
    /// \brief Defining matrix addition
    /// \param other The matrix to add with
    /// \return The matrix added
    ///
    virtual ezc3d::Matrix& operator+=(
            const ezc3d::Matrix& other);

    ///
    /// \brief Defining the subtraction with a scalar
    /// \param scalar The scalar to subtract with
    /// \return The matrix subtracted
    ///
    virtual ezc3d::Matrix operator-(
            double scalar);

    ///
    /// \brief Defining the subtraction with a scalar
    /// \param scalar The scalar to subtract with
    /// \return The matrix subtracted
    ///
    virtual ezc3d::Matrix& operator-=(
            double scalar);

    ///
    /// \brief Defining matrix subtraction
    /// \param other The matrix to subtract with
    /// \return The matrix subtracted
    ///
    virtual ezc3d::Matrix operator-(
            const ezc3d::Matrix& other);

    ///
    /// \brief Defining matrix subtraction
    /// \param other The matrix to subtract with
    /// \return The matrix subtracted
    ///
    virtual ezc3d::Matrix& operator-=(
            const ezc3d::Matrix& other);

    ///
    /// \brief Defining matrix multiplication with a scalar
    /// \param scalar The scalar to multiply with
    /// \return The matrix multiplied
    ///
    virtual ezc3d::Matrix operator*(
            double scalar);

    ///
    /// \brief Defining matrix multiplication with a scalar
    /// \param scalar The scalar to multiply with
    /// \return The matrix multiplied
    ///
    virtual ezc3d::Matrix& operator*=(
            double scalar);

    ///
    /// \brief Defining matrix multiplication
    /// \param other The matrix to multiply with
    /// \return The matrix multiplied
    ///
    virtual ezc3d::Matrix operator*(
            const ezc3d::Matrix& other);

    ///
    /// \brief Defining matrix division with a scalar
    /// \param scalar The scalar to divide by
    /// \return The matrix divided
    ///
    virtual ezc3d::Matrix operator/(
            double scalar);

    ///
    /// \brief Defining matrix division with a scalar
    /// \param scalar The scalar to divide by
    /// \return The matrix divided
    ///
    virtual ezc3d::Matrix& operator/=(
            double scalar);

};

#ifndef SWIG
///
/// \brief Defining matrix addition with a scalar
/// \param scalar The scalar to add with
/// \return The matrix added
///
ezc3d::Matrix operator+(
        double scalar,
        ezc3d::Matrix mat);

///
/// \brief Defining matrix addition with a scalar
/// \param scalar The scalar to add with
/// \return The matrix added
///
ezc3d::Matrix operator-(
        double scalar,
        ezc3d::Matrix mat);

///
/// \brief Defining matrix multiplication with a scalar
/// \param scalar The scalar to multiply with
/// \return The matrix multiplied
///
ezc3d::Matrix operator*(
        double scalar,
        ezc3d::Matrix mat);
#endif

#endif

#ifndef EZC3D_MATH_MATRIX_H
#define EZC3D_MATH_MATRIX_H
///
/// \file Matrix.h
/// \brief Declaration of a Matrix class
/// \author Pariterre
/// \version 1.0
/// \date March 25th, 2020
///

#include "ezc3d/ezc3dNamespace.h"
#include <vector>

///
/// \brief Matrix of unknown dimension
///
class EZC3D_VISIBILITY ezc3d::Matrix {
  //---- CONSTRUCTORS ----//
public:
  ///
  /// \brief Create an empty Matrix with the memory allocated
  ///
  EZC3D_API Matrix();

  ///
  /// \brief Create an empty Matrix with the memory allocated
  /// \param nbRows The number of rows
  /// \param nbCols The number of columns
  ///
  EZC3D_API Matrix(size_t nbRows, size_t nbCols);

  ///
  /// \brief Copy a Matrix
  /// \param vector The matrix to copy
  ///
  EZC3D_API Matrix(const ezc3d::Matrix &matrix);

  ///
  /// \brief Matrix Create a matrix from vector of Vector3d
  /// \param matrix The matrix to copy
  ///
  EZC3D_API Matrix(const std::vector<Vector3d> &matrix);

  ///
  /// \brief Matrix Create a matrix from vector of Vector6d
  /// \param matrix The matrix to copy
  ///
  EZC3D_API Matrix(const std::vector<Vector6d> &matrix);

  // Declare Friendship
  friend class ezc3d::Matrix33;
  friend class ezc3d::Matrix44;
  friend class ezc3d::Matrix66;
  friend class ezc3d::Vector3d;
  friend class ezc3d::Vector6d;

  //---- STREAM ----//
public:
  ///
  ///
  /// \brief Print the Matrix
  ///
  /// Print the values of the matrix to the console
  ///
  EZC3D_API virtual void print() const;

  //---- DATA ----//
protected:
  size_t _nbRows;            ///< Number of rows
  size_t _nbCols;            ///< Number of columns
  std::vector<double> _data; ///< Value of the Matrix

public:
  ///
  /// \brief The sum of all elements in the matrix
  /// \return The sum of all elements in the matrix
  ///
  EZC3D_API virtual double sum() const;

  ///
  /// \brief Set all values to zero
  ///
  EZC3D_API virtual void setZeros();

  ///
  /// \brief Set all values to one
  ///
  EZC3D_API virtual void setOnes();

  ///
  /// \brief Set the matrix to an identity matrix
  ///
  EZC3D_API virtual void setIdentity();

  ///
  /// \brief Return the number of element in the matrix (nbRows * nbCols)
  /// \return The number of element in the matrix (nbRows * nbCols)
  ///
  EZC3D_API virtual size_t size() const;

  ///
  /// \brief Return the number of rows
  /// \return The number of rows
  ///
  EZC3D_API virtual size_t nbRows() const;

  ///
  /// \brief Return the number of columns
  /// \return The number of columns
  ///
  EZC3D_API virtual size_t nbCols() const;

  ///
  /// \brief Change the size of the matrix
  /// \param nbRows The new number of rows
  /// \param nbCols The new number of columns
  ///
  /// Changes the size of the matrix.
  /// Warning, there is no garanty for the position of the data in the
  /// matrix after the resize
  ///
  EZC3D_API virtual void resize(size_t nbRows, size_t nbCols);

  //---- OPERATIONS ----//
public:
#ifndef SWIG
  ///
  /// \brief Get a specific value of the matrix
  /// \param row The row index
  /// \param col The column index
  ///
  EZC3D_API virtual double operator()(size_t row, size_t col) const;
#endif

  ///
  /// \brief Get a reference to a specific value of the matrix
  /// \param row The row index
  /// \param col The column index
  ///
  EZC3D_API virtual double &operator()(size_t row, size_t col);

  ///
  /// \brief Defining matrix transpose
  /// \return The matrix transposed
  ///
  EZC3D_API virtual ezc3d::Matrix T() const;

  ///
  /// \brief Defining the addition with a scalar
  /// \param scalar The scalar to add with
  /// \return The matrix added
  ///
  EZC3D_API virtual ezc3d::Matrix operator+(double scalar);

  ///
  /// \brief Defining the addition with a scalar
  /// \param scalar The scalar to add with
  /// \return The matrix added
  ///
  EZC3D_API virtual ezc3d::Matrix &operator+=(double scalar);

  ///
  /// \brief Defining matrix addition
  /// \param other The matrix to add with
  /// \return The matrix added
  ///
  EZC3D_API virtual ezc3d::Matrix operator+(const ezc3d::Matrix &other);

  ///
  /// \brief Defining matrix addition
  /// \param other The matrix to add with
  /// \return The matrix added
  ///
  EZC3D_API virtual ezc3d::Matrix &operator+=(const ezc3d::Matrix &other);

  ///
  /// \brief Defining the subtraction with a scalar
  /// \param scalar The scalar to subtract with
  /// \return The matrix subtracted
  ///
  EZC3D_API virtual ezc3d::Matrix operator-(double scalar);

  ///
  /// \brief Defining the subtraction with a scalar
  /// \param scalar The scalar to subtract with
  /// \return The matrix subtracted
  ///
  EZC3D_API virtual ezc3d::Matrix &operator-=(double scalar);

  ///
  /// \brief Defining matrix subtraction
  /// \param other The matrix to subtract with
  /// \return The matrix subtracted
  ///
  EZC3D_API virtual ezc3d::Matrix operator-(const ezc3d::Matrix &other);

  ///
  /// \brief Defining matrix subtraction
  /// \param other The matrix to subtract with
  /// \return The matrix subtracted
  ///
  EZC3D_API virtual ezc3d::Matrix &operator-=(const ezc3d::Matrix &other);

  ///
  /// \brief Defining matrix multiplication with a scalar
  /// \param scalar The scalar to multiply with
  /// \return The matrix multiplied
  ///
  EZC3D_API virtual ezc3d::Matrix operator*(double scalar);

  ///
  /// \brief Defining matrix multiplication with a scalar
  /// \param scalar The scalar to multiply with
  /// \return The matrix multiplied
  ///
  EZC3D_API virtual ezc3d::Matrix &operator*=(double scalar);

  ///
  /// \brief Defining matrix multiplication
  /// \param other The matrix to multiply with
  /// \return The matrix multiplied
  ///
  EZC3D_API virtual ezc3d::Matrix operator*(const ezc3d::Matrix &other);

  ///
  /// \brief Defining matrix division with a scalar
  /// \param scalar The scalar to divide by
  /// \return The matrix divided
  ///
  EZC3D_API virtual ezc3d::Matrix operator/(double scalar);

  ///
  /// \brief Defining matrix division with a scalar
  /// \param scalar The scalar to divide by
  /// \return The matrix divided
  ///
  EZC3D_API virtual ezc3d::Matrix &operator/=(double scalar);
};

#ifndef SWIG
///
/// \brief Defining matrix addition with a scalar
/// \param scalar The scalar to add with
/// \return The matrix added
///
EZC3D_VISIBILITY EZC3D_API ezc3d::Matrix operator+(double scalar,
                                                   ezc3d::Matrix mat);

///
/// \brief Defining matrix addition with a scalar
/// \param scalar The scalar to add with
/// \return The matrix added
///
EZC3D_VISIBILITY EZC3D_API ezc3d::Matrix operator-(double scalar,
                                                   ezc3d::Matrix mat);

///
/// \brief Defining matrix multiplication with a scalar
/// \param scalar The scalar to multiply with
/// \return The matrix multiplied
///
EZC3D_VISIBILITY EZC3D_API ezc3d::Matrix operator*(double scalar,
                                                   ezc3d::Matrix mat);

///
/// \brief Allows for printing matrices
/// \param out The stream to print to
/// \param m The matrix to print
/// \return The stream to print to
///
EZC3D_VISIBILITY EZC3D_API std::ostream &operator<<(std::ostream &out,
                                                    const ezc3d::Matrix &m);

#endif

#endif

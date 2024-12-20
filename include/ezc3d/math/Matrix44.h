#ifndef EZC3D_MATH_MATRIX44_H
#define EZC3D_MATH_MATRIX44_H
///
/// \file Matrix44.cpp
/// \brief Implementation of Matrix44 class
/// \author Pariterre
/// \version 1.0
/// \date April 30th, 2022
///

#include "ezc3d/math/Matrix.h"

///
/// \brief Matrix of unknown dimension
///
class EZC3D_VISIBILITY ezc3d::Matrix44 : public ezc3d::Matrix {
  //---- CONSTRUCTORS ----//
public:
  ///
  /// \brief Create an 4x4 Matrix with the memory allocated
  ///
  EZC3D_API Matrix44();

  ///
  /// \brief Create an 4x4 Matrix initalized with the 16 elements
  /// \param elem00 first col, first row
  /// \param elem01 first col, second row
  /// \param elem02 first col, third row
  /// \param elem03 first col, fourth row
  /// \param elem10 second col, first row
  /// \param elem11 second col, second row
  /// \param elem12 second col, third row
  /// \param elem13 second col, fourth row
  /// \param elem20 third col, first row
  /// \param elem21 third col, second row
  /// \param elem22 third col, third row
  /// \param elem23 third col, fourth row
  /// \param elem30 fourth col, first row
  /// \param elem31 fourth col, second row
  /// \param elem32 fourth col, third row
  /// \param elem33 fourth col, fourth row
  ///
  EZC3D_API Matrix44(double elem00, double elem01, double elem02, double elem03,
                     double elem10, double elem11, double elem12, double elem13,
                     double elem20, double elem21, double elem22, double elem23,
                     double elem30, double elem31, double elem32,
                     double elem33);

  ///
  /// \brief Copy a matrix into a Matrix44
  /// \param other The matrix to copy
  ///
  EZC3D_API Matrix44(const ezc3d::Matrix &other);

  //---- DATA ----//
public:
  ///
  /// \brief Return the number of element in the matrix (nbRows * nbCols = 16)
  /// \return The number of element in the matrix (nbRows * nbCols = 16)
  ///
  EZC3D_API size_t size() const override;

  ///
  /// \brief Return the number of rows (4)
  /// \return The number of rows (4)
  ///
  EZC3D_API size_t nbRows() const override;

  ///
  /// \brief Return the number of columns (4)
  /// \return The number of columns (4)
  ///
  EZC3D_API size_t nbCols() const override;

  ///
  /// \brief Do nothing, it is not possible to resize a 4x4 matrix
  ///
  EZC3D_API void resize(size_t, size_t) override;

  ///
  /// \brief Set an 4x4 Matrix initalized with the 16 elements
  /// \param elem00 first col, first row
  /// \param elem01 first col, second row
  /// \param elem02 first col, third row
  /// \param elem03 first col, fourth row
  /// \param elem10 second col, first row
  /// \param elem11 second col, second row
  /// \param elem12 second col, third row
  /// \param elem13 second col, fourth row
  /// \param elem20 third col, first row
  /// \param elem21 third col, second row
  /// \param elem22 third col, third row
  /// \param elem23 third col, fourth row
  /// \param elem30 fourth col, first row
  /// \param elem31 fourth col, second row
  /// \param elem32 fourth col, third row
  /// \param elem33 fourth col, fourth row
  ///
  EZC3D_API virtual void set(double elem00, double elem01, double elem02,
                             double elem03, double elem10, double elem11,
                             double elem12, double elem13, double elem20,
                             double elem21, double elem22, double elem23,
                             double elem30, double elem31, double elem32,
                             double elem33);

  //---- OPERATIONS ----//
public:
  ///
  /// \brief Defining matrix multiplication with a Vector3d
  /// \param other The vector to multiply with
  /// \return The vector multiplied
  ///
  EZC3D_API virtual ezc3d::Vector3d operator*(const ezc3d::Vector3d &other);

  ///
  /// \brief Defining matrix multiplication with a Matrix44
  /// \param other The matrix to multiply with
  /// \return The matrix multiplied
  ///
  EZC3D_API virtual ezc3d::Matrix44 operator*(const ezc3d::Matrix44 &other);
};

#endif

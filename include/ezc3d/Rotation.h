#ifndef ROTATION_H
#define ROTATION_H
///
/// \file Rotation.cpp
/// \brief Implementation of Rotation class
/// \author Pariterre
/// \version 1.0
/// \date April 30th, 2022
///

#include "ezc3d/math/Matrix44.h"

///
/// \brief 3D rotation data
///
class EZC3D_VISIBILITY ezc3d::DataNS::RotationNS::Rotation
    : public ezc3d::Matrix44 {
  //---- CONSTRUCTORS ----//
public:
  ///
  /// \brief Create an empty rotation with memory allocated but not filled
  ///
  EZC3D_API Rotation();

  ///
  /// \brief Create an empty rotation with memory allocated but not filled///
  /// \param elem00 first col, first row \param elem01 first col, second row
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
  /// \param reliability The reliability
  ///
  EZC3D_API Rotation(double elem00, double elem01, double elem02, double elem03,
                     double elem10, double elem11, double elem12, double elem13,
                     double elem20, double elem21, double elem22, double elem23,
                     double elem30, double elem31, double elem32, double elem33,
                     double reliability);

  ///
  /// \brief Create a filled rotation class at a given subframe from a given
  /// file \param c3d Reference to the c3d to copy the data in \param file File
  /// to copy the data from \param info The information about the rotations
  ///
  EZC3D_API Rotation(ezc3d::c3d &c3d, std::fstream &file,
                     const RotationNS::Info &info);

  ///
  /// \brief Copy a rotation
  /// \param rotation The rotation to copy
  ///
  EZC3D_API Rotation(const ezc3d::DataNS::RotationNS::Rotation &rotation);

  //---- STREAM ----//
public:
  ///
  ///
  /// \brief Print the rotation
  ///
  /// Print the rotation matrix to the console
  ///
  EZC3D_API virtual void print() const override;

  ///
  /// \brief Write the rotation to an opened file (scaleFactor is necessarily
  /// -1) \param f Already opened fstream file with write access
  ///
  /// Write the values of the rotation to a file
  ///
  EZC3D_API void write(std::fstream &f) const;

  //---- DATA ----//
protected:
  double _reliability; ///< Reliability metric of the rotation

public:
  ///
  /// \brief set All the values of the rotation at once
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
  /// \param reliability The reliability of the rotation
  ///
  EZC3D_API virtual void set(double elem00, double elem01, double elem02,
                             double elem03, double elem10, double elem11,
                             double elem12, double elem13, double elem20,
                             double elem21, double elem22, double elem23,
                             double elem30, double elem31, double elem32,
                             double elem33, double reliability);

  ///
  /// \brief set All the values of the rotation at once. Don't change the
  /// reliability value \param elem00 first col, first row \param elem01 first
  /// col, second row \param elem02 first col, third row \param elem03 first
  /// col, fourth row \param elem10 second col, first row \param elem11 second
  /// col, second row \param elem12 second col, third row \param elem13 second
  /// col, fourth row \param elem20 third col, first row \param elem21 third
  /// col, second row \param elem22 third col, third row \param elem23 third
  /// col, fourth row \param elem30 fourth col, first row \param elem31 fourth
  /// col, second row \param elem32 fourth col, third row \param elem33 fourth
  /// col, fourth row
  ///
  EZC3D_API virtual void set(double elem00, double elem01, double elem02,
                             double elem03, double elem10, double elem11,
                             double elem12, double elem13, double elem20,
                             double elem21, double elem22, double elem23,
                             double elem30, double elem31, double elem32,
                             double elem33) override;

  ///
  /// \brief Get the reliability component of the rotation
  /// \return The reliability component of the rotation
  ///
  EZC3D_API virtual double reliability() const;

  ///
  /// \brief Set the reliability component of the rotation
  /// \param reliability The reliability component of the rotation
  ///
  EZC3D_API virtual void reliability(double reliability);

  ///
  /// \brief If the rotation is valid
  ///
  EZC3D_API virtual bool isValid() const;

  ///
  /// \brief Returns if the rotation is empty
  /// \return If the rotation is empty
  ///
  EZC3D_API virtual bool isEmpty() const;
};

#endif

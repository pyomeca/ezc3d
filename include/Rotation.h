#ifndef ROTATION_H
#define ROTATION_H
///
/// \file Rotation.cpp
/// \brief Implementation of Rotation class
/// \author Pariterre
/// \version 1.0
/// \date April 30th, 2022
///

#include "math/Matrix44.h"

///
/// \brief 3D rotation data
///
class EZC3D_API ezc3d::DataNS::RotationNS::Rotation :
        public ezc3d::Matrix44 {
    //---- CONSTRUCTORS ----//
public:
    ///
    /// \brief Create an empty rotation with memory allocated but not filled
    ///
    Rotation();

    ///
    /// \brief Copy a rotation
    /// \param rotation The rotation to copy
    ///
    Rotation(
            const ezc3d::DataNS::RotationNS::Rotation& rotation);


    //---- STREAM ----//
public:
    ///
    ///
    /// \brief Print the rotation
    ///
    /// Print the rotation matrix to the console
    ///
    virtual void print() const override;

    ///
    /// \brief Write the rotation to an opened file
    /// \param f Already opened fstream file with write access
    /// \param scaleFactor The factor to scale the data with
    ///
    /// Write the values of the rotation to a file
    ///
    void write(
            std::fstream &f,
            float scaleFactor) const;


    //---- DATA ----//
protected:
    double _residual; ///< Residual of the rotation

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
    /// \param residual The residual of the rotation
    ///
    virtual void set(
            double elem00, double elem01, double elem02, double elem03,
            double elem10, double elem11, double elem12, double elem13,
            double elem20, double elem21, double elem22, double elem23,
            double elem30, double elem31, double elem32, double elem33,
            double residual);

    ///
    /// \brief set All the values of the rotation at once. Don't change the residual value
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
    virtual void set(
            double elem00, double elem01, double elem02, double elem03,
            double elem10, double elem11, double elem12, double elem13,
            double elem20, double elem21, double elem22, double elem23,
            double elem30, double elem31, double elem32, double elem33) override;

    ///
    /// \brief Get the residual component of the rotation
    /// \return The residual component of the rotation
    ///
    virtual double residual() const;

    ///
    /// \brief Set the residualZ component of the rotation
    /// \param residual The residual component of the rotation
    ///
    virtual void residual(
            double residual);

};

#endif

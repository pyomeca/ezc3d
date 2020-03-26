#ifndef VECTOR3D_H
#define VECTOR3D_H
///
/// \file Vector3d.h
/// \brief Declaration of a Vector3d class
/// \author Pariterre
/// \version 1.0
/// \date March 25th, 2020
///

#include <Matrix.h>

///
/// \brief 3D data
///
class EZC3D_API ezc3d::Vector3d : public ezc3d::Matrix {
    //---- CONSTRUCTORS ----//
public:
    ///
    /// \brief Create an empty 3D Vector with memory allocated but not filled
    ///
    Vector3d();

    ///
    /// \brief Copy a 3D Vector
    /// \param vector The vector to copy
    ///
    Vector3d(
            const ezc3d::Vector3d& vector);


    //---- STREAM ----//
public:
    ///
    ///
    /// \brief Print the 3D Vector
    ///
    /// Print the values of the 3D Vector to the console
    ///
    virtual void print() const override;


    //---- DATA ----//
public:
    ///
    /// \brief set All the 3dVector at once
    /// \param x The X-component of the 3D Vector
    /// \param y The Y-component of the 3D Vector
    /// \param z The Z-component of the 3D Vector
    ///
    virtual void set(
            float x,
            float y,
            float z);

    ///
    /// \brief Get the X component of the 3D Vector
    /// \return The X component of the 3d Vector
    ///
    virtual float x() const;

    ///
    /// \brief Set the X component of the 3D Vector
    /// \param x The X component of the 3d Vector
    ///
    virtual void x(
            float x);

    ///
    /// \brief Get the Y component of the 3D Vector
    /// \return The Y component of the 3d Vector
    ///
    virtual float y() const;

    ///
    /// \brief Set the Y component of the 3D Vector
    /// \param y The Y component of the 3d Vector
    ///
    virtual void y(
            float y);

    ///
    /// \brief Get the Z component of the 3D Vector
    /// \return The Z component of the 3d Vector
    ///
    virtual float z() const;

    ///
    /// \brief Set the Z component of the 3D Vector
    /// \param z The Z component of the 3d Vector
    ///
    virtual void z(
            float z);

    ///
    /// \brief If any component is a NAN then the 3D Vector is invalid.
    /// \return If the 3D Vector is invalid
    ///
    bool isValid() const;

};

#endif

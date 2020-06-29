#ifndef EZC3D_MATH_VECTOR3D_H
#define EZC3D_MATH_VECTOR3D_H
///
/// \file Vector3d.h
/// \brief Declaration of a Vector3d class
/// \author Pariterre
/// \version 1.0
/// \date March 25th, 2020
///

#include "Matrix.h"

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
    /// \brief Vector3d Construct a 3d Vector from XYZ components
    /// \param x The X-component of the 3D Vector
    /// \param y The Y-component of the 3D Vector
    /// \param z The Z-component of the 3D Vector
    ///
    Vector3d(
            double x,
            double y,
            double z);

    ///
    /// \brief Copy a 3D Vector
    /// \param vector The vector to copy
    ///
    Vector3d(
            const ezc3d::Matrix& vector);


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
    /// \brief Do nothing, it is not possible to resize a 3x1 vector
    ///
    void resize(
            size_t,
            size_t) override;

    ///
    /// \brief set All the 3dVector at once
    /// \param x The X-component of the 3D Vector
    /// \param y The Y-component of the 3D Vector
    /// \param z The Z-component of the 3D Vector
    ///
    virtual void set(
            double x,
            double y,
            double z);

    ///
    /// \brief Get the X component of the 3D Vector
    /// \return The X component of the 3d Vector
    ///
    virtual double x() const;

    ///
    /// \brief Set the X component of the 3D Vector
    /// \param x The X component of the 3d Vector
    ///
    virtual void x(
            double x);

    ///
    /// \brief Get the Y component of the 3D Vector
    /// \return The Y component of the 3d Vector
    ///
    virtual double y() const;

    ///
    /// \brief Set the Y component of the 3D Vector
    /// \param y The Y component of the 3d Vector
    ///
    virtual void y(
            double y);

    ///
    /// \brief Get the Z component of the 3D Vector
    /// \return The Z component of the 3d Vector
    ///
    virtual double z() const;

    ///
    /// \brief Set the Z component of the 3D Vector
    /// \param z The Z component of the 3d Vector
    ///
    virtual void z(
            double z);

    ///
    /// \brief If any component is a NAN then the 3D Vector is invalid.
    /// \return If the 3D Vector is invalid
    ///
    virtual bool isValid() const;

    //---- OPERATIONS ----//
public:
#ifndef SWIG
    ezc3d::Vector3d& operator=(
            const ezc3d::Matrix& other);

    ///
    /// \brief Get a specific value of the vector
    /// \param row The index
    ///
    virtual double operator()(
            size_t row) const;
#endif

    ///
    /// \brief Get a reference to a specific value of the vector
    /// \param row The index
    ///
    virtual double& operator()(
            size_t row);

    ///
    /// \brief Returns the dot-product of two vectors
    /// \param other The vector to dot-product with
    /// \return The dot-product
    ///
    virtual double dot(const ezc3d::Vector3d& other);

    ///
    /// \brief Returns the cross-product of two vectors
    /// \param other The vector to cross-product with
    /// \return The cross-product
    ///
    virtual ezc3d::Vector3d cross(const ezc3d::Vector3d& other);

    ///
    /// \brief Returns the euclidian norm of the vector (i.e. sqrt(vx^2 + vy^2 + vz^2) )
    /// \return The norm of the vector
    ///
    virtual double norm();

    ///
    /// \brief Normalize the current vector (i.e. v = v/norm(v) )
    ///
    virtual void normalize();

};

#endif

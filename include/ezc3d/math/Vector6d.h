#ifndef EZC3D_MATH_VECTOR6D_H
#define EZC3D_MATH_VECTOR6D_H
///
/// \file Vector6d.h
/// \brief Declaration of a Vector6d class
/// \author Pariterre
/// \version 1.0
/// \date March 25th, 2020
///

#include "Matrix.h"

///
/// \brief 3D data
///
class EZC3D_API ezc3d::Vector6d : public ezc3d::Matrix {
    //---- CONSTRUCTORS ----//
public:
    ///
    /// \brief Create an empty 6D Vector with memory allocated but not filled
    ///
    Vector6d();

    ///
    /// \brief Construct a 6d Vector
    /// \param e0 The first element
    /// \param e1 The second element
    /// \param e2 The third element
    /// \param e3 The fourth element
    /// \param e4 The fifth element
    /// \param e5 The sixth element
    ///
    Vector6d(
            double e0,
            double e1,
            double e2,
            double e3,
            double e4,
            double e5);

    ///
    /// \brief Copy a 6D Vector
    /// \param vector The vector to copy
    ///
    Vector6d(
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

    //---- OPERATIONS ----//
public:
    ///
    /// \brief Do nothing, it is not possible to resize a 6x1 vector
    ///
    void resize(
            size_t,
            size_t) override;

#ifndef SWIG
    ezc3d::Vector6d& operator=(
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

};

#endif

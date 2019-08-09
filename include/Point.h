#ifndef POINT_H
#define POINT_H
///
/// \file Point.h
/// \brief Declaration of Point class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include <ezc3d.h>

///
/// \brief 3D point data
///
class EZC3D_API ezc3d::DataNS::Points3dNS::Point{
    //---- CONSTRUCTORS ----//
public:
    ///
    /// \brief Create an empty 3D point with memory allocated but not filled
    ///
    Point();

    ///
    /// \brief Copy a 3D point
    /// \param point The point to copy
    ///
    Point(const ezc3d::DataNS::Points3dNS::Point& point);


    //---- STREAM ----//
public:
    ///
    ///
    /// \brief Print the point
    ///
    /// Print the values of the point to the console
    ///
    void print() const;

    ///
    /// \brief Write the point to an opened file
    /// \param f Already opened fstream file with write access
    ///
    /// Write the values of the point to a file
    ///
    void write(std::fstream &f) const;


    //---- DATA ----//
protected:
    std::vector<float> _data; ///< Value of the point
public:
    ///
    /// \brief Get a reference to the STL vector where the 3D point is store
    /// \return The 3d point
    ///
    const std::vector<float> data() const;

    ///
    /// \brief Get a reference to the STL vector where the 3D point is store in order to be modified by the caller
    /// \return The 3d point
    ///
    /// Get a reference to the STL vector where the 3D point is store in the form of a non-const reference.
    /// The user can thereafter modify these points at will, but with the caution it requires.
    ///
    std::vector<float> data_nonConst();

    ///
    /// \brief Get the X component of the 3D point
    /// \return The X component of the 3d point
    ///
    float x() const;

    ///
    /// \brief Set the X component of the 3D point
    /// \param x The X component of the 3d point
    ///
    void x(float x);

    ///
    /// \brief Get the Y component of the 3D point
    /// \return The Y component of the 3d point
    ///
    float y() const;

    ///
    /// \brief Set the Y component of the 3D point
    /// \param y The Y component of the 3d point
    ///
    void y(float y);

    ///
    /// \brief Get the Z component of the 3D point
    /// \return The Z component of the 3d point
    ///
    float z() const;

    ///
    /// \brief Set the Z component of the 3D point
    /// \param z The Z component of the 3d point
    ///
    void z(float z);

    ///
    /// \brief Get the residual component of the 3D point
    /// \return The residual component of the 3d point
    ///
    float residual() const;

    ///
    /// \brief Set the residualZ component of the 3D point
    /// \param residual The residual component of the 3d point
    ///
    void residual(float residual);

    ///
    /// \brief Return if the point is empty
    /// \return if the point is empty
    ///
    bool isempty() const;
};

#endif

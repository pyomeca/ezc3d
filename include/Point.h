#ifndef POINT_H
#define POINT_H
///
/// \file Point.h
/// \brief Declaration of Point class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include <sstream>
#include <memory>
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
    ///
    Point(const ezc3d::DataNS::Points3dNS::Point&);


    //---- STREAM ----//
public:
    ///
    ///
    /// \brief Print the point
    ///
    /// Print the actual values of the point to the console
    ///
    void print() const;

    ///
    /// \brief Write the point to an opened file
    /// \param f Already opened fstream file with write access
    ///
    /// Write the actual values of the point to a file
    ///
    void write(std::fstream &f) const;


    //---- METADATA ----//
protected:
    std::string _name; ///< Name of the point
public:
    ///
    /// \brief Get the name of the point
    /// \return The name of a point
    ///
    const std::string& name() const;

    ///
    /// \brief Set the name of the point
    /// \param name Name of the point
    ///
    void name(const std::string &name);


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
    /// \brief Get the X component of the 3D point
    /// \return The X component of the 3d point
    ///
    float x() const;

    ///
    /// \brief Set the X component of the 3D point
    /// \return The X component of the 3d point
    ///
    void x(float x);

    ///
    /// \brief Get the Y component of the 3D point
    /// \return The Y component of the 3d point
    ///
    float y() const;

    ///
    /// \brief Set the Y component of the 3D point
    /// \return The Y component of the 3d point
    ///
    void y(float y);

    ///
    /// \brief Get the Z component of the 3D point
    /// \return The Z component of the 3d point
    ///
    float z() const;

    ///
    /// \brief Set the Z component of the 3D point
    /// \return The Z component of the 3d point
    ///
    void z(float z);

    ///
    /// \brief Get the residual component of the 3D point
    /// \return The residual component of the 3d point
    ///
    float residual() const;

    ///
    /// \brief Set the residualZ component of the 3D point
    /// \return The residual component of the 3d point
    ///
    void residual(float residual);
};

#endif

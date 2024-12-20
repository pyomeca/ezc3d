#ifndef POINT_H
#define POINT_H
///
/// \file Point.h
/// \brief Declaration of Point class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "ezc3d/math/Vector3d.h"

///
/// \brief 3D point data
///
class EZC3D_VISIBILITY ezc3d::DataNS::Points3dNS::Point
    : public ezc3d::Vector3d {
  //---- CONSTRUCTORS ----//
public:
  ///
  /// \brief Create an empty 3D point with memory allocated but not filled
  ///
  EZC3D_API Point();

  ///
  /// \brief Copy a 3D point
  /// \param point The point to copy
  ///
  EZC3D_API Point(const ezc3d::DataNS::Points3dNS::Point &point);

  ///
  /// \brief Create a filled Point class from a given file
  /// \param c3d Reference to the c3d to copy the data in
  /// \param file File to copy the data from
  /// \param info The information about the points
  ///
  EZC3D_API Point(ezc3d::c3d &c3d, std::fstream &file,
                  const Points3dNS::Info &info);

  //---- STREAM ----//
public:
  ///
  ///
  /// \brief Print the point
  ///
  /// Print the values of the point to the console
  ///
  EZC3D_API virtual void print() const override;

  ///
  /// \brief Write the point to an opened file
  /// \param f Already opened fstream file with write access
  /// \param scaleFactor The factor to scale the data with
  ///
  /// Write the values of the point to a file
  ///
  EZC3D_API void write(std::fstream &f, float scaleFactor) const;

  //---- DATA ----//
protected:
  double _residual;               ///< Residual of the point
  std::vector<bool> _cameraMasks; ///< If the cameras 1-7 are masked

public:
  ///
  /// \brief set All the point at once
  /// \param x The X-component of the point
  /// \param y The Y-component of the point
  /// \param z The Z-component of the point
  /// \param residual The residual of the point
  ///
  EZC3D_API virtual void set(double x, double y, double z, double residual);

  ///
  /// \brief set All the point at once. Don't change the residual value
  /// \param x The X-component of the point
  /// \param y The Y-component of the point
  /// \param z The Z-component of the point
  ///
  EZC3D_API virtual void set(double x, double y, double z) override;

  ///
  /// \brief Get the X component of the Point
  /// \return The X component of the Point
  ///
  EZC3D_API virtual double x() const override;

  ///
  /// \brief Set the X component of the 3D Point
  /// \param x The X component of the 3d Point
  ///
  EZC3D_API virtual void x(double x) override;

  ///
  /// \brief Get the Y component of the Point
  /// \return The Y component of the Point
  ///
  EZC3D_API virtual double y() const override;

  ///
  /// \brief Set the Y component of the 3D Point
  /// \param y The Y component of the 3d Point
  ///
  EZC3D_API virtual void y(double y) override;

  ///
  /// \brief Get the Z component of the Point
  /// \return The Z component of the Point
  ///
  EZC3D_API virtual double z() const override;

  ///
  /// \brief Set the Z component of the 3D Point
  /// \param y The Z component of the 3d Point
  ///
  EZC3D_API virtual void z(double z) override;

  ///
  /// \brief Get the residual component of the 3D point
  /// \return The residual component of the 3d point
  ///
  EZC3D_API virtual double residual() const;

  ///
  /// \brief Set the residualZ component of the 3D point
  /// \param residual The residual component of the 3d point
  ///
  EZC3D_API virtual void residual(double residual);

  ///
  /// \brief Return if the cameras of index 0 to 6 are masked
  /// \return If the camera are masked
  ///
  EZC3D_API virtual const std::vector<bool> &cameraMask() const;

  ///
  /// \brief Set the masks for the camera of index 0 to 6
  /// \param masks If the cameras are masked or not
  ///
  EZC3D_API virtual void cameraMask(const std::vector<bool> &masks);

  ///
  /// \brief Set the masks. The byte is break down into 7 camera bit by bit
  /// \param byte The cameras masks into a byte format
  ///
  EZC3D_API virtual void cameraMask(int byte);

  ///
  /// \brief Return if the point is empty
  /// \return if the point is empty
  ///
  EZC3D_API virtual bool isEmpty() const;
};

#endif

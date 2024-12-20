#ifndef POINTS_H
#define POINTS_H
///
/// \file Points.h
/// \brief Declaration of Points class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "ezc3d/Point.h"

///
/// \brief Points holder for C3D data 3D points data
///
class EZC3D_VISIBILITY ezc3d::DataNS::Points3dNS::Points {
  //---- CONSTRUCTORS ----//
public:
  ///
  /// \brief Create an empty holder for 3D points
  ///
  EZC3D_API Points();

  ///
  /// \brief Create an empty holder for 3D points preallocating the size of it
  /// \param nbPoints Number of 3D points to be in the holder
  ///
  EZC3D_API Points(size_t nbPoints);

  ///
  /// \brief Create a filled Points class at a given frame from a given file
  /// \param c3d Reference to the c3d to copy the data in
  /// \param file File to copy the data from
  /// \param info The information about the points
  ///
  EZC3D_API Points(ezc3d::c3d &c3d, std::fstream &file,
                   const Points3dNS::Info &info);

  //---- STREAM ----//
public:
  ///
  ///
  /// \brief Print the points
  ///
  /// Print the points to the console by calling sequentially the print method
  /// for all the points
  ///
  EZC3D_API void print() const;

  ///
  /// \brief Write points to an opened file
  /// \param f Already opened fstream file with write access
  /// \param scaleFactor The factor to scale the data with
  ///
  /// Write all the points to a file by calling sequentially the write method of
  /// each point
  ///
  EZC3D_API void write(std::fstream &f, std::vector<double> scaleFactor) const;

  //---- POINT ----//
protected:
  std::vector<ezc3d::DataNS::Points3dNS::Point>
      _points; ///< Holder of the 3D points
public:
  ///
  /// \brief Get the number of points
  /// \return The number of points
  ///
  EZC3D_API size_t nbPoints() const;

  ///
  /// \brief Get a particular point of index idx from the 3D points data
  /// \param idx The index of the point
  /// \return The point
  ///
  /// Throw a std::out_of_range exception if idx is larger than the number of
  /// frames
  ///
  EZC3D_API const ezc3d::DataNS::Points3dNS::Point &point(size_t idx) const;

  ///
  /// \brief Get a particular point of index idx from the 3D points data in
  /// order to be modified by the caller \param idx The index of the point
  /// \return The point
  ///
  /// Get a particular point of index idx from the 3D points data in the form of
  /// a non-const reference. The user can thereafter modify these points at
  /// will, but with the caution it requires.
  ///
  /// Throw a std::out_of_range exception if idx is larger than the number of
  /// frames
  ///
  EZC3D_API ezc3d::DataNS::Points3dNS::Point &point(size_t idx);

  ///
  /// \brief Add/replace a point to the points data set
  /// \param point The point to add
  /// \param idx The index of the point in the points data set
  ///
  /// Add or replace a particular point to the points data set.
  ///
  /// If no idx is sent, then the point is appended to the points data set.
  /// If the idx correspond to a pre-existing point, it replaces it.
  /// If idx is larger than the number of points, it resize the points
  /// accordingly and add the point where it belongs but leaves the other
  /// created points empty.
  ///
  EZC3D_API void point(const ezc3d::DataNS::Points3dNS::Point &point,
                       size_t idx = SIZE_MAX);

  ///
  /// \brief Get all the points from the 3D points data
  /// \return The points
  ///
  EZC3D_API const std::vector<ezc3d::DataNS::Points3dNS::Point> &points() const;

  ///
  /// \brief Return if the points are empty
  /// \return if the points are empty
  ///
  EZC3D_API bool isEmpty() const;
};

#endif

#ifndef FRAME_H
#define FRAME_H
///
/// \file Frame.h
/// \brief Declaration of Frame class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "ezc3d/Analogs.h"
#include "ezc3d/Points.h"
#include "ezc3d/Rotations.h"

///
/// \brief Frame holder for C3D data
///
class EZC3D_VISIBILITY ezc3d::DataNS::Frame {
  //---- CONSTRUCTORS ----//
public:
  ///
  /// \brief Create an empty frame
  ///
  EZC3D_API Frame();

  //---- STREAM ----//
public:
  ///
  ///
  /// \brief Print the frame
  ///
  /// Print the frame to the console by calling sequentially the print method
  /// for points and analogs
  ///
  EZC3D_API void print() const;

  ///
  /// \brief Write a frame to an opened file
  /// \param f Already opened fstream file with write access
  /// \param pointScaleFactor The factor to scale the point data with
  /// \param analogScaleFactors The factor to scale the analog data with
  /// \param dataTypeToWrite The type of data block (0 points/analogs, 1
  /// rotations)
  ///
  /// Write the frame to a file by calling sequentially the write method for
  /// points and analogs
  ///
  EZC3D_API void write(std::fstream &f, std::vector<double> pointScaleFactor,
                       std::vector<double> analogScaleFactors,
                       int dataTypeToWrite) const;

  //---- POINTS ----//
protected:
  std::shared_ptr<ezc3d::DataNS::Points3dNS::Points>
      _points; ///< All the points for this frame
public:
  ///
  /// \brief Return a reference to all the points
  /// \return Reference to all the points
  ///
  EZC3D_API const ezc3d::DataNS::Points3dNS::Points &points() const;

  ///
  /// \brief Return a reference to all the points in order to be modified by the
  /// caller \return A non-const reference to all the points
  ///
  /// Get all the points in the form of a non-const reference.
  /// The user can thereafter modify these points at will, but with the caution
  /// it requires.
  ///
  EZC3D_API ezc3d::DataNS::Points3dNS::Points &points();

  //---- ANALOGS ----//
protected:
  std::shared_ptr<ezc3d::DataNS::AnalogsNS::Analogs>
      _analogs; ///< All the subframes for all the analogs
public:
  ///
  /// \brief Return a reference to all the analogs
  /// \return Reference to all the analogs
  ///
  EZC3D_API const ezc3d::DataNS::AnalogsNS::Analogs &analogs() const;

  ///
  /// \brief Return a reference to all the analogs in order to be modified by
  /// the caller \return A non-const reference to all the analogs
  ///
  /// Get all the analogs in the form of a non-const reference.
  /// The user can thereafter modify these analogs at will, but with the caution
  /// it requires.
  ///
  EZC3D_API ezc3d::DataNS::AnalogsNS::Analogs &analogs();

  //---- ROTATIONS ----//
protected:
  std::shared_ptr<ezc3d::DataNS::RotationNS::Rotations>
      _rotations; ///< All the rotations for this frame

public:
  ///
  /// \brief Return a reference to all the rotations
  /// \return Reference to all the rotations
  ///
  EZC3D_API const ezc3d::DataNS::RotationNS::Rotations &rotations() const;

  ///
  /// \brief Return a reference to all the rotations in order to be modified by
  /// the caller \return A non-const reference to all the rotations
  ///
  /// Get all the rotations in the form of a non-const reference.
  /// The user can thereafter modify these rotations at will, but with the
  /// caution it requires.
  ///
  EZC3D_API ezc3d::DataNS::RotationNS::Rotations &rotations();

  //---- ACCESSORS ----//
public:
  ///
  /// \brief Add a frame by copying a sent frame
  /// \param frame The frame to copy
  ///
  EZC3D_API void add(const ezc3d::DataNS::Frame &frame);

  ///
  /// \brief Add points to a frame
  /// \param points The 3D points to add
  ///
  EZC3D_API void add(const ezc3d::DataNS::Points3dNS::Points &points);

  ///
  /// \brief Add analogs to a frame
  /// \param analogs The analogous data to add
  ///
  EZC3D_API void add(const ezc3d::DataNS::AnalogsNS::Analogs &analogs);

  ///
  /// \brief Add rotations to a frame
  /// \param rotations The rotations data to add
  ///
  EZC3D_API void add(const ezc3d::DataNS::RotationNS::Rotations &rotations);

  ///
  /// \brief Add points and analogs to a frame
  /// \param points The 3D points to add
  /// \param analogs The analogous data to add
  ///
  EZC3D_API void add(const ezc3d::DataNS::Points3dNS::Points &points,
                     const ezc3d::DataNS::AnalogsNS::Analogs &analogs);

  ///
  /// \brief Add points and analogs to a frame
  /// \param points The 3D points to add
  /// \param analogs The analogous data to add
  /// \param rotations The rotations data to add
  ///
  EZC3D_API void add(const ezc3d::DataNS::Points3dNS::Points &points,
                     const ezc3d::DataNS::AnalogsNS::Analogs &analogs,
                     const ezc3d::DataNS::RotationNS::Rotations &rotations);

  ///
  /// \brief Return if the frame is empty
  /// \return if the frame is empty
  ///
  EZC3D_API bool isEmpty() const;
};

#endif

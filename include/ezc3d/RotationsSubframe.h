#ifndef ROTATIONS_SUBFRAME_H
#define ROTATIONS_SUBFRAME_H
///
/// \file RotationsSubframe.h
/// \brief Declaration of Subframe class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "ezc3d/Rotation.h"

///
/// \brief Subframe for the rotation data
///
class EZC3D_VISIBILITY ezc3d::DataNS::RotationNS::SubFrame {
  //---- CONSTRUCTORS ----//
public:
  ///
  /// \brief Create an empty subframe for rotation data
  ///
  EZC3D_API SubFrame();

  ///
  /// \brief Create a filled SubFrame class at a given frame from a given file
  /// \param c3d Reference to the c3d to copy the data in
  /// \param file File to copy the data from
  /// \param info The information about the rotations
  ///
  EZC3D_API SubFrame(ezc3d::c3d &c3d, std::fstream &file,
                     const RotationNS::Info &info);

  //---- STREAM ----//
public:
  ///
  ///
  /// \brief Print the subframe
  ///
  /// Print the subframe to the console by calling sequentially the print method
  /// of all of the rotation
  ///
  EZC3D_API void print() const;

  ///
  /// \brief Write the subframe to an opened file
  /// \param f Already opened fstream file with write access
  ///
  /// Write the subframe to a file by calling sequentially the write method of
  /// all of the rotation
  ///
  EZC3D_API void write(std::fstream &f) const;

  //---- ROTATIONS ----//
protected:
  std::vector<ezc3d::DataNS::RotationNS::Rotation>
      _rotations; ///< Holder for the rotations
public:
  ///
  /// \brief Get the number of rotations
  /// \return The number of rotations
  ///
  EZC3D_API size_t nbRotations() const;

  ///
  /// \brief Resize the number of rotations. Warning, this function drops data
  /// if rotations are downsized. \param nRotations Number of rotations in the
  /// subframe
  ///
  EZC3D_API void nbRotations(size_t nRotations);

  ///
  /// \brief Get a particular rotation of index idx from the rotations data
  /// \param idx The index of the rotation
  /// \return The rotation
  ///
  /// Throw a std::out_of_range exception if idx is larger than the number of
  /// frames
  ///
  EZC3D_API const ezc3d::DataNS::RotationNS::Rotation &
  rotation(size_t idx) const;

  ///
  /// \brief Get a particular rotation of index idx from the rotations data
  /// \param idx The index of the rotation
  /// \return The rotation
  ///
  /// Throw a std::out_of_range exception if idx is larger than the number of
  /// frames
  ///
  EZC3D_API ezc3d::DataNS::RotationNS::Rotation &rotation(size_t idx);

  ///
  /// \brief Add/replace a rotation to the analog subframe data set
  /// \param rotation the rotation to add
  /// \param idx the index of the rotation in the subframe data set
  ///
  /// Add or replace a particular rotation to the subframe data set.
  ///
  /// If no idx is sent, then the rotation is appended to the points data set.
  /// If the idx correspond to a pre-existing rotation, it replaces it.
  /// If idx is larger than the number of rotation, it resize the subframe
  /// accordingly and add the channel where it belongs but leaves the other
  /// created rotations empty.
  ///
  EZC3D_API void rotation(const ezc3d::DataNS::RotationNS::Rotation &rotation,
                          size_t idx = SIZE_MAX);

  ///
  /// \brief Get all the rotations from the 3D rotations data at a specific
  /// frame \return The rotations
  ///
  EZC3D_API const std::vector<ezc3d::DataNS::RotationNS::Rotation> &
  rotations() const;

  ///
  /// \brief Return if the subframe is empty
  /// \return if the subframe is empty
  ///
  EZC3D_API bool isEmpty() const;
};

#endif

#ifndef ROTATIONS_H
#define ROTATIONS_H
///
/// \file Rotations.cpp
/// \brief Implementation of Rotations class
/// \author Pariterre
/// \version 1.0
/// \date April 30th, 2022
///

#include "ezc3d/RotationsSubframe.h"

///
/// \brief Rotation holder for C3D Rotations data
/// base on documentation from
/// https://www.c-motion.com/v3dwiki/index.php?title=ROTATION_DATA_TYPE
///
class EZC3D_VISIBILITY ezc3d::DataNS::RotationNS::Rotations {
  //---- CONSTRUCTORS ----//
public:
  ///
  /// \brief Create an empty holder for Rotation data
  ///
  EZC3D_API Rotations();

  ///
  /// \brief Create an empty holder for Rotation data preallocating the size of
  /// it \param c3d Reference to the c3d to copy the data in \param file File to
  /// copy the data from \param info The information about the rotations
  ///
  EZC3D_API Rotations(ezc3d::c3d &c3d, std::fstream &file,
                      const RotationNS::Info &info);

  //---- STREAM ----//
public:
  ///
  ///
  /// \brief Print the rotations
  ///
  /// Print the Rotations to the console by calling sequentially the print
  /// method for all the rotations
  ///
  EZC3D_API void print() const;

  ///
  /// \brief Write rotations to an opened file (scaleFactor is necessarily -1)
  /// \param f Already opened fstream file with write access
  ///
  /// Write all the rotations to a file by calling sequentially the write method
  /// of each rotation
  ///
  EZC3D_API void write(std::fstream &f) const;

  //---- ROTATION ----//
protected:
  std::vector<ezc3d::DataNS::RotationNS::SubFrame>
      _subframe; ///< Holder of the 3D rotations at each frame

public:
  ///
  /// \brief Get the number of subframes
  /// \return The number of subframes
  ///
  EZC3D_API size_t nbSubframes() const;

  ///
  /// \brief Resize the number of subframes. Warning, this function drops data
  /// if subframes is downsized \param nbSubframes The number of subframes to be
  /// in the holder
  ///
  EZC3D_API void nbSubframes(size_t nbSubframes);

  ///
  /// \brief Get a particular subframe of index idx from the rotation data set
  /// \param idx The index of the subframe
  /// \return The Subframe
  ///
  /// Get a particular subframe of index idx from the rotation data set.
  ///
  /// Throw a std::out_of_range exception if idx is larger than the number of
  /// subframes
  ///
  EZC3D_API const ezc3d::DataNS::RotationNS::SubFrame &
  subframe(size_t idx) const;

  ///
  /// \brief Get a particular subframe of index idx from the rotation data set
  /// in order to be modified by the caller \param idx The index of the subframe
  /// \return A non-const reference to the subframe
  ///
  /// Get a particular subframe of index idx from the rotation data in the form
  /// of a non-const reference. The user can thereafter modify these points at
  /// will, but with the caution it requires.
  ///
  /// Throw a std::out_of_range exception if idx is larger than the number of
  /// subframes
  ///
  EZC3D_API ezc3d::DataNS::RotationNS::SubFrame &subframe(size_t idx);

  ///
  /// \brief Add/replace a subframe to the rotation data set
  /// \param subframe The subframe to add
  /// \param idx The index of the subframe in the rotation data set
  ///
  /// Add or replace a subframe to the rotation data set.
  ///
  /// If no idx is sent, then the subframe is appended to the rotation data set.
  /// If the idx correspond to a pre-existing subframe, it replaces it.
  /// If idx is larger than the number of subframes, it resize the rotation data
  /// set accordingly and add the subframe where it belongs but leaves the other
  /// created subframes empty.
  ///
  EZC3D_API void subframe(const ezc3d::DataNS::RotationNS::SubFrame &subframe,
                          size_t idx = SIZE_MAX);

  ///
  /// \brief Get all the subframes from the rotation data set
  /// \return The subframes
  ///
  EZC3D_API const std::vector<ezc3d::DataNS::RotationNS::SubFrame> &
  subframes() const;

  ///
  /// \brief Return if the rotations are empty
  /// \return if the rotations are empty
  ///
  EZC3D_API bool isEmpty() const;
};

#endif

#ifndef ROTATIONS_H
#define ROTATIONS_H
///
/// \file Rotations.cpp
/// \brief Implementation of Rotations class
/// \author Pariterre
/// \version 1.0
/// \date April 30th, 2022
///

#include "Rotation.h"

///
/// \brief Rotation holder for C3D Rotations data
/// base on documentation from https://www.c-motion.com/v3dwiki/index.php?title=ROTATION_DATA_TYPE
///
class EZC3D_API ezc3d::DataNS::RotationNS::Rotations{
    //---- CONSTRUCTORS ----//
public:
    ///
    /// \brief Create an empty holder for Rotation data
    ///
    Rotations();

    ///
    /// \brief Create an empty holder for Rotation data preallocating the size of it
    /// \param c3d Reference to the c3d to copy the data in
    /// \param file File to copy the data from
    ///
    Rotations(
            ezc3d::c3d &c3d,
            std::fstream &file);


    //---- STREAM ----//
public:
    ///
    ///
    /// \brief Print the rotations
    ///
    /// Print the Rotations to the console by calling sequentially the print method for all the rotations
    ///
    void print() const;

    ///
    /// \brief Write rotations to an opened file (scaleFactor is necessarily -1)
    /// \param f Already opened fstream file with write access
    ///
    /// Write all the rotations to a file by calling sequentially the write method of each rotation
    ///
    void write(
            std::fstream &f) const;


    //---- ROTATION ----//
protected:
    std::vector<std::vector<ezc3d::DataNS::RotationNS::Rotation>> _rotations; ///< Holder of the 3D rotations at each frame
    size_t _nbRotations; ///< Remembers the number of rotations

public:
    ///
    /// \brief Get the number of frames
    /// \return The number of frames
    ///
    size_t nbFrames() const;

    ///
    /// \brief Get the number of rotations
    /// \return The number of rotations
    ///
    size_t nbRotations() const;

    ///
    /// \brief Get a particular rotation of index idx from the rotations data
    /// \param frame The frame index
    /// \param idx The index of the rotation
    /// \return The rotation
    ///
    /// Throw a std::out_of_range exception if idx is larger than the number of frames
    ///
    const ezc3d::DataNS::RotationNS::Rotation& rotation(
            size_t frame,
            size_t idx) const;

    ///
    /// \brief Get a particular rotation of index idx from the 3D rotations data in order to be modified by the caller
    /// \param frame The frame index
    /// \param idx The index of the rotation
    /// \return The rotation
    ///
    /// Get a particular rotation of index idx from the 3D rotations data in the form of a non-const reference.
    /// The user can thereafter modify these rotations at will, but with the caution it requires.
    ///
    /// Throw a std::out_of_range exception if idx is larger than the number of frames
    ///
    ezc3d::DataNS::RotationNS::Rotation& rotation(
            size_t frame,
            size_t idx);

    ///
    /// \brief Get all the rotations from the 3D rotations data at a specific frame
    /// \param frame The frame
    /// \return The rotations
    ///
    const std::vector<ezc3d::DataNS::RotationNS::Rotation>& rotations(
            size_t frame) const;

    ///
    /// \brief Return if the rotations are empty
    /// \return if the rotations are empty
    ///
    bool isEmpty() const;
};

#endif

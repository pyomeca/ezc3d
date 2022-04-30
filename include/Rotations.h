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
    /// \param nbRotations Number of Rotation data to be in the holder
    ///
    Rotations(
            size_t nbRotations);


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
    /// \brief Write rotations to an opened file
    /// \param f Already opened fstream file with write access
    /// \param scaleFactor The factor to scale the data with
    ///
    /// Write all the rotations to a file by calling sequentially the write method of each rotation
    ///
    void write(
            std::fstream &f,
            float scaleFactor) const;


    //---- ROTATION ----//
protected:
    std::vector<ezc3d::DataNS::RotationNS::Rotation> _rotations; ///< Holder of the 3D rotations
public:
    ///
    /// \brief Get the number of rotations
    /// \return The number of rotations
    ///
    size_t nbRotations() const;

    ///
    /// \brief Get a particular rotation of index idx from the rotations data
    /// \param idx The index of the rotation
    /// \return The rotation
    ///
    /// Throw a std::out_of_range exception if idx is larger than the number of frames
    ///
    const ezc3d::DataNS::RotationNS::Rotation& rotation(
            size_t idx) const;

    ///
    /// \brief Get a particular rotation of index idx from the 3D rotations data in order to be modified by the caller
    /// \param idx The index of the rotation
    /// \return The rotation
    ///
    /// Get a particular rotation of index idx from the 3D rotations data in the form of a non-const reference.
    /// The user can thereafter modify these rotations at will, but with the caution it requires.
    ///
    /// Throw a std::out_of_range exception if idx is larger than the number of frames
    ///
    ezc3d::DataNS::RotationNS::Rotation& rotation(
            size_t idx);

    ///
    /// \brief Add/replace a rotation to the rotations data set
    /// \param rotation The rotation to add
    /// \param idx The index of the rotation in the rotations data set
    ///
    /// Add or replace a particular rotation to the rotations data set.
    ///
    /// If no idx is sent, then the rotation is appended to the rotations data set.
    /// If the idx correspond to a pre-existing rotation, it replaces it.
    /// If idx is larger than the number of rotations, it resize the rotations accordingly and add the rotation
    /// where it belongs but leaves the other created rotations empty.
    ///
    void rotation(
            const ezc3d::DataNS::RotationNS::Rotation& rotation,
            size_t idx = SIZE_MAX);

    ///
    /// \brief Get all the rotations from the 3D rotations data
    /// \return The rotations
    ///
    const std::vector<ezc3d::DataNS::RotationNS::Rotation>& rotations() const;

    ///
    /// \brief Return if the rotations are empty
    /// \return if the rotations are empty
    ///
    bool isEmpty() const;
};

#endif

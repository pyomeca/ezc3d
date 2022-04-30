#define EZC3D_API_EXPORTS
///
/// \file Rotations.cpp
/// \brief Implementation of Rotations class
/// \author Pariterre
/// \version 1.0
/// \date April 30th, 2022
///

#include "Rotations.h"

// Rotations data
ezc3d::DataNS::RotationNS::Rotations::Rotations() {
}

ezc3d::DataNS::RotationNS::Rotations::Rotations(
        size_t nbRotations) {
    _rotations.resize(nbRotations);
}

void ezc3d::DataNS::RotationNS::Rotations::print() const {
    for (size_t i = 0; i < nbRotations(); ++i)
        rotation(i).print();
}

void ezc3d::DataNS::RotationNS::Rotations::write(
        std::fstream &f,
        float scaleFactor) const {
    for (size_t i = 0; i < nbRotations(); ++i)
        rotation(i).write(f, scaleFactor);
}

size_t ezc3d::DataNS::RotationNS::Rotations::nbRotations() const {
    return _rotations.size();
}

const ezc3d::DataNS::RotationNS::Rotation&
ezc3d::DataNS::RotationNS::Rotations::rotation(
        size_t idx) const {
    try {
        return _rotations.at(idx);
    } catch(std::out_of_range) {
        throw std::out_of_range(
                    "Rotations::rotation method is trying to access the rotation "
                    + std::to_string(idx) +
                    " while the maximum number of rotations is "
                    + std::to_string(nbRotations()) + ".");
    }
}

ezc3d::DataNS::RotationNS::Rotation&
ezc3d::DataNS::RotationNS::Rotations::rotation(
        size_t idx) {
    try {
        return _rotations.at(idx);
    } catch(std::out_of_range) {
        throw std::out_of_range(
                    "Rotations::rotation method is trying to access the rotation "
                    + std::to_string(idx) +
                    " while the maximum number of rotations is "
                    + std::to_string(nbRotations()) + ".");
    }
}

void ezc3d::DataNS::RotationNS::Rotations::rotation(
        const ezc3d::DataNS::RotationNS::Rotation &rotation,
        size_t idx) {
    if (idx == SIZE_MAX) {
        _rotations.push_back(rotation);
    }
    else {
        if (idx >= nbRotations()) {
            _rotations.resize(idx+1);
        }
        _rotations[idx] = rotation;
    }
}

const std::vector<ezc3d::DataNS::RotationNS::Rotation>&
ezc3d::DataNS::RotationNS::Rotations::rotations() const {
    return _rotations;
}

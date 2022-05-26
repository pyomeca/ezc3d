#define EZC3D_API_EXPORTS
///
/// \file Rotations.cpp
/// \brief Implementation of Rotations class
/// \author Pariterre
/// \version 1.0
/// \date April 30th, 2022
///

#include "ezc3d/Rotations.h"
#include "ezc3d/ezc3d.h"
#include "ezc3d/Header.h"
#include "ezc3d/Parameters.h"
#include "ezc3d/RotationsInfo.h"
#include "ezc3d/RotationsSubframe.h"
#include <iostream>
#include <stdexcept>

// Rotations data
ezc3d::DataNS::RotationNS::Rotations::Rotations()
{

}

ezc3d::DataNS::RotationNS::Rotations::Rotations(
        ezc3d::c3d &c3d,
        std::fstream &file,
        const ezc3d::DataNS::RotationNS::Info& info)
{
    if (!c3d.header().hasRotationalData())
        return;

    size_t nbSubframes = info.ratio();
    for (size_t k = 0; k < nbSubframes; ++k){
        subframe(ezc3d::DataNS::RotationNS::SubFrame(c3d, file, info), k);
    }
}

void ezc3d::DataNS::RotationNS::Rotations::print() const {
    for (size_t i = 0; i < nbSubframes(); ++i){
        std::cout << "Subframe = " << i << "\n";
        subframe(i).print();
        std::cout << "\n";
    }
}

void ezc3d::DataNS::RotationNS::Rotations::write(
        std::fstream &f) const {
    for (size_t i = 0; i < nbSubframes(); ++i) {
        subframe(i).write(f);
    }
}

size_t ezc3d::DataNS::RotationNS::Rotations::nbSubframes() const {
    return _subframe.size();
}

void ezc3d::DataNS::RotationNS::Rotations::nbSubframes(
        size_t nbSubframes) {
    _subframe.resize(nbSubframes);
}

const ezc3d::DataNS::RotationNS::SubFrame&
ezc3d::DataNS::RotationNS::Rotations::subframe(
        size_t idx) const {
    try {
        return _subframe.at(idx);
    } catch(std::out_of_range) {
        throw std::out_of_range("Analogs::subframe method is trying to access the subframe "
                                + std::to_string(idx) +
                                " while the maximum number of subframes is "
                                + std::to_string(nbSubframes()) + ".");
    }
}

ezc3d::DataNS::RotationNS::SubFrame&
ezc3d::DataNS::RotationNS::Rotations::subframe(
        size_t idx) {
    try {
        return _subframe.at(idx);
    } catch(std::out_of_range) {
        throw std::out_of_range("Analogs::subframe method is trying to access the subframe "
                                + std::to_string(idx) +
                                " while the maximum number of subframes is "
                                + std::to_string(nbSubframes()) + ".");
    }
}

void ezc3d::DataNS::RotationNS::Rotations::subframe(
        const ezc3d::DataNS::RotationNS::SubFrame& subframe,
        size_t idx) {
    if (idx == SIZE_MAX) {
        _subframe.push_back(subframe);
    }
    else {
        if (idx >= nbSubframes()) {
            _subframe.resize(idx+1);
        }
        _subframe[idx] = subframe;
    }
}

const std::vector<ezc3d::DataNS::RotationNS::SubFrame>&
ezc3d::DataNS::RotationNS::Rotations::subframes() const {
    return _subframe;
}

bool ezc3d::DataNS::RotationNS::Rotations::isEmpty() const {
    for (SubFrame subframe : subframes()) {
        if (!subframe.isEmpty()) {
            return false;
        }
    }
    return true;
}

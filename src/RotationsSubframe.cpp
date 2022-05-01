#define EZC3D_API_EXPORTS
///
/// \file RotationsSubframe.cpp
/// \brief Implementation of Subframe class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "RotationsSubframe.h"
#include "Header.h"
#include "RotationsInfo.h"

ezc3d::DataNS::RotationNS::SubFrame::SubFrame() {

}

ezc3d::DataNS::RotationNS::SubFrame::SubFrame(
        ezc3d::c3d &c3d,
        std::fstream &file,
        const ezc3d::DataNS::RotationNS::Info &info)
{
    nbRotations(info.used());

    // Read the rotations
    for (size_t i = 0; i < nbRotations(); ++i){
        // Scale -1 is mandatory (Float)
        double elem00 = c3d.readFloat(info.processorType(), file);
        double elem10 = c3d.readFloat(info.processorType(), file);
        double elem20 = c3d.readFloat(info.processorType(), file);
        double elem30 = c3d.readFloat(info.processorType(), file);
        double elem01 = c3d.readFloat(info.processorType(), file);
        double elem11 = c3d.readFloat(info.processorType(), file);
        double elem21 = c3d.readFloat(info.processorType(), file);
        double elem31 = c3d.readFloat(info.processorType(), file);
        double elem02 = c3d.readFloat(info.processorType(), file);
        double elem12 = c3d.readFloat(info.processorType(), file);
        double elem22 = c3d.readFloat(info.processorType(), file);
        double elem32 = c3d.readFloat(info.processorType(), file);
        double elem03 = c3d.readFloat(info.processorType(), file);
        double elem13 = c3d.readFloat(info.processorType(), file);
        double elem23 = c3d.readFloat(info.processorType(), file);
        double elem33 = c3d.readFloat(info.processorType(), file);
        double reliability = c3d.readFloat(info.processorType(), file);

        rotation(ezc3d::DataNS::RotationNS::Rotation(
                    elem00, elem01, elem02, elem03,
                    elem10, elem11, elem12, elem13,
                    elem20, elem21, elem22, elem23,
                    elem30, elem31, elem32, elem33,
                    reliability), i);
    }
}

void ezc3d::DataNS::RotationNS::SubFrame::print() const {
    for (size_t j = 0; j < nbRotations(); ++j){
        std::cout << "Rotation: " << j << "\n";
        rotation(j).print();
    }
}

void ezc3d::DataNS::RotationNS::SubFrame::write(
        std::fstream &f) const
{
    for (size_t i = 0; i < nbRotations(); ++i){
        rotation(i).write(f);
    }
}

size_t ezc3d::DataNS::RotationNS::SubFrame::nbRotations() const {
    return _rotations.size();
}

void ezc3d::DataNS::RotationNS::SubFrame::nbRotations(
        size_t nbRotations) {
    _rotations.resize(nbRotations);
}

const ezc3d::DataNS::RotationNS::Rotation&
ezc3d::DataNS::RotationNS::SubFrame::rotation(size_t idx) const {
    try {
        return _rotations.at(idx);
    } catch(std::out_of_range) {
        throw std::out_of_range(
                    "Subframe::rotation method is trying to access the rotation "
                    + std::to_string(idx) +
                    " while the maximum number of rotations is "
                    + std::to_string(nbRotations()) + ".");
    }
}

ezc3d::DataNS::RotationNS::Rotation&
ezc3d::DataNS::RotationNS::SubFrame::rotation(
        size_t idx) {
    try {
        return _rotations.at(idx);
    } catch(std::out_of_range) {
        throw std::out_of_range(
                    "Subframe::rotation method is trying to access the rotation "
                    + std::to_string(idx) +
                    " while the maximum number of rotations is "
                    + std::to_string(nbRotations()) + ".");
    }
}

void ezc3d::DataNS::RotationNS::SubFrame::rotation(
        const ezc3d::DataNS::RotationNS::Rotation &rotation,
        size_t idx) {
    if (idx == SIZE_MAX)
        _rotations.push_back(rotation);
    else{
        if (idx >= nbRotations())
            _rotations.resize(idx+1);
        _rotations[idx] = rotation;
    }
}

const std::vector<ezc3d::DataNS::RotationNS::Rotation>&
ezc3d::DataNS::RotationNS::SubFrame::rotations() const {
    return _rotations;
}

bool ezc3d::DataNS::RotationNS::SubFrame::isEmpty() const {
    for (Rotation rotation : rotations())
        if (!rotation.isEmpty())
            return false;
    return true;
}

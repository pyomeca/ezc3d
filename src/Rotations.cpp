#define EZC3D_API_EXPORTS
///
/// \file Rotations.cpp
/// \brief Implementation of Rotations class
/// \author Pariterre
/// \version 1.0
/// \date April 30th, 2022
///

#include "Header.h"
#include "Parameters.h"
#include "Rotations.h"

// Rotations data
ezc3d::DataNS::RotationNS::Rotations::Rotations() {

}

ezc3d::DataNS::RotationNS::Rotations::Rotations(
        ezc3d::c3d &c3d,
        std::fstream &file) {
    if (!c3d.header().hasRotationalData())
        return;

    // Do a sanity check
    auto& group = c3d.parameters().group("ROTATION");
    if (!group.isParameter("USED")){
        throw std::runtime_error("USED is not present in ROTATION.");
    }
    if (!group.isParameter("DATA_START")){
        throw std::runtime_error("DATA_START is not present in ROTATION.");
    }
    if (!group.isParameter("FRAMES")){
        throw std::runtime_error("FRAMES is not present in ROTATION.");
    }
    if (!group.isParameter("RATIO") && !group.isParameter("RATE")){
        throw std::runtime_error("RATIO or RATE must be present in ROTATION.");
    }
    if (!group.isParameter("LABELS")){
        throw std::runtime_error("LABELS is not present in ROTATION.");
    }

    // Prepare the reading
    group.isParameter("DATA_START");
    file.seekg(static_cast<int>(c3d.header().dataStart()-1)*512, std::ios::beg);
    PROCESSOR_TYPE processorType(c3d.parameters().processorType());

    std::vector<std::string> names = c3d.parameters()
            .group("ROTATION").parameter("LABELS").valuesAsString();
    int ratio = group.isParameter("RATIO") ?
                group.isParameter("RATIO") :
                group.isParameter("RATE") / c3d.header().frameRate();
    _nbRotations = group.parameter("USED").valuesAsInt()[0];
    size_t nbFrames = c3d.header().nbFrames() * ratio;

    // Read the data
    _rotations.resize(nbFrames);
    for (size_t j = 0; j < nbFrames; ++j){
        if (file.eof())
            break;

        _rotations[j].resize(nbRotations());
        // Read the rotations
        for (size_t i = 0; i < nbRotations(); ++i){
            // Scale -1 is mandatory (Float)
            double elem00 = c3d.readFloat(processorType, file);
            double elem10 = c3d.readFloat(processorType, file);
            double elem20 = c3d.readFloat(processorType, file);
            double elem30 = c3d.readFloat(processorType, file);
            double elem01 = c3d.readFloat(processorType, file);
            double elem11 = c3d.readFloat(processorType, file);
            double elem21 = c3d.readFloat(processorType, file);
            double elem31 = c3d.readFloat(processorType, file);
            double elem02 = c3d.readFloat(processorType, file);
            double elem12 = c3d.readFloat(processorType, file);
            double elem22 = c3d.readFloat(processorType, file);
            double elem32 = c3d.readFloat(processorType, file);
            double elem03 = c3d.readFloat(processorType, file);
            double elem13 = c3d.readFloat(processorType, file);
            double elem23 = c3d.readFloat(processorType, file);
            double elem33 = c3d.readFloat(processorType, file);
            double residual = c3d.readFloat(processorType, file);

            _rotations[j][i] = ezc3d::DataNS::RotationNS::Rotation(
                        elem00, elem01, elem02, elem03,
                        elem10, elem11, elem12, elem13,
                        elem20, elem21, elem22, elem23,
                        elem30, elem31, elem32, elem33,
                        residual);

        }
    }
}

void ezc3d::DataNS::RotationNS::Rotations::print() const {
    for (size_t i = 0; i < nbFrames(); ++i){
        for (size_t j = 0; j < nbRotations(); ++j){
            rotation(i, j).print();
        }
    }
}

void ezc3d::DataNS::RotationNS::Rotations::write(
        std::fstream &f,
        float scaleFactor) const {
    for (size_t i = 0; i < nbFrames(); ++i){
        for (size_t j = 0; j < nbRotations(); ++j){
            rotation(i, j).write(f, scaleFactor);
        }
    }
}

size_t ezc3d::DataNS::RotationNS::Rotations::nbFrames() const {
    return _rotations.size();
}

size_t ezc3d::DataNS::RotationNS::Rotations::nbRotations() const
{
    return _nbRotations;
}

const ezc3d::DataNS::RotationNS::Rotation&
ezc3d::DataNS::RotationNS::Rotations::rotation(
        size_t frame,
        size_t idx) const {
    try {
        return _rotations.at(frame).at(idx);
    } catch(std::out_of_range) {
        throw std::out_of_range(
                    "Rotations::rotation method is trying to access the frame "
                    + std::to_string(frame) + " rotation " + std::to_string(idx) +
                    " while the maximum number of frames is "
                    + std::to_string(nbFrames()) + " and rotations is " +
                    std::to_string(_nbRotations) + ".");
    }
}

ezc3d::DataNS::RotationNS::Rotation&
ezc3d::DataNS::RotationNS::Rotations::rotation(
        size_t frame,
        size_t idx) {
    try {
        return _rotations.at(frame).at(idx);
    } catch(std::out_of_range) {
        throw std::out_of_range(
                    "Rotations::rotation method is trying to access the frame "
                    + std::to_string(frame) + " rotation " + std::to_string(idx) +
                    " while the maximum number of frames is "
                    + std::to_string(nbFrames()) + " and rotations is " +
                    std::to_string(_nbRotations) + ".");
    }
}

const std::vector<ezc3d::DataNS::RotationNS::Rotation>&
ezc3d::DataNS::RotationNS::Rotations::rotations(size_t frame) const {
    return _rotations.at(frame);
}

#define EZC3D_API_EXPORTS
///
/// \file Data.cpp
/// \brief Implementation of Data class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "Data.h"
#include "ezc3d.h"
#include "Header.h"
#include "Parameters.h"
#include "AnalogsInfo.h"
#include "PointsInfo.h"
#include "RotationsInfo.h"
#include "DataStartInfo.h"
#include <iostream>
#include <stdexcept>

ezc3d::DataNS::Data::Data() {
}

ezc3d::DataNS::Data::Data(
        ezc3d::c3d &c3d, std::fstream &file) {
    // Firstly move the pointer to the data start position
    file.seekg(static_cast<int>(c3d.header().dataStart()-1)*512, std::ios::beg);

    // Read the data
    ezc3d::DataNS::Points3dNS::Info pointsInfo(c3d);
    ezc3d::DataNS::AnalogsNS::Info analogsInfo(c3d);
    ezc3d::DataNS::RotationNS::Info rotationsInfo(c3d);

    for (size_t j = 0; j < c3d.header().nbFrames(); ++j){
        if (file.eof())
            break;

        ezc3d::DataNS::Frame f;
        // Read point 3d
        f.add(ezc3d::DataNS::Points3dNS::Points(c3d, file, pointsInfo));

        // Read analogs
        f.add(ezc3d::DataNS::AnalogsNS::Analogs(c3d, file, analogsInfo));
        _frames.push_back(f);
    }

    // Read the rotation data
    if (c3d.header().hasRotationalData()){
        // Prepare the reading
        file.seekg(static_cast<int>(rotationsInfo.dataStart()-1)*512, std::ios::beg);

        for (size_t i = 0; i < c3d.header().nbFrames(); ++i){
            if (file.eof())
                break;

            _frames[i].add(ezc3d::DataNS::RotationNS::Rotations(c3d, file, rotationsInfo));
        }
    }
}

void ezc3d::DataNS::Data::print() const {
    for (size_t i = 0; i < nbFrames(); ++i){
        std::cout << "Frame " << i << "\n";
        frame(i).print();
        std::cout << "\n";
    }
}

void ezc3d::DataNS::Data::write(
        const ezc3d::Header& header,
        std::fstream &f,
        float pointScaleFactor,
        std::vector<double> analogScaleFactors,
        ezc3d::DataStartInfo& dataStartInfoToFill) const {

    dataStartInfoToFill.setPointDataStart(f.tellg());
    for (size_t i = 0; i < nbFrames(); ++i)
        frame(i).write(f, pointScaleFactor, analogScaleFactors, 0);

    if (header.hasRotationalData()){
        ezc3d::c3d::moveCursorToANewBlock(f);
        dataStartInfoToFill.setRotationsDataStart(f.tellg());
        for (size_t i = 0; i < nbFrames(); ++i)
            frame(i).write(f, pointScaleFactor, analogScaleFactors, 1);
    }
}

size_t ezc3d::DataNS::Data::nbFrames() const {
    return _frames.size();
}

const ezc3d::DataNS::Frame& ezc3d::DataNS::Data::frame(
        size_t idx) const {
    try {
        return _frames.at(idx);
    } catch(std::out_of_range) {
        throw std::out_of_range(
                    "Data::frame method is trying to access the frame "
                    + std::to_string(idx) +
                    " while the maximum number of frame is "
                    + std::to_string(nbFrames()) + ".");
    }
}

ezc3d::DataNS::Frame& ezc3d::DataNS::Data::frame(
        size_t idx) {
    try {
        return _frames.at(idx);
    } catch(std::out_of_range) {
        throw std::out_of_range(
                    "Data::frame method is trying to access the frame "
                    + std::to_string(idx) +
                    " while the maximum number of frames is "
                    + std::to_string(nbFrames()) + ".");
    }
}

void ezc3d::DataNS::Data::frame(
        const ezc3d::DataNS::Frame &frame,
        size_t idx) {
    if (idx == SIZE_MAX)
        _frames.push_back(frame);
    else {
        if (idx >= _frames.size())
            _frames.resize(idx+1);
        _frames[idx].add(frame);
    }
}

const std::vector<ezc3d::DataNS::Frame> &ezc3d::DataNS::Data::frames() const {
    return _frames;
}



#define EZC3D_API_EXPORTS
///
/// \file Data.cpp
/// \brief Implementation of Data class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "Data.h"
#include "Header.h"
#include "Parameters.h"

ezc3d::DataNS::Data::Data() {
}

ezc3d::DataNS::Data::Data(
        ezc3d::c3d &c3d, std::fstream &file) {
    // Firstly move the pointer to the data start position
    file.seekg(static_cast<int>(c3d.header().dataStart()-1)*512, std::ios::beg);

    // Get names of the data
    std::vector<std::string> pointNames;
    if (c3d.header().nb3dPoints() > 0)
        pointNames = c3d.parameters()
                .group("POINT").parameter("LABELS")
                .valuesAsString();
    std::vector<std::string> analogNames;
    if (c3d.header().nbAnalogs() > 0)
        analogNames = c3d.parameters()
                .group("ANALOG").parameter("LABELS")
                .valuesAsString();

    // Read the data
    PROCESSOR_TYPE processorType(c3d.parameters().processorType());
    double pointScaleFactor(-1);
    if (c3d.header().nb3dPoints())
        pointScaleFactor = c3d.parameters()
                .group("POINT").parameter("SCALE")
                .valuesAsDouble()[0];
    std::vector<double> analogScaleFactors;
    if (c3d.header().nbAnalogs())
        analogScaleFactors = c3d.parameters()
                .group("ANALOG").parameter("SCALE")
                .valuesAsDouble();
    double analogGeneralFactor(c3d.parameters()
                              .group("ANALOG").parameter("GEN_SCALE")
                              .valuesAsDouble()[0]);
    std::vector<int> analogZeroOffset(
                c3d.parameters()
                .group("ANALOG").parameter("OFFSET")
                .valuesAsInt());
    for (size_t j = 0; j < c3d.header().nbFrames(); ++j){
        if (file.eof())
            break;

        ezc3d::DataNS::Frame f;
        // Read point 3d
        ezc3d::DataNS::Points3dNS::Points ptsAtAFrame(
                    c3d.header().nb3dPoints());
        for (size_t i = 0; i < c3d.header().nb3dPoints(); ++i){
            ezc3d::DataNS::Points3dNS::Point pt;
            if (c3d.header().scaleFactor() < 0){ // if it is float
                pt.x(c3d.readFloat(processorType, file));
                pt.y(c3d.readFloat(processorType, file));
                pt.z(c3d.readFloat(processorType, file));
                if (processorType == PROCESSOR_TYPE::INTEL){
                    pt.cameraMask(c3d.readInt(
                                      processorType, file, ezc3d::DATA_TYPE::WORD));
                    pt.residual(static_cast<float>(c3d.readInt(
                                    processorType, file, ezc3d::DATA_TYPE::WORD))
                                * -pointScaleFactor);
                }
                else if (processorType == PROCESSOR_TYPE::DEC){
                    pt.residual(static_cast<float>(c3d.readInt(
                                    processorType, file, ezc3d::DATA_TYPE::WORD))
                                * -pointScaleFactor);
                    pt.cameraMask(c3d.readInt(
                                      processorType, file, ezc3d::DATA_TYPE::WORD));
                }
                else if (processorType == PROCESSOR_TYPE::MIPS){
                    throw std::runtime_error(
                                "MIPS processor type not supported yet, please open a "
                                "GitHub issue to report that you want this feature!");
                }
            } else {
                pt.x(static_cast<float>(
                         c3d.readInt(
                             processorType, file, ezc3d::DATA_TYPE::WORD))
                     * pointScaleFactor);
                pt.y(static_cast<float>(
                         c3d.readInt(
                             processorType, file, ezc3d::DATA_TYPE::WORD))
                     * pointScaleFactor);
                pt.z(static_cast<float>(
                         c3d.readInt(
                             processorType, file, ezc3d::DATA_TYPE::WORD))
                     * pointScaleFactor);
                if (processorType == PROCESSOR_TYPE::INTEL){
                    pt.cameraMask(c3d.readInt(
                                      processorType, file, ezc3d::DATA_TYPE::BYTE));
                    pt.residual(static_cast<float>(
                                    c3d.readInt(processorType,
                                                file, ezc3d::DATA_TYPE::BYTE))
                                * pointScaleFactor);
                }
                else if (processorType == PROCESSOR_TYPE::DEC){
                    pt.cameraMask(c3d.readInt(
                                      processorType, file, ezc3d::DATA_TYPE::BYTE));
                    pt.residual(static_cast<float>(
                                    c3d.readInt(processorType,
                                                file, ezc3d::DATA_TYPE::BYTE))
                                * pointScaleFactor);
                }
                else if (processorType == PROCESSOR_TYPE::MIPS){
                    throw std::runtime_error(
                                "MIPS processor type not supported yet, please open a "
                                "GitHub issue to report that you want this feature!");
                }
            }
            if (pt.residual() < 0){
                pt.set(NAN, NAN, NAN);
            }
            ptsAtAFrame.point(pt, i);
        }
        // modified by pts_tp which is an nonconst ref to internal points
        f.add(ptsAtAFrame);

        // Read analogs
        ezc3d::DataNS::AnalogsNS::Analogs analog;
        analog.nbSubframes(c3d.header().nbAnalogByFrame());
        for (size_t k = 0; k < c3d.header().nbAnalogByFrame(); ++k){
            ezc3d::DataNS::AnalogsNS::SubFrame sub;
            sub.nbChannels(c3d.header().nbAnalogs());
            for (size_t i = 0; i < c3d.header().nbAnalogs(); ++i){
                ezc3d::DataNS::AnalogsNS::Channel c;
                if (c3d.header().scaleFactor() < 0) // if it is float
                    c.data( (c3d.readFloat(processorType, file)
                             - analogZeroOffset[i])
                            *  analogScaleFactors[i] * analogGeneralFactor );
                else
                    c.data( (static_cast<float>(
                                 c3d.readInt(processorType, file,
                                             ezc3d::DATA_TYPE::WORD))
                             - analogZeroOffset[i])
                            *  analogScaleFactors[i] * analogGeneralFactor );
                sub.channel(c, i);
            }
            analog.subframe(sub, k);
        }
        f.add(analog);
        _frames.push_back(f);
    }

    // remove the trailing empty frames if they exist
    size_t nFrames(_frames.size());
    if (nFrames > 0)
        for (size_t i=0; i<nFrames-1; i--){
            // -1 so we at least keep one frame if frames are empty
            if (_frames.back().isEmpty())
                _frames.pop_back();
            else
                break;
        }
}

void ezc3d::DataNS::Data::print() const {
    for (size_t i = 0; i < nbFrames(); ++i){
        std::cout << "Frame " << i << std::endl;
        frame(i).print();
        std::cout << std::endl;
    }
}

void ezc3d::DataNS::Data::write(
        std::fstream &f,
        float pointScaleFactor,
        std::vector<double> analogScaleFactors) const {
    for (size_t i = 0; i < nbFrames(); ++i)
        frame(i).write(f, pointScaleFactor, analogScaleFactors);
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



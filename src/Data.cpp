#define EZC3D_API_EXPORTS
///
/// \file Data.cpp
/// \brief Implementation of Data class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "Data.h"

ezc3d::DataNS::Data::Data()
{

}

ezc3d::DataNS::Data::Data(ezc3d::c3d &file)
{
    // Firstly read a dummy value just prior to the data so it moves the pointer to the right place
    file.readInt(ezc3d::DATA_TYPE::BYTE,
                 static_cast<int>(256*ezc3d::DATA_TYPE::WORD*(file.header().parametersAddress()-1) + file.header().nbOfZerosBeforeHeader() +
                 256*ezc3d::DATA_TYPE::WORD*file.parameters().nbParamBlock() -
                 ezc3d::DATA_TYPE::BYTE), std::ios::beg); // "- BYTE" so it is just prior

    // Get names of the data
    std::vector<std::string> pointNames;
    if (file.header().nb3dPoints() > 0)
        pointNames = file.parameters().group("POINT").parameter("LABELS").valuesAsString();
    std::vector<std::string> analogNames;
    if (file.header().nbAnalogs() > 0)
        analogNames = file.parameters().group("ANALOG").parameter("LABELS").valuesAsString();

    // Read the actual data
    for (size_t j = 0; j < file.header().nbFrames(); ++j){
        if (file.eof())
            break;

        ezc3d::DataNS::Frame f;
        if (file.header().scaleFactor() < 0){ // if it is float
            // Read point 3d
            ezc3d::DataNS::Points3dNS::Points ptsAtAFrame(file.header().nb3dPoints());
            for (size_t i = 0; i < file.header().nb3dPoints(); ++i){
                ezc3d::DataNS::Points3dNS::Point pt;
                pt.x(file.readFloat());
                pt.y(file.readFloat());
                pt.z(file.readFloat());
                pt.residual(file.readFloat());
                if (i < pointNames.size())
                    pt.name(pointNames[i]);
                else {
                    std::stringstream unlabel;
                    unlabel << "unlabeled_point_" << i;
                    pt.name(unlabel.str());
                }
                ptsAtAFrame.point(pt, i);
            }
            f.add(ptsAtAFrame); // modified by pts_tp which is an nonconst ref to internal points

            // Read analogs
            ezc3d::DataNS::AnalogsNS::Analogs analog;
            analog.nbSubframes(file.header().nbAnalogByFrame());
            for (size_t k = 0; k < file.header().nbAnalogByFrame(); ++k){
                ezc3d::DataNS::AnalogsNS::SubFrame sub;
                sub.nbChannels(file.header().nbAnalogs());
                for (size_t i = 0; i < file.header().nbAnalogs(); ++i){
                    ezc3d::DataNS::AnalogsNS::Channel c;
                    c.data(file.readFloat());
                    if (i < analogNames.size())
                        c.name(analogNames[i]);
                    else {
                        std::stringstream unlabel;
                        unlabel << "unlabeled_analog_" << i;
                        c.name(unlabel.str());
                    }
                    sub.channel(c, i);
                }
                analog.subframe(sub, k);
            }
            f.add(analog);
            _frames.push_back(f);
        }
        else
            throw std::invalid_argument("Points were recorded using int number which is not implemented yet");
    }

    // remove the trailing empty frames if they exist
    size_t nFrames(_frames.size());
    for (size_t i=0; i<nFrames-1; i--){ // -1 so we at least keep one frame if frames are empty
        if (_frames.back().isempty())
            _frames.pop_back();
        else
            break;
    }
}

void ezc3d::DataNS::Data::print() const
{
    for (size_t i = 0; i < nbFrames(); ++i){
        std::cout << "Frame " << i << std::endl;
        frame(i).print();
        std::cout << std::endl;
    }
}

void ezc3d::DataNS::Data::write(std::fstream &f) const
{
    for (size_t i = 0; i < nbFrames(); ++i)
        frame(i).write(f);
}

size_t ezc3d::DataNS::Data::nbFrames() const
{
    return _frames.size();
}

const ezc3d::DataNS::Frame& ezc3d::DataNS::Data::frame(size_t idx) const
{
    try {
        return _frames.at(idx);
    } catch(std::out_of_range) {
        throw std::out_of_range("Data::frame method is trying to access the frame "
                                + std::to_string(idx) +
                                " while the maximum number of frame is "
                                + std::to_string(nbFrames()) + ".");
    }
}

ezc3d::DataNS::Frame &ezc3d::DataNS::Data::frame_nonConst(size_t idx)
{
    try {
        return _frames.at(idx);
    } catch(std::out_of_range) {
        throw std::out_of_range("Data::frame method is trying to access the frame "
                                + std::to_string(idx) +
                                " while the maximum number of frames is "
                                + std::to_string(nbFrames()) + ".");
    }
}

void ezc3d::DataNS::Data::frame(const ezc3d::DataNS::Frame &frame, size_t idx)
{
    if (idx == SIZE_MAX)
        _frames.push_back(frame);
    else {
        if (idx >= _frames.size())
            _frames.resize(idx+1);
        _frames[idx].add(frame);
    }
}

const std::vector<ezc3d::DataNS::Frame> &ezc3d::DataNS::Data::frames() const
{
    return _frames;
}



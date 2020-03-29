#define EZC3D_API_EXPORTS
///
/// \file Frame.cpp
/// \brief Implementation of Frame class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "Frame.h"

ezc3d::DataNS::Frame::Frame() {
    _points = std::shared_ptr<ezc3d::DataNS::Points3dNS::Points>(
                new ezc3d::DataNS::Points3dNS::Points());
    _analogs = std::shared_ptr<ezc3d::DataNS::AnalogsNS::Analogs>(
                new ezc3d::DataNS::AnalogsNS::Analogs());
}

void ezc3d::DataNS::Frame::print() const {
    points().print();
    analogs().print();
}

void ezc3d::DataNS::Frame::write(
        std::fstream &f,
        float pointScaleFactor,
        std::vector<double> analogScaleFactors) const {
    points().write(f, pointScaleFactor);
    analogs().write(f, analogScaleFactors);
}

const ezc3d::DataNS::Points3dNS::Points& ezc3d::DataNS::Frame::points() const {
    return *_points;
}

ezc3d::DataNS::Points3dNS::Points &ezc3d::DataNS::Frame::points() {
    return *_points;
}

const ezc3d::DataNS::AnalogsNS::Analogs& ezc3d::DataNS::Frame::analogs() const {
    return *_analogs;
}

ezc3d::DataNS::AnalogsNS::Analogs &ezc3d::DataNS::Frame::analogs() {
    return *_analogs;
}

void ezc3d::DataNS::Frame::add(
        const ezc3d::DataNS::Frame &frame) {
    add(frame.points(), frame.analogs());
}

void ezc3d::DataNS::Frame::add(
        const ezc3d::DataNS::Points3dNS::Points &point3d_frame) {
    _points = std::shared_ptr<ezc3d::DataNS::Points3dNS::Points>(
                new ezc3d::DataNS::Points3dNS::Points(point3d_frame));
}

void ezc3d::DataNS::Frame::add(
        const ezc3d::DataNS::AnalogsNS::Analogs &analogs_frame) {
    _analogs = std::shared_ptr<ezc3d::DataNS::AnalogsNS::Analogs>(
                new ezc3d::DataNS::AnalogsNS::Analogs(analogs_frame));
}

void ezc3d::DataNS::Frame::add(
        const ezc3d::DataNS::Points3dNS::Points &point3d_frame,
        const ezc3d::DataNS::AnalogsNS::Analogs &analog_frame) {
    add(point3d_frame);
    add(analog_frame);
}

bool ezc3d::DataNS::Frame::isEmpty() const {
    return points().isEmpty() && analogs().isEmpty();
}

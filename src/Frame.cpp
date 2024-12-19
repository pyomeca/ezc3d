#define EZC3D_API_EXPORTS
///
/// \file Frame.cpp
/// \brief Implementation of Frame class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "ezc3d/Frame.h"
#include "ezc3d/DataStartInfo.h"
#include <stdexcept>

ezc3d::DataNS::Frame::Frame() {
  _points = std::shared_ptr<ezc3d::DataNS::Points3dNS::Points>(
      new ezc3d::DataNS::Points3dNS::Points());
  _analogs = std::shared_ptr<ezc3d::DataNS::AnalogsNS::Analogs>(
      new ezc3d::DataNS::AnalogsNS::Analogs());
  _rotations = std::shared_ptr<ezc3d::DataNS::RotationNS::Rotations>(
      new ezc3d::DataNS::RotationNS::Rotations());
}

void ezc3d::DataNS::Frame::print() const {
  points().print();
  analogs().print();
  rotations().print();
}

void ezc3d::DataNS::Frame::write(std::fstream &f,
                                 std::vector<double> pointScaleFactor,
                                 std::vector<double> analogScaleFactors,
                                 int dataTypeToWrite) const {
  if (dataTypeToWrite == 0) { // Points and analogs
    points().write(f, pointScaleFactor);
    analogs().write(f, analogScaleFactors);
  } else if (dataTypeToWrite == 1) { // Rotations
    rotations().write(f);
  } else {
    throw std::runtime_error("Data type not implemented yet");
  }
}

const ezc3d::DataNS::Points3dNS::Points &ezc3d::DataNS::Frame::points() const {
  return *_points;
}

ezc3d::DataNS::Points3dNS::Points &ezc3d::DataNS::Frame::points() {
  return *_points;
}

const ezc3d::DataNS::AnalogsNS::Analogs &ezc3d::DataNS::Frame::analogs() const {
  return *_analogs;
}

ezc3d::DataNS::AnalogsNS::Analogs &ezc3d::DataNS::Frame::analogs() {
  return *_analogs;
}

const ezc3d::DataNS::RotationNS::Rotations &
ezc3d::DataNS::Frame::rotations() const {
  return *_rotations;
}

ezc3d::DataNS::RotationNS::Rotations &ezc3d::DataNS::Frame::rotations() {
  return *_rotations;
}

void ezc3d::DataNS::Frame::add(const ezc3d::DataNS::Frame &frame) {
  add(frame.points(), frame.analogs(), frame.rotations());
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
    const ezc3d::DataNS::RotationNS::Rotations &rotations) {
  _rotations = std::shared_ptr<ezc3d::DataNS::RotationNS::Rotations>(
      new ezc3d::DataNS::RotationNS::Rotations(rotations));
}

void ezc3d::DataNS::Frame::add(
    const ezc3d::DataNS::Points3dNS::Points &point3d_frame,
    const ezc3d::DataNS::AnalogsNS::Analogs &analog_frame) {
  add(point3d_frame);
  add(analog_frame);
}

void ezc3d::DataNS::Frame::add(
    const ezc3d::DataNS::Points3dNS::Points &points,
    const ezc3d::DataNS::AnalogsNS::Analogs &analogs,
    const ezc3d::DataNS::RotationNS::Rotations &rotations) {
  add(points);
  add(analogs);
  add(rotations);
}

bool ezc3d::DataNS::Frame::isEmpty() const {
  return points().isEmpty() && analogs().isEmpty();
}

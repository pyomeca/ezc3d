#define EZC3D_API_EXPORTS
///
/// \file Data.cpp
/// \brief Implementation of Data class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "ezc3d/Data.h"
#include "ezc3d/AnalogsInfo.h"
#include "ezc3d/DataStartInfo.h"
#include "ezc3d/Header.h"
#include "ezc3d/Parameters.h"
#include "ezc3d/PointsInfo.h"
#include "ezc3d/RotationsInfo.h"
#include "ezc3d/ezc3d.h"
#include <iostream>
#include <stdexcept>

ezc3d::DataNS::Data::Data() {}

ezc3d::DataNS::Data::Data(ezc3d::c3d &c3d, std::fstream &file) {
  // Firstly move the pointer to the data start position
  file.seekg(static_cast<int>(c3d.header().dataStart() - 1) * 512,
             std::ios::beg);

  // Read the data
  ezc3d::DataNS::Points3dNS::Info pointsInfo(c3d);
  ezc3d::DataNS::AnalogsNS::Info analogsInfo(c3d);
  ezc3d::DataNS::RotationNS::Info rotationsInfo(c3d);

  for (size_t j = 0; j < c3d.header().nbFrames(); ++j) {
    ezc3d::DataNS::Frame f;
    // Read point 3d
    f.add(ezc3d::DataNS::Points3dNS::Points(c3d, file, pointsInfo));

    // Read analogs
    f.add(ezc3d::DataNS::AnalogsNS::Analogs(c3d, file, analogsInfo));

    // If we ran out of space, then leave. The reason we test here is because
    // is set after failing, resulting in one extra frame added if this if
    // is after the push_back
    if (file.eof())
      break;

    _frames.push_back(f);
  }

  // Read the rotation data
  if (c3d.header().hasRotationalData()) {
    // Prepare the reading

    // If the max length of the file is smaller than the data start, then there
    // is no data
    std::streampos fileSize = file.seekg(0, std::ios::end).tellg();
    int targetPos(static_cast<int>(rotationsInfo.dataStart() - 1) * 512);
    if (fileSize < targetPos) {
      return;
    }
    file.seekg(targetPos, std::ios::beg);

    for (size_t i = 0; i < c3d.header().nbFrames(); ++i) {
      if (file.eof())
        break;

      _frames[i].add(
          ezc3d::DataNS::RotationNS::Rotations(c3d, file, rotationsInfo));
    }
  }
}

void ezc3d::DataNS::Data::print() const {
  for (size_t i = 0; i < nbFrames(); ++i) {
    std::cout << "Frame " << i << "\n";
    frame(i).print();
    std::cout << "\n";
  }
}

void ezc3d::DataNS::Data::write(
    const ezc3d::Header &header, std::fstream &f,
    std::vector<double> pointScaleFactor,
    std::vector<double> analogScaleFactors,
    ezc3d::DataStartInfo &dataStartInfoToFill) const {

  dataStartInfoToFill.setPointDataStart(f.tellg());
  for (size_t i = 0; i < nbFrames(); ++i)
    frame(i).write(f, pointScaleFactor, analogScaleFactors, 0);

  if (header.hasRotationalData()) {
    ezc3d::c3d::moveCursorToANewBlock(f);
    dataStartInfoToFill.setRotationsDataStart(f.tellg());
    for (size_t i = 0; i < nbFrames(); ++i)
      frame(i).write(f, pointScaleFactor, analogScaleFactors, 1);
  }
}

size_t ezc3d::DataNS::Data::nbFrames() const { return _frames.size(); }

const ezc3d::DataNS::Frame &ezc3d::DataNS::Data::frame(size_t idx) const {
  try {
    return _frames.at(idx);
  } catch (const std::out_of_range &) {
    throw std::out_of_range(
        "Data::frame method is trying to access the frame " +
        std::to_string(idx) + " while the maximum number of frame is " +
        std::to_string(nbFrames()) + ".");
  }
}

ezc3d::DataNS::Frame &ezc3d::DataNS::Data::frame(size_t idx) {
  try {
    return _frames.at(idx);
  } catch (const std::out_of_range &) {
    throw std::out_of_range(
        "Data::frame method is trying to access the frame " +
        std::to_string(idx) + " while the maximum number of frames is " +
        std::to_string(nbFrames()) + ".");
  }
}

void ezc3d::DataNS::Data::frame(const ezc3d::DataNS::Frame &frame, size_t idx) {
  if (idx == SIZE_MAX)
    _frames.push_back(frame);
  else {
    if (idx >= _frames.size())
      _frames.resize(idx + 1);
    _frames[idx].add(frame);
  }
}

const std::vector<ezc3d::DataNS::Frame> &ezc3d::DataNS::Data::frames() const {
  return _frames;
}

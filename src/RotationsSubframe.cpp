#define EZC3D_API_EXPORTS
///
/// \file RotationsSubframe.cpp
/// \brief Implementation of Subframe class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "ezc3d/RotationsSubframe.h"
#include "ezc3d/Header.h"
#include "ezc3d/RotationsInfo.h"
#include <iostream>
#ifdef _WIN32
#include <string>
#endif
#include <stdexcept>

ezc3d::DataNS::RotationNS::SubFrame::SubFrame() {}

ezc3d::DataNS::RotationNS::SubFrame::SubFrame(
    ezc3d::c3d &c3d, std::fstream &file,
    const ezc3d::DataNS::RotationNS::Info &info) {
  nbRotations(info.used());

  // Read the rotations
  for (size_t i = 0; i < nbRotations(); ++i) {
    rotation(ezc3d::DataNS::RotationNS::Rotation(c3d, file, info), i);
  }
}

void ezc3d::DataNS::RotationNS::SubFrame::print() const {
  for (size_t j = 0; j < nbRotations(); ++j) {
    std::cout << "Rotation: " << j << "\n";
    rotation(j).print();
  }
}

void ezc3d::DataNS::RotationNS::SubFrame::write(std::fstream &f) const {
  for (size_t i = 0; i < nbRotations(); ++i) {
    rotation(i).write(f);
  }
}

size_t ezc3d::DataNS::RotationNS::SubFrame::nbRotations() const {
  return _rotations.size();
}

void ezc3d::DataNS::RotationNS::SubFrame::nbRotations(size_t nbRotations) {
  _rotations.resize(nbRotations);
}

const ezc3d::DataNS::RotationNS::Rotation &
ezc3d::DataNS::RotationNS::SubFrame::rotation(size_t idx) const {
  try {
    return _rotations.at(idx);
  } catch (const std::out_of_range&) {
    throw std::out_of_range(
        "Subframe::rotation method is trying to access the rotation " +
        std::to_string(idx) + " while the maximum number of rotations is " +
        std::to_string(nbRotations()) + ".");
  }
}

ezc3d::DataNS::RotationNS::Rotation &
ezc3d::DataNS::RotationNS::SubFrame::rotation(size_t idx) {
  try {
    return _rotations.at(idx);
  } catch (const std::out_of_range&) {
    throw std::out_of_range(
        "Subframe::rotation method is trying to access the rotation " +
        std::to_string(idx) + " while the maximum number of rotations is " +
        std::to_string(nbRotations()) + ".");
  }
}

void ezc3d::DataNS::RotationNS::SubFrame::rotation(
    const ezc3d::DataNS::RotationNS::Rotation &rotation, size_t idx) {
  if (idx == SIZE_MAX)
    _rotations.push_back(rotation);
  else {
    if (idx >= nbRotations())
      _rotations.resize(idx + 1);
    _rotations[idx] = rotation;
  }
}

const std::vector<ezc3d::DataNS::RotationNS::Rotation> &
ezc3d::DataNS::RotationNS::SubFrame::rotations() const {
  return _rotations;
}

bool ezc3d::DataNS::RotationNS::SubFrame::isEmpty() const {
  for (Rotation rotation : rotations())
    if (!rotation.isEmpty())
      return false;
  return true;
}

#define EZC3D_API_EXPORTS
///
/// \file Rotation.cpp
/// \brief Implementation of Rotation class
/// \author Pariterre
/// \version 1.0
/// \date April 30th, 2022
///

#include "ezc3d/Rotation.h"
#include "ezc3d/RotationsInfo.h"
#include "ezc3d/ezc3d.h"
#include <bitset>
#include <cmath>
#include <iostream>

ezc3d::DataNS::RotationNS::Rotation::Rotation()
    : ezc3d::Matrix44(), _reliability(-1) {}

ezc3d::DataNS::RotationNS::Rotation::Rotation(
    double elem00, double elem01, double elem02, double elem03, double elem10,
    double elem11, double elem12, double elem13, double elem20, double elem21,
    double elem22, double elem23, double elem30, double elem31, double elem32,
    double elem33, double reliability)
    : ezc3d::Matrix44(elem00, elem01, elem02, elem03, elem10, elem11, elem12,
                      elem13, elem20, elem21, elem22, elem23, elem30, elem31,
                      elem32, elem33),
      _reliability(reliability) {}

ezc3d::DataNS::RotationNS::Rotation::Rotation(
    ezc3d::c3d &c3d, std::fstream &file,
    const ezc3d::DataNS::RotationNS::Info &info)
    : ezc3d::Matrix44() {
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
  set(elem00, elem01, elem02, elem03, elem10, elem11, elem12, elem13, elem20,
      elem21, elem22, elem23, elem30, elem31, elem32, elem33);
  _reliability = c3d.readFloat(info.processorType(), file);
}

ezc3d::DataNS::RotationNS::Rotation::Rotation(
    const ezc3d::DataNS::RotationNS::Rotation &r)
    : ezc3d::Matrix44(r) {
  reliability(r.reliability());
}

void ezc3d::DataNS::RotationNS::Rotation::print() const {
  for (size_t i = 0; i < _nbRows; ++i) {
    for (size_t j = 0; j < _nbCols; ++j) {
      std::cout << operator()(i, j);
      if (j != _nbCols - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "\n";
  };
  std::cout << "Reliability = " << reliability() << "\n";
}

void ezc3d::DataNS::RotationNS::Rotation::write(std::fstream &f) const {
  for (size_t i = 0; i < 16; ++i) {
    float data(static_cast<float>(_reliability < 0 ? NAN : _data[i]));
    f.write(reinterpret_cast<const char *>(&data), ezc3d::DATA_TYPE::FLOAT);
  }

  float reliability(static_cast<float>(_reliability));
  f.write(reinterpret_cast<const char *>(&reliability),
          ezc3d::DATA_TYPE::FLOAT);
}

void ezc3d::DataNS::RotationNS::Rotation::set(
    double elem00, double elem01, double elem02, double elem03, double elem10,
    double elem11, double elem12, double elem13, double elem20, double elem21,
    double elem22, double elem23, double elem30, double elem31, double elem32,
    double elem33, double reliability) {
  ezc3d::Matrix44::set(elem00, elem01, elem02, elem03, elem10, elem11, elem12,
                       elem13, elem20, elem21, elem22, elem23, elem30, elem31,
                       elem32, elem33);
  _reliability = reliability;
}

void ezc3d::DataNS::RotationNS::Rotation::set(
    double elem00, double elem01, double elem02, double elem03, double elem10,
    double elem11, double elem12, double elem13, double elem20, double elem21,
    double elem22, double elem23, double elem30, double elem31, double elem32,
    double elem33) {
  ezc3d::Matrix44::set(elem00, elem01, elem02, elem03, elem10, elem11, elem12,
                       elem13, elem20, elem21, elem22, elem23, elem30, elem31,
                       elem32, elem33);
  reliability(0);
}

double ezc3d::DataNS::RotationNS::Rotation::reliability() const {
  return _reliability;
}

void ezc3d::DataNS::RotationNS::Rotation::reliability(double reliability) {
  _reliability = reliability;
}

bool ezc3d::DataNS::RotationNS::Rotation::isValid() const {
  return _reliability < 0 ? false : true;
}

bool ezc3d::DataNS::RotationNS::Rotation::isEmpty() const { return !isValid(); }

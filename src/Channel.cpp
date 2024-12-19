#define EZC3D_API_EXPORTS
///
/// \file Channel.cpp
/// \brief Implementation of Channel class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "ezc3d/Channel.h"
#include "ezc3d/AnalogsInfo.h"
#include "ezc3d/Header.h"
#include "ezc3d/ezc3d.h"
#include <iostream>

ezc3d::DataNS::AnalogsNS::Channel::Channel() {}

ezc3d::DataNS::AnalogsNS::Channel::Channel(
    const ezc3d::DataNS::AnalogsNS::Channel &channel)
    : _data(channel._data) {}

ezc3d::DataNS::AnalogsNS::Channel::Channel(
    ezc3d::c3d &c3d, std::fstream &file,
    const ezc3d::DataNS::AnalogsNS::Info &info, size_t channelIndex) {
  if (c3d.header().scaleFactor() < 0) // if it is float
    data((c3d.readFloat(info.processorType(), file) -
          info.zeroOffset()[channelIndex]) *
         info.scaleFactors()[channelIndex] * info.generalFactor());
  else
    data((static_cast<float>(
              c3d.readInt(info.processorType(), file, ezc3d::DATA_TYPE::WORD)) -
          info.zeroOffset()[channelIndex]) *
         info.scaleFactors()[channelIndex] * info.generalFactor());
}

void ezc3d::DataNS::AnalogsNS::Channel::print() const {
  std::cout << "Analog = " << data() << "\n";
}

void ezc3d::DataNS::AnalogsNS::Channel::write(std::fstream &f,
                                              double scaleFactor) const {
  float data(static_cast<float>(_data / scaleFactor));
  f.write(reinterpret_cast<const char *>(&data), ezc3d::DATA_TYPE::FLOAT);
}

double ezc3d::DataNS::AnalogsNS::Channel::data() const { return _data; }

void ezc3d::DataNS::AnalogsNS::Channel::data(double v) { _data = v; }

bool ezc3d::DataNS::AnalogsNS::Channel::isEmpty() const {
  if (static_cast<double>(data()) == 0.0) {
    return true;
  } else {
    return false;
  }
}

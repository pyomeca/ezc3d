#define EZC3D_API_EXPORTS
///
/// \file ezc3d.cpp
/// \brief Implementation of ezc3d class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "ezc3d/ezc3d.h"
#include "ezc3d/Data.h"
#include "ezc3d/DataStartInfo.h"
#include "ezc3d/Header.h"
#include "ezc3d/Parameters.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

void ezc3d::removeTrailingSpaces(std::string &s) {
  // Remove the spaces at the end of the strings
  for (int i = static_cast<int>(s.size()); i >= 0; --i)
    if (s.size() > 0 && s[s.size() - 1] == ' ')
      s.pop_back();
    else
      break;
}

std::string ezc3d::toUpper(const std::string &str) {
  std::string new_str = str;
  std::transform(new_str.begin(), new_str.end(), new_str.begin(), ::toupper);
  return new_str;
}

ezc3d::c3d::c3d()
    : _filePath(""), m_nByteToRead_float(4 * ezc3d::DATA_TYPE::BYTE),
      m_nByteToReadMax_int(100) {
  c_float = std::vector<char>(m_nByteToRead_float + 1);
  c_float_tp = std::vector<char>(m_nByteToRead_float + 1);
  c_int = std::vector<char>(m_nByteToReadMax_int + 1);
  c_int_tp = std::vector<char>(m_nByteToReadMax_int + 1);

  _header = std::shared_ptr<ezc3d::Header>(new ezc3d::Header());
  _parameters = std::shared_ptr<ezc3d::ParametersNS::Parameters>(
      new ezc3d::ParametersNS::Parameters());
  _data = std::shared_ptr<ezc3d::DataNS::Data>(new ezc3d::DataNS::Data());
}

ezc3d::c3d::c3d(const std::string &filePath, bool ignoreBadFormatting)
    : _filePath(filePath), m_nByteToRead_float(4 * ezc3d::DATA_TYPE::BYTE),
      m_nByteToReadMax_int(100) {
  std::fstream stream(_filePath, std::ios::in | std::ios::binary);
  c_float = std::vector<char>(m_nByteToRead_float + 1);
  c_float_tp = std::vector<char>(m_nByteToRead_float + 1);
  c_int = std::vector<char>(m_nByteToReadMax_int + 1);
  c_int_tp = std::vector<char>(m_nByteToReadMax_int + 1);

  if (!stream.is_open())
    throw std::ios_base::failure(
        "The c3d file could not be opened, please verify the path");

  // Read all the section
  _header = std::shared_ptr<ezc3d::Header>(new ezc3d::Header(*this, stream));
  _parameters = std::shared_ptr<ezc3d::ParametersNS::Parameters>(
      new ezc3d::ParametersNS::Parameters(*this, stream, ignoreBadFormatting));

  // header may be inconsistent with the parameters, so it must be
  // update to make sure sizes are consistent
  updateHeader();

  // Now read the data
  _data = std::shared_ptr<ezc3d::DataNS::Data>(
      new ezc3d::DataNS::Data(*this, stream));

  // Parameters and header may be inconsistent with data,
  // so reprocess them if needed
  updateParameters();

  // Close the file
  stream.close();
}

void ezc3d::c3d::print() const {
  header().print();
  parameters().print();
  data().print();
}

void ezc3d::c3d::write(const std::string &filePath,
                       const WRITE_FORMAT &format) const {
  parametrizedWrite(filePath, format);
}

void ezc3d::c3d::parametrizedWrite(const std::string &filePath,
                                   const WRITE_FORMAT &format,
                                   bool forceZeroBasedOnFrameCount) const {

  std::fstream f(filePath, std::ios::out | std::ios::binary);

  ezc3d::DataStartInfo dataStartInfoToFill;

  // Write the header
  header().write(f, dataStartInfoToFill, forceZeroBasedOnFrameCount);

  // Write the parameters
  ezc3d::ParametersNS::Parameters p(
      parameters().write(f, dataStartInfoToFill, header(), format));

  // Write the data (Should the scales be taken from p?)
  std::vector<double> pointScaleFactor(pointScales());
  std::vector<double> pointAnalogFactors(channelScales());
  data().write(header(), f, pointScaleFactor, pointAnalogFactors,
               dataStartInfoToFill);

  // Go back and write all the required data start
  writeDataStart(f, dataStartInfoToFill);

  f.close();
}

void ezc3d::c3d::resizeCharHolder(unsigned int nByteToRead) {
  m_nByteToReadMax_int = nByteToRead;
  c_int = std::vector<char>(m_nByteToReadMax_int + 1);
  c_int_tp = std::vector<char>(m_nByteToReadMax_int + 1);
}

void ezc3d::c3d::readFile(std::fstream &file, unsigned int nByteToRead,
                          std::vector<char> &c, int nByteFromPrevious,
                          const std::ios_base::seekdir &pos) {
  if (pos != 1)
    file.seekg(nByteFromPrevious, pos); // Move to number analogs
  file.read(&c[0], nByteToRead);
  c[nByteToRead] = '\0'; // Make sure last char is NULL
}

unsigned int ezc3d::c3d::hex2uint(const std::vector<char> &val,
                                  unsigned int len) {
  unsigned int ret(0);
  // Discard any extra bytes to avoid overflow of int
  unsigned int max_bytes = std::min(len, 4u);
  for (unsigned int i = 0; i < max_bytes; ++i)
    ret |= static_cast<unsigned int>(static_cast<unsigned char>(val[i]))
           << (8 * i);
  return ret;
}

int ezc3d::c3d::hex2int(const std::vector<char> &val, unsigned int len) {
  unsigned int tp(hex2uint(val, len));

  // convert to signed int
  // Find max int value
  // Discard any extra bytes to avoid overflow of int
  unsigned int max(0);
  unsigned int max_bytes = std::min(len, 4u);
  for (unsigned int i = 0; i < max_bytes; ++i)
    max |= 0xFFu << (8 * i);

  // If the value is over uint_max / 2 then it is a negative number
  int out;
  if (tp > max / 2)
    out = static_cast<int>(tp - max - 1);
  else
    out = static_cast<int>(tp);

  return out;
}

void ezc3d::c3d::writeDataStart(
    std::fstream &f, const ezc3d::DataStartInfo &dataStartPosition) const {

  if (dataStartPosition.hasHeaderPointDataStart()) {
    f.seekg(dataStartPosition.headerPointDataStart());
    int nBlocksToNext = int(dataStartPosition.pointDataStart()) / 512 +
                        1; // DATA_START is 1-based
    f.write(reinterpret_cast<const char *>(&nBlocksToNext),
            dataStartPosition.headerPointDataStartSize());
  }

  if (dataStartPosition.hasParameterPointDataStart()) {
    f.seekg(dataStartPosition.parameterPointDataStart());
    int nBlocksToNext = int(dataStartPosition.pointDataStart()) / 512 +
                        1; // DATA_START is 1-based
    f.write(reinterpret_cast<const char *>(&nBlocksToNext),
            dataStartPosition.parameterPointDataStartSize());
  }

  if (dataStartPosition.hasParameterRotationsDataStart()) {
    f.seekg(dataStartPosition.parameterRotationsDataStart());
    int nBlocksToNext = int(dataStartPosition.rotationsDataStart()) / 512 +
                        1; // DATA_START is 1-based
    f.write(reinterpret_cast<const char *>(&nBlocksToNext),
            dataStartPosition.parameterRotationsDataStartSize());
  }
}

int ezc3d::c3d::readInt(PROCESSOR_TYPE processorType, std::fstream &file,
                        unsigned int nByteToRead, int nByteFromPrevious,
                        const std::ios_base::seekdir &pos) {
  if (nByteToRead > m_nByteToReadMax_int)
    resizeCharHolder(nByteToRead);

  readFile(file, nByteToRead, c_int, nByteFromPrevious, pos);

  int out;
  if (processorType == PROCESSOR_TYPE::MIPS) {
    // This is more or less good. Sometimes, it should not reverse...
    for (size_t i = 0; i < nByteToRead; ++i) {
      c_int_tp[i] = c_int[nByteToRead - 1 - i];
    }
    c_int_tp[nByteToRead] = '\0';
    out = hex2int(c_int_tp, nByteToRead);
  } else {
    // make sure it is an int and not an unsigned int
    out = hex2int(c_int, nByteToRead);
  }

  return out;
}

size_t ezc3d::c3d::readUint(PROCESSOR_TYPE processorType, std::fstream &file,
                            unsigned int nByteToRead, int nByteFromPrevious,
                            const std::ios_base::seekdir &pos) {
  if (nByteToRead > m_nByteToReadMax_int)
    resizeCharHolder(nByteToRead);

  readFile(file, nByteToRead, c_int, nByteFromPrevious, pos);

  size_t out;
  if (processorType == PROCESSOR_TYPE::MIPS) {
    // This is more or less good. Sometimes, it should not reverse...
    for (size_t i = 0; i < nByteToRead; ++i) {
      c_int_tp[i] = c_int[nByteToRead - 1 - i];
    }
    c_int_tp[nByteToRead] = '\0';
    // make sure it is an int and not an unsigned int
    out = hex2uint(c_int_tp, nByteToRead);
  } else {
    // make sure it is an int and not an unsigned int
    out = hex2uint(c_int, nByteToRead);
  }

  return out;
}

float ezc3d::c3d::readFloat(PROCESSOR_TYPE processorType, std::fstream &file,
                            int nByteFromPrevious,
                            const std::ios_base::seekdir &pos) {
  readFile(file, m_nByteToRead_float, c_float, nByteFromPrevious, pos);
  float out;
  if (processorType == PROCESSOR_TYPE::INTEL) {
    out = *reinterpret_cast<float *>(&c_float[0]);
  } else if (processorType == PROCESSOR_TYPE::DEC) {
    c_float_tp[0] = c_float[2];
    c_float_tp[1] = c_float[3];
    c_float_tp[2] = c_float[0];
    if (c_float[1] != 0)
      c_float_tp[3] = c_float[1] - 1;
    else
      c_float_tp[3] = c_float[1];
    c_float_tp[4] = '\0';
    out = *reinterpret_cast<float *>(&c_float_tp[0]);
  } else if (processorType == PROCESSOR_TYPE::MIPS) {
    for (unsigned int i = 0; i < m_nByteToRead_float; ++i)
      c_float_tp[i] = c_float[m_nByteToRead_float - 1 - i];
    c_float_tp[m_nByteToRead_float] = '\0';
    out = *reinterpret_cast<float *>(&c_float_tp[0]);
  } else {
    throw std::runtime_error("Wrong type of processor for floating points");
  }
  return out;
}

std::string ezc3d::c3d::readString(std::fstream &file, unsigned int nByteToRead,
                                   int nByteFromPrevious,
                                   const std::ios_base::seekdir &pos) {
  if (nByteToRead > m_nByteToReadMax_int)
    resizeCharHolder(nByteToRead);

  std::vector<char> c = std::vector<char>(nByteToRead + 1);
  readFile(file, nByteToRead, c, nByteFromPrevious, pos);
  std::string out(&c[0]);
  return out;
}

void ezc3d::c3d::readParam(PROCESSOR_TYPE processorType, std::fstream &file,
                           unsigned int dataLenghtInBytes,
                           const std::vector<size_t> &dimension,
                           std::vector<int> &param_data, size_t currentIdx) {
  for (size_t i = 0; i < dimension[currentIdx]; ++i)
    if (currentIdx == dimension.size() - 1)
      param_data.push_back(readInt(processorType, file,
                                   dataLenghtInBytes * ezc3d::DATA_TYPE::BYTE));
    else
      readParam(processorType, file, dataLenghtInBytes, dimension, param_data,
                currentIdx + 1);
}

void ezc3d::c3d::readParam(PROCESSOR_TYPE processorType, std::fstream &file,
                           const std::vector<size_t> &dimension,
                           std::vector<double> &param_data, size_t currentIdx) {
  for (size_t i = 0; i < dimension[currentIdx]; ++i)
    if (currentIdx == dimension.size() - 1)
      param_data.push_back(readFloat(processorType, file));
    else
      readParam(processorType, file, dimension, param_data, currentIdx + 1);
}

void ezc3d::c3d::readParam(std::fstream &file,
                           const std::vector<size_t> &dimension,
                           std::vector<std::string> &param_data_string) {
  std::vector<std::string> param_data_string_tp;
  _readMatrix(file, dimension, param_data_string_tp);

  // Vicon c3d stores text length on first dimension, I am not sure if
  // this is a standard or a custom made stuff.
  // I implemented it like that for now
  if (dimension.size() == 1) {
    if (dimension[0] != 0) {
      std::string tp;
      for (size_t j = 0; j < dimension[0]; ++j)
        tp += param_data_string_tp[j];
      ezc3d::removeTrailingSpaces(tp);
      param_data_string.push_back(tp);
    }
  } else
    _dispatchMatrix(dimension, param_data_string_tp, param_data_string);
}

void ezc3d::c3d::moveCursorToANewBlock(std::fstream &f) {
  // Move the cursor to the beginning of a block as rotations should start at a
  // new block
  int blankValue(0);
  std::streampos currentPos(f.tellg());
  for (int i = 0; i < 512 - static_cast<int>(currentPos) % 512; ++i) {
    f.write(reinterpret_cast<const char *>(&blankValue), ezc3d::BYTE);
  }
}

size_t
ezc3d::c3d::_dispatchMatrix(const std::vector<size_t> &dimension,
                            const std::vector<std::string> &param_data_in,
                            std::vector<std::string> &param_data_out,
                            size_t idxInParam, size_t currentIdx) {
  for (size_t i = 0; i < dimension[currentIdx]; ++i)
    if (currentIdx == dimension.size() - 1) {
      std::string tp;
      for (size_t j = 0; j < dimension[0]; ++j) {
        tp += param_data_in[idxInParam];
        ++idxInParam;
      }
      ezc3d::removeTrailingSpaces(tp);
      param_data_out.push_back(tp);
    } else
      idxInParam = _dispatchMatrix(dimension, param_data_in, param_data_out,
                                   idxInParam, currentIdx + 1);
  return idxInParam;
}

void ezc3d::c3d::_readMatrix(std::fstream &file,
                             const std::vector<size_t> &dimension,
                             std::vector<std::string> &param_data,
                             size_t currentIdx) {
  for (size_t i = 0; i < dimension[currentIdx]; ++i)
    if (currentIdx == dimension.size() - 1)
      param_data.push_back(readString(file, ezc3d::DATA_TYPE::BYTE));
    else
      _readMatrix(file, dimension, param_data, currentIdx + 1);
}

const ezc3d::Header &ezc3d::c3d::header() const { return *_header; }

const ezc3d::ParametersNS::Parameters &ezc3d::c3d::parameters() const {
  return *_parameters;
}

const ezc3d::DataNS::Data &ezc3d::c3d::data() const { return *_data; }

const std::vector<std::string> ezc3d::c3d::pointNames() const {
  std::vector<std::string> labels =
      parameters().group("POINT").parameter("LABELS").valuesAsString();
  int i = 2;
  while (
      parameters().group("POINT").isParameter("LABELS" + std::to_string(i))) {
    const std::vector<std::string> &labels_tp =
        parameters()
            .group("POINT")
            .parameter("LABELS" + std::to_string(i))
            .valuesAsString();
    labels.insert(labels.end(), labels_tp.begin(), labels_tp.end());
    ++i;
  }
  return labels;
}

const std::vector<double> ezc3d::c3d::pointScales() const {
  std::vector<double> scales =
      parameters().group("POINT").parameter("SCALE").valuesAsDouble();
  int i = 2;
  while (parameters().group("POINT").isParameter("SCALE" + std::to_string(i))) {
    const auto &scales_tp = parameters()
                                .group("POINT")
                                .parameter("SCALE" + std::to_string(i))
                                .valuesAsDouble();
    scales.insert(scales.end(), scales_tp.begin(), scales_tp.end());
    ++i;
  }
  return scales;
}

size_t ezc3d::c3d::pointIdx(const std::string &pointName) const {
  const std::vector<std::string> &currentNames(pointNames());
  for (size_t i = 0; i < currentNames.size(); ++i)
    if (!currentNames[i].compare(pointName))
      return i;
  throw std::invalid_argument("ezc3d::pointIdx could not find " + pointName +
                              " in the points data set.");
}

const std::vector<std::string> ezc3d::c3d::channelNames() const {

  std::vector<std::string> labels =
      parameters().group("ANALOG").parameter("LABELS").valuesAsString();
  int i = 2;
  while (
      parameters().group("ANALOG").isParameter("LABELS" + std::to_string(i))) {
    const std::vector<std::string> &labels_tp =
        parameters()
            .group("ANALOG")
            .parameter("LABELS" + std::to_string(i))
            .valuesAsString();
    labels.insert(labels.end(), labels_tp.begin(), labels_tp.end());
    ++i;
  }
  return labels;
}

const std::vector<double> ezc3d::c3d::channelScales() const {
  std::vector<double> scales =
      parameters().group("ANALOG").parameter("SCALE").valuesAsDouble();
  int i = 2;
  while (
      parameters().group("ANALOG").isParameter("SCALE" + std::to_string(i))) {
    const auto &scales_tp = parameters()
                                .group("ANALOG")
                                .parameter("SCALE" + std::to_string(i))
                                .valuesAsDouble();
    scales.insert(scales.end(), scales_tp.begin(), scales_tp.end());
    ++i;
  }
  return scales;
}

const std::vector<int> ezc3d::c3d::channelOffsets() const {
  std::vector<int> offsets =
      parameters().group("ANALOG").parameter("OFFSET").valuesAsInt();
  int i = 2;
  while (
      parameters().group("ANALOG").isParameter("OFFSET" + std::to_string(i))) {
    const auto &offsets_tp = parameters()
                                 .group("ANALOG")
                                 .parameter("OFFSET" + std::to_string(i))
                                 .valuesAsInt();
    offsets.insert(offsets.end(), offsets_tp.begin(), offsets_tp.end());
    ++i;
  }
  return offsets;
}

size_t ezc3d::c3d::channelIdx(const std::string &channelName) const {
  const std::vector<std::string> &currentNames(channelNames());
  for (size_t i = 0; i < currentNames.size(); ++i)
    if (!currentNames[i].compare(channelName))
      return i;
  throw std::invalid_argument("ezc3d::channelIdx could not find " +
                              channelName + " in the analogous data set");
}

void ezc3d::c3d::setFirstFrame(size_t firstFrame) {
  _header->firstFrame(firstFrame);
}

void ezc3d::c3d::setGroupMetadata(const std::string &groupName,
                                  const std::string &description,
                                  bool isLocked) {
  size_t idx;
  try {
    idx = parameters().groupIdx(groupName);
  } catch (const std::invalid_argument &) {
    _parameters->group(ezc3d::ParametersNS::GroupNS::Group(groupName));
    idx = parameters().groupIdx(groupName);
  }

  _parameters->group(idx).description(description);
  if (isLocked) {
    _parameters->group(idx).lock();
  } else {
    _parameters->group(idx).unlock();
  }
}

void ezc3d::c3d::parameter(const std::string &groupName,
                           const ezc3d::ParametersNS::GroupNS::Parameter &p) {
  if (!p.name().compare("")) {
    throw std::invalid_argument("Parameter must have a name");
  }

  size_t idx;
  try {
    idx = parameters().groupIdx(groupName);
  } catch (const std::invalid_argument &) {
    _parameters->group(ezc3d::ParametersNS::GroupNS::Group(groupName));
    idx = parameters().groupIdx(groupName);
  }

  _parameters->group(idx).parameter(p);

  // Do a sanity check on the header if important stuff like number
  // of frames or number of elements is changed
  updateHeader();
}

void ezc3d::c3d::remove(const std::string &groupName,
                        const std::string &parameterName) {
  if (_parameters->isMandatory(groupName, parameterName)) {
    throw std::invalid_argument("You can't remove a mandatory parameter");
  }

  _parameters->group(groupName).remove(parameterName);
}

void ezc3d::c3d::remove(const std::string &groupName) {
  if (_parameters->isMandatory(groupName)) {
    throw std::invalid_argument("You can't remove a mandatory parameter");
  }

  _parameters->remove(groupName);
}

void ezc3d::c3d::lockGroup(const std::string &groupName) {
  _parameters->group(groupName).lock();
}

void ezc3d::c3d::unlockGroup(const std::string &groupName) {
  _parameters->group(groupName).unlock();
}

void ezc3d::c3d::frame(const ezc3d::DataNS::Frame &f, size_t idx,
                       bool skipInternalUpdates) {

  // Replace the jth frame
  _data->frame(f, idx);

  if (skipInternalUpdates)
    return;

  // Make sure f.points().points() is the same as data.f[ANY].points()
  size_t nPoints(static_cast<size_t>(
      parameters().group("POINT").parameter("USED").valuesAsInt()[0]));
  if (nPoints != 0 && f.points().nbPoints() != nPoints)
    throw std::runtime_error(
        "Number of points in POINT:USED parameter must equal to "
        "the number of points sent in the frame");

  std::vector<std::string> labels(
      parameters().group("POINT").parameter("LABELS").valuesAsString());

  try {
    // Check if all the labels are in the actual LABELSX parameter
    const std::vector<std::string> &namesParameter(pointNames());
    for (size_t i = 0; i < labels.size(); ++i) {
      for (size_t i = 0; i < namesParameter.size(); ++i) {
        if (!namesParameter[i].compare(labels[i])) {
          break;
        }
      }
    }
  } catch (const std::invalid_argument &) {
    throw std::invalid_argument("All the points in the frame must appear "
                                "in the POINT:LABELS parameter");
  }

  if (f.points().nbPoints() > 0 &&
      parameters().group("POINT").parameter("RATE").valuesAsDouble()[0] ==
          0.0) {
    throw std::runtime_error(
        "Point frame rate must be specified if you add some");
  }
  if (f.analogs().nbSubframes() > 0 &&
      parameters().group("ANALOG").parameter("RATE").valuesAsDouble()[0] ==
          0.0) {
    throw std::runtime_error(
        "Analog frame rate must be specified if you add some");
  }

  size_t nAnalogs(static_cast<size_t>(
      parameters().group("ANALOG").parameter("USED").valuesAsInt()[0]));
  size_t subSize(f.analogs().nbSubframes());
  if (subSize != 0) {
    size_t nChannel(f.analogs().subframe(0).nbChannels());
    size_t nAnalogByFrames(header().nbAnalogByFrame());
    if (!(nAnalogs == 0 && nAnalogByFrames == 0) && nChannel != nAnalogs)
      throw std::runtime_error(
          "Number of analogs in ANALOG:USED parameter must equal "
          "the number of analogs sent in the frame");
  }

  // Finalize the internal structure
  updateParameters();
}

void ezc3d::c3d::frames(const std::vector<ezc3d::DataNS::Frame> frames,
                        size_t firstFrameidx) {

  for (size_t i = 0; i < frames.size(); i++) {
    // Only performs internal updates on the first and last frames
    bool skipInternalUpdates = i > 0 && i < frames.size() - 1;
    frame(frames[i], firstFrameidx == SIZE_MAX ? SIZE_MAX : firstFrameidx + i,
          skipInternalUpdates);
  }
}

void ezc3d::c3d::point(const std::string &pointName) {
  if (data().nbFrames() > 0) {
    std::vector<ezc3d::DataNS::Frame> dummy_frames;
    ezc3d::DataNS::Points3dNS::Points dummy_pts;
    ezc3d::DataNS::Points3dNS::Point emptyPoint;
    dummy_pts.point(emptyPoint);
    ezc3d::DataNS::Frame frame;
    frame.add(dummy_pts);
    for (size_t f = 0; f < data().nbFrames(); ++f)
      dummy_frames.push_back(frame);
    point(pointName, dummy_frames);
  } else {
    updateParameters({pointName});
  }
}

void ezc3d::c3d::point(const std::string &pointName,
                       const std::vector<ezc3d::DataNS::Frame> &frames) {
  std::vector<std::string> names;
  names.push_back(pointName);
  point(names, frames);
}

void ezc3d::c3d::point(const std::vector<std::string> &ptsNames) {
  if (data().nbFrames() > 0) {
    std::vector<ezc3d::DataNS::Frame> dummy_frames;
    ezc3d::DataNS::Points3dNS::Points dummy_pts;
    ezc3d::DataNS::Points3dNS::Point emptyPoint;
    for (size_t i = 0; i < ptsNames.size(); ++i) {
      dummy_pts.point(emptyPoint);
    }
    ezc3d::DataNS::Frame frame;
    frame.add(dummy_pts);
    for (size_t f = 0; f < data().nbFrames(); ++f)
      dummy_frames.push_back(frame);
    point(ptsNames, dummy_frames);
  } else {
    updateParameters(ptsNames);
  }
}

void ezc3d::c3d::point(const std::vector<std::string> &ptsNames,
                       const std::vector<ezc3d::DataNS::Frame> &frames) {
  if (frames.size() == 0 || frames.size() != data().nbFrames())
    throw std::invalid_argument(
        "Size of the array of frames must equal the number of "
        "frames already present in the data set");
  if (frames[0].points().nbPoints() == 0)
    throw std::invalid_argument("Points in the frames cannot be empty");

  const std::vector<std::string> &labels(pointNames());

  for (size_t idx = 0; idx < ptsNames.size(); ++idx) {
    for (size_t i = 0; i < labels.size(); ++i)
      if (!ptsNames[idx].compare(labels[i]))
        throw std::invalid_argument(
            "The point you try to create already exists "
            "in the data set");

    for (size_t f = 0; f < data().nbFrames(); ++f)
      _data->frame(f).points().point(frames[f].points().point(idx));
  }
  updateParameters(ptsNames);
}

void ezc3d::c3d::analog(const std::string &channelName) {
  if (data().nbFrames() > 0) {
    std::vector<ezc3d::DataNS::Frame> dummy_frames;
    ezc3d::DataNS::AnalogsNS::SubFrame dummy_subframes;
    ezc3d::DataNS::AnalogsNS::Channel emptyChannel;
    emptyChannel.data(0);
    ezc3d::DataNS::Frame frame;
    dummy_subframes.channel(emptyChannel);
    for (size_t sf = 0; sf < header().nbAnalogByFrame(); ++sf)
      frame.analogs().subframe(dummy_subframes);
    for (size_t f = 0; f < data().nbFrames(); ++f)
      dummy_frames.push_back(frame);
    analog(channelName, dummy_frames);
  } else {
    updateParameters({}, {channelName});
  }
}

void ezc3d::c3d::analog(std::string channelName,
                        const std::vector<ezc3d::DataNS::Frame> &frames) {
  std::vector<std::string> names;
  names.push_back(channelName);
  analog(names, frames);
}

void ezc3d::c3d::analog(const std::vector<std::string> &channelNames) {
  if (data().nbFrames() > 0) {
    std::vector<ezc3d::DataNS::Frame> dummy_frames;
    ezc3d::DataNS::AnalogsNS::SubFrame dummy_subframes;
    ezc3d::DataNS::AnalogsNS::Channel emptyChannel;
    emptyChannel.data(0);
    ezc3d::DataNS::Frame frame;
    for (size_t i = 0; i < channelNames.size(); ++i) {
      dummy_subframes.channel(emptyChannel);
    }
    for (size_t sf = 0; sf < header().nbAnalogByFrame(); ++sf)
      frame.analogs().subframe(dummy_subframes);
    for (size_t f = 0; f < data().nbFrames(); ++f)
      dummy_frames.push_back(frame);
    analog(channelNames, dummy_frames);
  } else {
    updateParameters({}, channelNames);
  }
}

void ezc3d::c3d::analog(const std::vector<std::string> &chanNames,
                        const std::vector<ezc3d::DataNS::Frame> &frames) {
  if (frames.size() != data().nbFrames())
    throw std::invalid_argument(
        "Size of the array of frames must equal the number of "
        "frames already present in the data set");
  if (frames[0].analogs().nbSubframes() != header().nbAnalogByFrame())
    throw std::invalid_argument(
        "Size of the subframes in the frames must equal the "
        "number of subframes already present in the data set");
  if (frames[0].analogs().subframe(0).nbChannels() == 0)
    throw std::invalid_argument("Channels in the frame cannot be empty");

  const std::vector<std::string> &labels(channelNames());
  for (size_t idx = 0; idx < chanNames.size(); ++idx) {
    for (size_t i = 0; i < labels.size(); ++i)
      if (!chanNames[idx].compare(labels[i]))
        throw std::invalid_argument("The channel you try to create already "
                                    "exists in the data set");

    for (size_t f = 0; f < data().nbFrames(); ++f) {
      for (size_t sf = 0; sf < header().nbAnalogByFrame(); ++sf) {
        _data->frame(f).analogs().subframe(sf).channel(
            frames[f].analogs().subframe(sf).channel(idx));
      }
    }
  }
  updateParameters({}, chanNames);
}

void ezc3d::c3d::updateHeader() {
  // Parameter is always consider as the right value.
  const auto &points(parameters().group("POINT"));
  size_t nbFrames(static_cast<size_t>(
      points.parameter("FRAMES").valuesConvertedAsInt()[0]));
  if (nbFrames != 0 && nbFrames != header().nbFrames()) {
    // The nbFrames != 0 is to account for Kistler implementation which does not
    // declare points If there is a discrepancy between them, change the header,
    // while keeping the firstFrame value
    _header->lastFrame(nbFrames + _header->firstFrame() - 1);
  }
  double pointRate(points.parameter("RATE").valuesAsDouble()[0]);
  float buffer(10000); // For decimal truncature
  if (static_cast<int>(pointRate * buffer) !=
      static_cast<int>(header().frameRate() * buffer)) {
    // If there are points but the rate don't match keep the one from header
    if (points.parameter("RATE").valuesAsDouble()[0] == 0.0 &&
        points.parameter("USED").valuesAsInt()[0] != 0) {
      ezc3d::ParametersNS::GroupNS::Parameter rate("RATE");
      rate.set(header().frameRate());
      parameter("POINT", rate);
    } else
      _header->frameRate(static_cast<float>(pointRate));
  }
  if (static_cast<size_t>(points.parameter("USED").valuesAsInt()[0]) !=
      header().nb3dPoints()) {
    _header->nb3dPoints(
        static_cast<size_t>(points.parameter("USED").valuesAsInt()[0]));
  }

  // Compare the subframe with data when possible, otherwise go with the
  // parameters
  const auto &analog(parameters().group("ANALOG"));
  if (_data != nullptr && data().nbFrames() > 0 &&
      data().frame(0).analogs().nbSubframes() != 0) {
    if (data().frame(0).analogs().nbSubframes() != header().nbAnalogByFrame())
      _header->nbAnalogByFrame(data().frame(0).analogs().nbSubframes());
  } else {
    if (static_cast<size_t>(pointRate) == 0) {
      if (header().nbAnalogByFrame() != 1)
        _header->nbAnalogByFrame(1);
    } else {
      if (static_cast<size_t>(analog.parameter("RATE").valuesAsDouble()[0] /
                              pointRate) != header().nbAnalogByFrame()) {
        if (header().nbAnalogByFrame() == 1 && parameters().isGroup("SHADOW")) {
          // The SHADOW company is not following the standard so they did not
          // set analog rate ezc3d automatically sets it to zero which results
          // in a discrepancy
          ezc3d::ParametersNS::GroupNS::Parameter &analogNonConst =
              _parameters->group("ANALOG").parameter("RATE");
          analogNonConst.set(static_cast<float>(header().nbAnalogByFrame()));
        } else {
          _header->nbAnalogByFrame(static_cast<size_t>(
              analog.parameter("RATE").valuesAsDouble()[0] / pointRate));
        }
      }
    }
  }

  if (static_cast<size_t>(analog.parameter("USED").valuesAsInt()[0]) !=
      header().nbAnalogs())
    _header->nbAnalogs(
        static_cast<size_t>(analog.parameter("USED").valuesAsInt()[0]));

  if (parameters().isGroup("ROTATION"))
    _header->hasRotationalData(true);
}

void ezc3d::c3d::updateParameters(const std::vector<std::string> &newPoints,
                                  const std::vector<std::string> &newAnalogs) {
  // If frames has been added
  ezc3d::ParametersNS::GroupNS::Group &grpPoint(
      _parameters->group(parameters().groupIdx("POINT")));
  size_t nFrames(data().nbFrames());
  if (nFrames != static_cast<size_t>(
                     grpPoint.parameter("FRAMES").valuesConvertedAsInt()[0])) {
    size_t idx(grpPoint.parameterIdx("FRAMES"));
    grpPoint.parameter(idx).set(nFrames);
  }

  // If points has been added
  size_t nPoints;
  if (data().nbFrames() > 0)
    nPoints = data().frame(0).points().nbPoints();
  else
    nPoints = parameters().group("POINT").parameter("USED").valuesAsInt()[0] +
              newPoints.size();
  int oldPointUsed(grpPoint.parameter("USED").valuesAsInt()[0]);
  if (nPoints != static_cast<size_t>(oldPointUsed)) {
    grpPoint.parameter("USED").set(nPoints);

    std::vector<std::string> newLabels;
    std::vector<std::string> newDescriptions;
    std::vector<std::string> newUnits;
    std::vector<std::string> ptsNames(pointNames());
    ptsNames.insert(ptsNames.end(), newPoints.begin(), newPoints.end());
    for (size_t i = nPoints - newPoints.size(); i < nPoints; ++i) {
      std::string name;
      if (data().nbFrames() == 0) {
        if (i < static_cast<size_t>(oldPointUsed))
          name = parameters()
                     .group("POINT")
                     .parameter("LABELS")
                     .valuesAsString()[i];
        else
          name = newPoints[i - oldPointUsed];
      } else {
        name = ptsNames[i];
        removeTrailingSpaces(name);
      }
      newLabels.push_back(name);
      newDescriptions.push_back("");
      newUnits.push_back("mm");
    }

    // Dispatch names in LABELS, LABELS2, etc.
    size_t first_idx = 0;
    size_t last_idx = 0;
    size_t i = 0;
    while (last_idx < newLabels.size()) {
      std::string mod("");
      if (i != 0) {
        mod = std::to_string(i + 1);
        if (!grpPoint.isParameter("LABELS" + mod)) {
          ezc3d::ParametersNS::GroupNS::Parameter labels("LABELS" + mod);
          labels.set(std::vector<std::string>() = {});
          grpPoint.parameter(labels);
        }
        if (!grpPoint.isParameter("DESCRIPTIONS" + mod)) {
          ezc3d::ParametersNS::GroupNS::Parameter descriptions("DESCRIPTIONS" +
                                                               mod);
          descriptions.set(std::vector<std::string>() = {});
          grpPoint.parameter(descriptions);
        }
        if (!grpPoint.isParameter("UNITS" + mod)) {
          ezc3d::ParametersNS::GroupNS::Parameter units("UNITS" + mod);
          units.set(std::vector<std::string>() = {});
          grpPoint.parameter(units);
        }
      }
      auto labels = grpPoint.parameter("LABELS" + mod).valuesAsString();
      auto descriptions =
          grpPoint.parameter("DESCRIPTIONS" + mod).valuesAsString();
      auto units = grpPoint.parameter("UNITS" + mod).valuesAsString();

      if (labels.size() != 255) {
        size_t off = grpPoint.parameter("LABELS" + mod).valuesAsString().size();
        last_idx = newLabels.size() >= first_idx + 255 - off
                       ? first_idx + 255 - off
                       : newLabels.size();
        labels.insert(labels.end(), newLabels.begin() + first_idx,
                      newLabels.begin() + last_idx);
        descriptions.insert(descriptions.end(),
                            newDescriptions.begin() + first_idx,
                            newDescriptions.begin() + last_idx);
        units.insert(units.end(), newUnits.begin() + first_idx,
                     newUnits.begin() + last_idx);

        grpPoint.parameter("LABELS" + mod).set(labels);
        grpPoint.parameter("DESCRIPTIONS" + mod).set(descriptions);
        grpPoint.parameter("UNITS" + mod).set(units);

        // Prepare next for
        first_idx = last_idx;
      }
      ++i;
    }
  }

  // If analogous data has been added
  ezc3d::ParametersNS::GroupNS::Group &grpAnalog(
      _parameters->group(parameters().groupIdx("ANALOG")));
  size_t nAnalogs;
  if (data().nbFrames() > 0) {
    if (data().frame(0).analogs().nbSubframes() > 0)
      nAnalogs = data().frame(0).analogs().subframe(0).nbChannels();
    else
      nAnalogs = 0;
  } else
    nAnalogs = parameters().group("ANALOG").parameter("USED").valuesAsInt()[0] +
               newAnalogs.size();

  // Should always be greater than 0..., but we have to take in
  // account Optotrak lazyness
  if (parameters().group("ANALOG").nbParameters()) {
    int oldAnalogUsed(grpAnalog.parameter("USED").valuesAsInt()[0]);
    if (nAnalogs != static_cast<size_t>(oldAnalogUsed)) {
      grpAnalog.parameter("USED").set(nAnalogs);

      std::vector<std::string> newLabels;
      std::vector<std::string> newDescriptions;
      std::vector<double> newScale;
      std::vector<int> newOffset;
      std::vector<std::string> newUnits;
      std::vector<std::string> chanNames(channelNames());
      chanNames.insert(chanNames.end(), newAnalogs.begin(), newAnalogs.end());
      for (size_t i = nAnalogs - newAnalogs.size(); i < nAnalogs; ++i) {
        std::string name;
        if (data().nbFrames() == 0) {
          if (i < static_cast<size_t>(oldAnalogUsed))
            name = parameters()
                       .group("ANALOG")
                       .parameter("LABELS")
                       .valuesAsString()[i];
          else
            name = newAnalogs[i - oldAnalogUsed];
        } else {
          name = chanNames[i];
          removeTrailingSpaces(name);
        }
        newLabels.push_back(name);
        newDescriptions.push_back("");
        newScale.push_back(1.0);
        newOffset.push_back(0);
        newUnits.push_back("");
      }

      // Dispatch names in LABELS, LABELS2, etc.
      size_t first_idx = 0;
      size_t last_idx = 0;
      size_t i = 0;
      while (last_idx < newLabels.size()) {
        std::string mod("");
        if (i != 0) {
          mod = std::to_string(i + 1);
          if (!grpAnalog.isParameter("LABELS" + mod)) {
            ezc3d::ParametersNS::GroupNS::Parameter labels("LABELS" + mod);
            labels.set(std::vector<std::string>() = {});
            grpAnalog.parameter(labels);
          }
          if (!grpAnalog.isParameter("DESCRIPTIONS" + mod)) {
            ezc3d::ParametersNS::GroupNS::Parameter descriptions(
                "DESCRIPTIONS" + mod);
            descriptions.set(std::vector<std::string>() = {});
            grpAnalog.parameter(descriptions);
          }
          if (!grpAnalog.isParameter("SCALE" + mod)) {
            ezc3d::ParametersNS::GroupNS::Parameter scale("SCALE" + mod);
            scale.set(std::vector<double>() = {});
            grpAnalog.parameter(scale);
          }
          if (!grpAnalog.isParameter("OFFSET" + mod)) {
            ezc3d::ParametersNS::GroupNS::Parameter offset("OFFSET" + mod);
            offset.set(std::vector<int>() = {});
            grpAnalog.parameter(offset);
          }
          if (!grpAnalog.isParameter("UNITS" + mod)) {
            ezc3d::ParametersNS::GroupNS::Parameter units("UNITS" + mod);
            units.set(std::vector<std::string>() = {});
            grpAnalog.parameter(units);
          }
        }

        auto labels = grpAnalog.parameter("LABELS" + mod).valuesAsString();
        auto descriptions =
            grpAnalog.parameter("DESCRIPTIONS" + mod).valuesAsString();
        auto scale = grpAnalog.parameter("SCALE" + mod).valuesAsDouble();
        auto offset = grpAnalog.parameter("OFFSET" + mod).valuesAsInt();
        auto units = grpAnalog.parameter("UNITS" + mod).valuesAsString();

        if (labels.size() != 255) {
          size_t off =
              grpAnalog.parameter("LABELS" + mod).valuesAsString().size();
          last_idx = newLabels.size() >= first_idx + 255 - off
                         ? first_idx + 255 - off
                         : newLabels.size();
          labels.insert(labels.end(), newLabels.begin() + first_idx,
                        newLabels.begin() + last_idx);
          descriptions.insert(descriptions.end(),
                              newDescriptions.begin() + first_idx,
                              newDescriptions.begin() + last_idx);
          scale.insert(scale.end(), newScale.begin() + first_idx,
                       newScale.begin() + last_idx);
          offset.insert(offset.end(), newOffset.begin() + first_idx,
                        newOffset.begin() + last_idx);
          units.insert(units.end(), newUnits.begin() + first_idx,
                       newUnits.begin() + last_idx);

          grpAnalog.parameter("LABELS" + mod).set(labels);
          grpAnalog.parameter("DESCRIPTIONS" + mod).set(descriptions);
          grpAnalog.parameter("SCALE" + mod).set(scale);
          grpAnalog.parameter("OFFSET" + mod).set(offset);
          grpAnalog.parameter("UNITS" + mod).set(units);

          // Prepare next for
          first_idx = last_idx;
        }
        ++i;
      }
    }
  }

  // Adjust some ROTATION parameters
  if (_parameters->isGroup("ROTATION")) {
    ezc3d::ParametersNS::GroupNS::Group &grpRotation(
        _parameters->group(parameters().groupIdx("ROTATION")));
    size_t nbRotations = 0;
    if (_data->frame(0).rotations().nbSubframes() > 0) {
      nbRotations = _data->frame(0).rotations().subframe(0).nbRotations();
    }
    grpRotation.parameter("USED").set(nbRotations);
  }

  updateHeader();
}

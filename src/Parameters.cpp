#define EZC3D_API_EXPORTS
///
/// \file Parameters.cpp
/// \brief Implementation of Parameters class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "ezc3d/Parameters.h"
#include "ezc3d/Header.h"
#include "ezc3d/ezc3d.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

ezc3d::ParametersNS::Parameters::Parameters()
    : _parametersStart(1), _checksum(0x50), _nbParamBlock(0),
      _processorType(PROCESSOR_TYPE::NO_PROCESSOR_TYPE) {
  setMandatoryParameters();
}

ezc3d::ParametersNS::Parameters::Parameters(ezc3d::c3d &c3d, std::fstream &file,
                                            bool ignoreBadFormatting)
    : _parametersStart(1), _checksum(0x50), _nbParamBlock(0),
      _processorType(PROCESSOR_TYPE::NO_PROCESSOR_TYPE) {

  _parametersStart =
      c3d.readUint(processorType(), file, 1 * ezc3d::DATA_TYPE::BYTE,
                   static_cast<int>(256 * ezc3d::DATA_TYPE::WORD *
                                        (c3d.header().parametersAddress() - 1) +
                                    c3d.header().nbOfZerosBeforeHeader()),
                   std::ios::beg);
  if (_parametersStart != 1) {
    // This is there for historical reasons. The parameters start is always 1
    // since the header is of fixed size but it is still in the file. So just
    // ignore it.
    _parametersStart = 1;
  }

  _checksum = c3d.readUint(processorType(), file, 1 * ezc3d::DATA_TYPE::BYTE);
  if (_checksum != 0x50) {
    // This is there for historical reasons. The checksum is not used anymore
    // but it is still in the file. So just ignore it.
    _checksum = 0x50;
  }

  _nbParamBlock =
      c3d.readUint(processorType(), file, 1 * ezc3d::DATA_TYPE::BYTE);
  size_t processorTypeId =
      c3d.readUint(processorType(), file, 1 * ezc3d::DATA_TYPE::BYTE);

  if (processorTypeId == 84)
    _processorType = ezc3d::PROCESSOR_TYPE::INTEL;
  else if (processorTypeId == 85)
    _processorType = ezc3d::PROCESSOR_TYPE::DEC;
  else if (processorTypeId == 86) {
    _processorType = ezc3d::PROCESSOR_TYPE::MIPS;
    throw std::runtime_error(
        "MIPS processor type not supported yet, please open a "
        "GitHub issue to report that you want this feature!");
  } else
    throw std::runtime_error("Could not read the processor type");

  // Read parameter or group
  std::streampos nextParamByteInFile(static_cast<int>(file.tellg()) +
                                     static_cast<int>(_parametersStart) -
                                     ezc3d::DATA_TYPE::BYTE);
  while (nextParamByteInFile) {
    // Check if we spontaneously got to the next parameter.
    // Otherwise c3d is messed up
    if (!ignoreBadFormatting && file.tellg() != nextParamByteInFile) {
      throw std::ios_base::failure(
          "The format is not standard. If you want to ignore this error, set "
          "ignoreBadFormatting to true");
    }

    // Nb of char in the group name, locked if negative,
    // 0 if we finished the section
    int nbCharInName(
        c3d.readInt(processorType(), file, 1 * ezc3d::DATA_TYPE::BYTE));
    if (nbCharInName == 0)
      break;
    int id(c3d.readInt(processorType(), file, 1 * ezc3d::DATA_TYPE::BYTE));

    // Make sure there at least enough group
    for (size_t i = _groups.size(); i < static_cast<size_t>(abs(id)); ++i)
      _groups.push_back(ezc3d::ParametersNS::GroupNS::Group());

    // Group ID always negative for groups
    // and positive parameter of group ID
    if (id < 0) {
      nextParamByteInFile = group(static_cast<size_t>(abs(id) - 1))
                                .read(c3d, *this, file, nbCharInName);
    } else {
      nextParamByteInFile = group(static_cast<size_t>(id - 1))
                                .parameter(c3d, *this, file, nbCharInName);
    }
  }

  // If some mandatory groups/parameters are not set by having a non
  // compliant C3D, fix it
  setMandatoryParameters();
}

bool ezc3d::ParametersNS::Parameters::isMandatory(
    const std::string &groupName) {
  if (!groupName.compare("POINT") || !groupName.compare("ANALOG") ||
      !groupName.compare("FORCE_PLATFORM")) {
    return true;
  }
  return false;
}

bool ezc3d::ParametersNS::Parameters::isMandatory(
    const std::string &groupName, const std::string &parameterName) {
  if (!groupName.compare("POINT")) {
    if (!parameterName.compare("USED") || !parameterName.compare("LABELS") ||
        !parameterName.compare("DESCRIPTIONS") ||
        !parameterName.compare("SCALE") || !parameterName.compare("UNITS") ||
        !parameterName.compare("RATE") ||
        !parameterName.compare("DATA_START") ||
        !parameterName.compare("FRAMES")) {
      return true;
    }
  } else if (!groupName.compare("ANALOG")) {
    if (!parameterName.compare("USED") || !parameterName.compare("LABELS") ||
        !parameterName.compare("DESCRIPTIONS") ||
        !parameterName.compare("GEN_SCALE") ||
        !parameterName.compare("SCALE") || !parameterName.compare("OFFSET") ||
        !parameterName.compare("UNITS") || !parameterName.compare("RATE") ||
        !parameterName.compare("FORMAT") || !parameterName.compare("BITS")) {
      return true;
    }
  } else if (!groupName.compare("FORCE_PLATFORM")) {
    if (!parameterName.compare("USED") || !parameterName.compare("TYPE") ||
        !parameterName.compare("CHANNEL") || !parameterName.compare("ZERO") ||
        !parameterName.compare("ORIGIN") || !parameterName.compare("CORNERS") ||
        !parameterName.compare("CAL_MATRIX")) {
      return true;
    }
  }
  return false;
}

void ezc3d::ParametersNS::Parameters::setMandatoryParameters() {
  // Mandatory groups
  {
    if (!isGroup("POINT")) {
      group(ezc3d::ParametersNS::GroupNS::Group("POINT"));
    }

    ezc3d::ParametersNS::GroupNS::Group &grp(group("POINT"));
    if (!grp.isParameter("USED")) {
      ezc3d::ParametersNS::GroupNS::Parameter p("USED", "");
      p.set(0);
      p.lock();
      grp.parameter(p);
    }
    if (!grp.isParameter("LABELS")) {
      ezc3d::ParametersNS::GroupNS::Parameter p("LABELS", "");
      p.set(std::vector<std::string>() = {});
      grp.parameter(p);
    }
    if (!grp.isParameter("DESCRIPTIONS")) {
      ezc3d::ParametersNS::GroupNS::Parameter p("DESCRIPTIONS", "");
      p.set(std::vector<std::string>() = {});
      grp.parameter(p);
    }
    if (!grp.isParameter("SCALE")) {
      ezc3d::ParametersNS::GroupNS::Parameter p("SCALE", "");
      p.set(-1.0);
      p.lock();
      grp.parameter(p);
    }
    if (!grp.isParameter("UNITS")) {
      ezc3d::ParametersNS::GroupNS::Parameter p("UNITS", "");
      p.set(std::vector<std::string>() = {});
      grp.parameter(p);
    }
    if (!grp.isParameter("RATE")) {
      ezc3d::ParametersNS::GroupNS::Parameter p("RATE", "");
      p.set(0.0);
      p.lock();
      grp.parameter(p);
    }
    if (!grp.isParameter("DATA_START")) {
      ezc3d::ParametersNS::GroupNS::Parameter p("DATA_START", "");
      p.set(0);
      p.lock();
      grp.parameter(p);
    }
    if (!grp.isParameter("FRAMES")) {
      ezc3d::ParametersNS::GroupNS::Parameter p("FRAMES", "");
      p.set(0);
      p.lock();
      grp.parameter(p);
    } else {
      // Make sure the value is actually a positive integer
      auto frames = grp.parameter("FRAMES").valuesAsInt();
      if (frames.size() > 1 || frames[0] < 0) {
        grp.parameter("FRAMES").set(static_cast<uint16_t>(frames[0]));
      }
    }
  }
  {
    if (!isGroup("ANALOG")) {
      group(ezc3d::ParametersNS::GroupNS::Group("ANALOG"));
    }

    ezc3d::ParametersNS::GroupNS::Group &grp(group("ANALOG"));
    if (!grp.isParameter("USED")) {
      ezc3d::ParametersNS::GroupNS::Parameter p("USED", "");
      p.set(0);
      p.lock();
      grp.parameter(p);
    }
    if (!grp.isParameter("LABELS")) {
      ezc3d::ParametersNS::GroupNS::Parameter p("LABELS", "");
      p.set(std::vector<std::string>() = {});
      grp.parameter(p);
    }
    if (!grp.isParameter("DESCRIPTIONS")) {
      ezc3d::ParametersNS::GroupNS::Parameter p("DESCRIPTIONS", "");
      p.set(std::vector<std::string>() = {});
      grp.parameter(p);
    }
    if (!grp.isParameter("GEN_SCALE")) {
      ezc3d::ParametersNS::GroupNS::Parameter p("GEN_SCALE", "");
      p.set(1.0);
      grp.parameter(p);
    }
    if (!grp.isParameter("SCALE")) {
      ezc3d::ParametersNS::GroupNS::Parameter p("SCALE", "");
      p.set(std::vector<double>() = {});
      grp.parameter(p);
    }
    if (!grp.isParameter("OFFSET")) {
      ezc3d::ParametersNS::GroupNS::Parameter p("OFFSET", "");
      p.set(std::vector<int>() = {});
      grp.parameter(p);
    }
    if (!grp.isParameter("UNITS")) {
      ezc3d::ParametersNS::GroupNS::Parameter p("UNITS", "");
      p.set(std::vector<std::string>() = {});
      grp.parameter(p);
    }
    if (!grp.isParameter("RATE")) {
      ezc3d::ParametersNS::GroupNS::Parameter p("RATE", "");
      p.set(0.0);
      p.lock();
      grp.parameter(p);
    }
    if (!grp.isParameter("FORMAT")) {
      ezc3d::ParametersNS::GroupNS::Parameter p("FORMAT", "");
      p.set(std::vector<std::string>() = {});
      grp.parameter(p);
    }
    if (!grp.isParameter("BITS")) {
      ezc3d::ParametersNS::GroupNS::Parameter p("BITS", "");
      p.set(std::vector<int>() = {});
      grp.parameter(p);
    }
  }
  {
    if (!isGroup("FORCE_PLATFORM")) {
      group(ezc3d::ParametersNS::GroupNS::Group("FORCE_PLATFORM"));
    }

    ezc3d::ParametersNS::GroupNS::Group &grp(group("FORCE_PLATFORM"));
    if (!grp.isParameter("USED")) {
      ezc3d::ParametersNS::GroupNS::Parameter p("USED", "");
      p.set(0);
      grp.parameter(p);
    }
    if (!grp.isParameter("TYPE")) {
      ezc3d::ParametersNS::GroupNS::Parameter p("TYPE", "");
      p.set(std::vector<int>() = {});
      grp.parameter(p);
    }
    if (!grp.isParameter("ZERO")) {
      ezc3d::ParametersNS::GroupNS::Parameter p("ZERO", "");
      p.set(std::vector<int>() = {1, 0});
      grp.parameter(p);
    }
    if (!grp.isParameter("CORNERS")) {
      ezc3d::ParametersNS::GroupNS::Parameter p("CORNERS", "");
      p.set(std::vector<double>() = {});
      grp.parameter(p);
    }
    if (!grp.isParameter("ORIGIN")) {
      ezc3d::ParametersNS::GroupNS::Parameter p("ORIGIN", "");
      p.set(std::vector<double>() = {});
      grp.parameter(p);
    }
    if (!grp.isParameter("CHANNEL")) {
      ezc3d::ParametersNS::GroupNS::Parameter p("CHANNEL", "");
      p.set(std::vector<int>() = {});
      grp.parameter(p);
    }
    if (!grp.isParameter("CAL_MATRIX")) {
      ezc3d::ParametersNS::GroupNS::Parameter p("CAL_MATRIX", "");
      p.set(std::vector<double>() = {});
      grp.parameter(p);
    }
  }
}

void ezc3d::ParametersNS::Parameters::setMandatoryParametersForSpecialGroup(
    const std::string &groupName) {
  // Mandatory groups
  if (!groupName.compare("ROTATION")) {
    if (!isGroup("ROTATION")) {
      group(ezc3d::ParametersNS::GroupNS::Group("ROTATION"));
    }

    ezc3d::ParametersNS::GroupNS::Group &grp(group("ROTATION"));
    if (!grp.isParameter("USED")) {
      ezc3d::ParametersNS::GroupNS::Parameter p("USED", "");
      p.set(0);
      grp.parameter(p);
    }
    if (!grp.isParameter("DATA_START")) {
      ezc3d::ParametersNS::GroupNS::Parameter p("DATA_START", "");
      p.set(std::vector<int>() = {1});
      grp.parameter(p);
    }
    if (!grp.isParameter("RATE")) {
      // Double is better as default than RATIO as RATIO is chosen in
      // priority when writing.
      ezc3d::ParametersNS::GroupNS::Parameter p("RATE", "");
      p.set(std::vector<double>() =
                group("POINT").parameter("RATE").valuesAsDouble());
      grp.parameter(p);
    }
    if (!grp.isParameter("LABELS")) {
      ezc3d::ParametersNS::GroupNS::Parameter p("LABELS", "");
      p.set(std::vector<std::string>() = {});
      grp.parameter(p);
    }
    if (!grp.isParameter("DESCRIPTIONS")) {
      ezc3d::ParametersNS::GroupNS::Parameter p("DESCRIPTIONS", "");
      p.set(std::vector<std::string>() = {});
      grp.parameter(p);
    }
  }
}

void ezc3d::ParametersNS::Parameters::print() const {
  std::cout << "Parameters header" << "\n";
  std::cout << "parametersStart = " << parametersStart() << "\n";
  std::cout << "nbParamBlock = " << nbParamBlock() << "\n";
  std::cout << "processorType = " << processorType() << "\n";

  for (size_t i = 0; i < nbGroups(); ++i) {
    std::cout << "Group " << i << "\n";
    group(i).print();
    std::cout << "\n";
  }
  std::cout << "\n";
}

ezc3d::ParametersNS::Parameters ezc3d::ParametersNS::Parameters::write(
    std::fstream &f, ezc3d::DataStartInfo &dataStartPositionToFill,
    const ezc3d::Header &header, const ezc3d::WRITE_FORMAT &format) const {
  ezc3d::ParametersNS::Parameters p(prepareCopyForWriting(header, format));

  // Write the header of parameters
  f.write(reinterpret_cast<const char *>(&p._parametersStart), ezc3d::BYTE);
  int checksum(0x50);
  f.write(reinterpret_cast<const char *>(&checksum), ezc3d::BYTE);
  // Leave a blank space which will be later fill
  // (number of block can't be known before writing them)
  std::streampos pos(f.tellg()); // remember where to input this value later
  int blankValue(0);
  f.write(reinterpret_cast<const char *>(&blankValue), ezc3d::BYTE);
  int processorType = PROCESSOR_TYPE::INTEL;
  f.write(reinterpret_cast<const char *>(&processorType), ezc3d::BYTE);

  // Write each groups
  for (size_t i = 0; i < p.nbGroups(); ++i) {
    const ezc3d::ParametersNS::GroupNS::Group &currentGroup(p.group(i));
    if (!currentGroup.isEmpty())
      currentGroup.write(f, -static_cast<int>(i + 1), dataStartPositionToFill);
  }

  // Move the cursor to a beginning of a block
  ezc3d::c3d::moveCursorToANewBlock(f);
  // Go back at the left blank space (next parameter position in the last
  // parameter) and write the current position
  std::streampos currentPos(f.tellg());
  currentPos = f.tellg();
  f.seekg(pos);
  int nBlocksToNext = int(currentPos - pos - 2) / 512;
  if (int(currentPos - pos - 2) % 512 > 0)
    ++nBlocksToNext;
  f.write(reinterpret_cast<const char *>(&nBlocksToNext), ezc3d::BYTE);

  // Go back to where to start writing the data
  f.seekg(currentPos);

  return p;
}

ezc3d::ParametersNS::Parameters
ezc3d::ParametersNS::Parameters::prepareCopyForWriting(
    const ezc3d::Header &header, const ezc3d::WRITE_FORMAT &format) const {
  // A copy must be done since modifications are made to some parameters
  ezc3d::ParametersNS::Parameters params(*this);

  // Reevalute the number of frames
  int nFrames(this->group("POINT").parameter("FRAMES").valuesAsInt()[0]);
  if (nFrames > 0xFFFF) {
    ezc3d::ParametersNS::GroupNS::Parameter frames(
        params.group("POINT").parameter("FRAMES"));
    frames.set(-1);
    params.group("POINT").parameter(frames);
  }

  // Ensure that the right point scale is in the file
  ezc3d::ParametersNS::GroupNS::Parameter scaleFactorParam;
  double pointScaleFactor;
  if (params.group("POINT").parameter("SCALE").valuesAsDouble().size()) {
    scaleFactorParam = params.group("POINT").parameter("SCALE");
    pointScaleFactor = -fabs(scaleFactorParam.valuesAsDouble()[0]);
  } else {
    pointScaleFactor = -fabs(header.scaleFactor());
    scaleFactorParam.name("SCALE");
  }
  scaleFactorParam.set(pointScaleFactor);
  params.group("POINT").parameter(scaleFactorParam);

  // Ensure that the right analog scale is in the file
  ezc3d::ParametersNS::GroupNS::Parameter analogScaleFactorParam;
  std::vector<double> analogScaleFactor;
  if (params.group("ANALOG").parameter("USED").valuesAsInt()[0] == 0) {
    analogScaleFactorParam.name("SCALE");
  } else if (params.group("ANALOG").parameter("SCALE").valuesAsDouble().size() >
             0) {
    analogScaleFactorParam = params.group("ANALOG").parameter("SCALE");
    analogScaleFactor =
        params.group("ANALOG").parameter("SCALE").valuesAsDouble();
  } else {
    analogScaleFactor.push_back(header.scaleFactor());
    analogScaleFactorParam.name("SCALE");
  }
  analogScaleFactorParam.set(analogScaleFactor);
  params.group("ANALOG").parameter(analogScaleFactorParam);

  // Use Intel floating with no extra scaling
  ezc3d::ParametersNS::GroupNS::Parameter genScale(
      params.group("ANALOG").parameter("GEN_SCALE"));
  genScale.set(1.0);
  params.group("ANALOG").parameter(genScale);

  size_t cmp = 1;
  std::string mod = "";
  do {
    auto offset(params.group("ANALOG").parameter("OFFSET" + mod));
    std::vector<int> offsetValues(offset.valuesAsInt().size());
    for (size_t i = 0; i < offsetValues.size(); ++i) {
      offsetValues[i] = 0;
    }
    offset.set(offsetValues);
    params.group("ANALOG").parameter(offset);
    ++cmp;
    mod = std::to_string(cmp);
  } while (params.group("ANALOG").isParameter("OFFSET" + mod));

  // Add the parameter EZC3D:VERSION and EZC3D:CONTACT
  if (!params.isGroup("EZC3D")) {
    params.group(ezc3d::ParametersNS::GroupNS::Group("EZC3D"));
  }
  // Add/replace the version in the EZC3D group
  ezc3d::ParametersNS::GroupNS::Parameter version("VERSION");
  version.set(EZC3D_VERSION);
  params.group("EZC3D").parameter(version);
  // Add/replace the CONTACT in the EZC3D group
  ezc3d::ParametersNS::GroupNS::Parameter contact("CONTACT");
  contact.set(EZC3D_CONTACT);
  params.group("EZC3D").parameter(contact);

  if (format == WRITE_FORMAT::NEXUS) {
    // Do some stuff related to Nexus
  }
  return params;
}

size_t ezc3d::ParametersNS::Parameters::parametersStart() const {
  return _parametersStart;
}

size_t ezc3d::ParametersNS::Parameters::checksum() const { return _checksum; }

size_t ezc3d::ParametersNS::Parameters::nbParamBlock() const {
  return _nbParamBlock;
}

ezc3d::PROCESSOR_TYPE ezc3d::ParametersNS::Parameters::processorType() const {
  return _processorType;
}

size_t ezc3d::ParametersNS::Parameters::nbGroups() const {
  return _groups.size();
}

bool ezc3d::ParametersNS::Parameters::isGroup(
    const std::string &groupName) const {
  try {
    groupIdx(groupName);
    return true;
  } catch (const std::invalid_argument &) {
    return false;
  }
}

size_t
ezc3d::ParametersNS::Parameters::groupIdx(const std::string &groupName) const {
  for (size_t i = 0; i < nbGroups(); ++i)
    if (!group(i).name().compare(groupName))
      return i;
  throw std::invalid_argument("Parameters::groupIdx could not find " +
                              groupName);
}

const ezc3d::ParametersNS::GroupNS::Group &
ezc3d::ParametersNS::Parameters::group(size_t idx) const {
  try {
    return _groups.at(idx);
  } catch (const std::out_of_range &) {
    throw std::out_of_range(
        "Parameters::group method is trying to access the group " +
        std::to_string(idx) + " while the maximum number of groups is " +
        std::to_string(nbGroups()) + ".");
  }
}

ezc3d::ParametersNS::GroupNS::Group &
ezc3d::ParametersNS::Parameters::group(size_t idx) {
  try {
    return _groups.at(idx);
  } catch (const std::out_of_range &) {
    throw std::out_of_range(
        "Parameters::group method is trying to access the group " +
        std::to_string(idx) + " while the maximum number of groups is " +
        std::to_string(nbGroups()) + ".");
  }
}

const ezc3d::ParametersNS::GroupNS::Group &
ezc3d::ParametersNS::Parameters::group(const std::string &groupName) const {
  return group(groupIdx(groupName));
}

ezc3d::ParametersNS::GroupNS::Group &
ezc3d::ParametersNS::Parameters::group(const std::string &groupName) {
  return group(groupIdx(groupName));
}

void ezc3d::ParametersNS::Parameters::group(
    const ezc3d::ParametersNS::GroupNS::Group &g) {
  // If the group already exist, override and merge
  size_t alreadyExtIdx(SIZE_MAX);
  for (size_t i = 0; i < nbGroups(); ++i)
    if (!group(i).name().compare(g.name()))
      alreadyExtIdx = i;
  if (alreadyExtIdx == SIZE_MAX)
    _groups.push_back(g);
  else {
    for (size_t i = 0; i < g.nbParameters(); ++i)
      _groups[alreadyExtIdx].parameter(g.parameter(i));
  }

  // Do a sanity check for some specific group
  setMandatoryParametersForSpecialGroup(g.name());
}

void ezc3d::ParametersNS::Parameters::remove(const std::string &name) {
  remove(groupIdx(name));
}

void ezc3d::ParametersNS::Parameters::remove(size_t idx) {
  if (idx >= nbGroups()) {
    throw std::out_of_range(
        "Parameters::group method is trying to access the group " +
        std::to_string(idx) + " while the maximum number of groups is " +
        std::to_string(nbGroups()) + ".");
  }
  _groups.erase(_groups.begin() + idx);
}

const std::vector<ezc3d::ParametersNS::GroupNS::Group> &
ezc3d::ParametersNS::Parameters::groups() const {
  return _groups;
}

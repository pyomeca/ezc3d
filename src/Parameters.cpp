#define EZC3D_API_EXPORTS
///
/// \file Parameters.cpp
/// \brief Implementation of Parameters class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "Parameters.h"
#include "Header.h"

ezc3d::ParametersNS::Parameters::Parameters():
    _parametersStart(1),
    _checksum(0x50),
    _nbParamBlock(0),
    _processorType(PROCESSOR_TYPE::NO_PROCESSOR_TYPE) {
    setMandatoryParameters();
}

ezc3d::ParametersNS::Parameters::Parameters(
        ezc3d::c3d &c3d,
        std::fstream &file) :
    _parametersStart(0),
    _checksum(0),
    _nbParamBlock(0),
    _processorType(PROCESSOR_TYPE::NO_PROCESSOR_TYPE) {
    // Read the Parameters Header (assuming Intel processor)
    _parametersStart = c3d.readUint(
                processorType(),
                file,
                1*ezc3d::DATA_TYPE::BYTE, static_cast<int>(
                    256*ezc3d::DATA_TYPE::WORD*(
                        c3d.header().parametersAddress()-1)
                    + c3d.header().nbOfZerosBeforeHeader()),
                std::ios::beg);
    _checksum = c3d.readUint(
                processorType(), file, 1*ezc3d::DATA_TYPE::BYTE);
    _nbParamBlock = c3d.readUint(
                processorType(), file, 1*ezc3d::DATA_TYPE::BYTE);
    size_t processorTypeId = c3d.readUint(
                processorType(), file, 1*ezc3d::DATA_TYPE::BYTE);
    if (_checksum == 0 && _parametersStart == 0){
        // In theory, if this happens, this is a bad c3d formatting and should
        // return an error, but for some reason Qualisys decided that they
        // would not comply to the standard.
        // Therefore set put "_parameterStart" and "_checksum" to 0
        // This is a patch for Qualisys bad formatting c3d
        _parametersStart = 1;
        _checksum = 0x50;
    }
    if (_checksum != 0x50) // If checkbyte is wrong
        throw std::ios_base::failure("File must be a valid c3d file");

    if (processorTypeId == 84)
        _processorType = ezc3d::PROCESSOR_TYPE::INTEL;
    else if (processorTypeId == 85)
        _processorType = ezc3d::PROCESSOR_TYPE::DEC;
    else if (processorTypeId == 86){
        _processorType = ezc3d::PROCESSOR_TYPE::MIPS;
        throw std::runtime_error(
                    "MIPS processor type not supported yet, please open a "
                    "GitHub issue to report that you want this feature!");
    }
    else
        throw std::runtime_error("Could not read the processor type");

    // Read parameter or group
    std::streampos nextParamByteInFile(
                static_cast<int>(file.tellg())
                + static_cast<int>(_parametersStart) - ezc3d::DATA_TYPE::BYTE);
    while (nextParamByteInFile)
    {
        // Check if we spontaneously got to the next parameter.
        // Otherwise c3d is messed up
        if (file.tellg() != nextParamByteInFile)
            throw std::ios_base::failure("Bad c3d formatting");

        // Nb of char in the group name, locked if negative,
        // 0 if we finished the section
        int nbCharInName(
                    c3d.readInt(
                        processorType(), file, 1*ezc3d::DATA_TYPE::BYTE));
        if (nbCharInName == 0)
            break;
        int id(c3d.readInt(processorType(), file, 1*ezc3d::DATA_TYPE::BYTE));

        // Make sure there at least enough group
        for (size_t i = _groups.size(); i < static_cast<size_t>(abs(id)); ++i)
            _groups.push_back(ezc3d::ParametersNS::GroupNS::Group());

        // Group ID always negative for groups
        // and positive parameter of group ID
        if (id < 0) {
            nextParamByteInFile = group(
                        static_cast<size_t>
                        (abs(id)-1)).read(c3d, *this, file, nbCharInName);
        }
        else {
            nextParamByteInFile = group(
                        static_cast<size_t>(id-1)).parameter(
                        c3d, *this, file, nbCharInName);
        }
    }

    // If some mandatory groups/parameters are not set by having a non
    // compliant C3D, fix it
    setMandatoryParameters();
}

void ezc3d::ParametersNS::Parameters::setMandatoryParameters() {
    // Mandatory groups
    {
        if (!isGroup("POINT")){
            group(ezc3d::ParametersNS::GroupNS::Group ("POINT"));
        }

        ezc3d::ParametersNS::GroupNS::Group& grp(group("POINT"));
        if (!grp.isParameter("USED")){
            ezc3d::ParametersNS::GroupNS::Parameter p("USED", "");
            p.set(0);
            p.lock();
            grp.parameter(p);
        }
        if (!grp.isParameter("SCALE")){
            ezc3d::ParametersNS::GroupNS::Parameter p("SCALE", "");
            p.set(-1.0);
            p.lock();
            grp.parameter(p);
        }
        if (!grp.isParameter("RATE")){
            ezc3d::ParametersNS::GroupNS::Parameter p("RATE", "");
            p.set(0.0);
            p.lock();
            grp.parameter(p);
        }
        if (!grp.isParameter("DATA_START")){
            ezc3d::ParametersNS::GroupNS::Parameter p("DATA_START", "");
            p.set(0);
            p.lock();
            grp.parameter(p);
        }
        if (!grp.isParameter("FRAMES")){
            ezc3d::ParametersNS::GroupNS::Parameter p("FRAMES", "");
            p.set(0);
            p.lock();
            grp.parameter(p);
        }
        if (!grp.isParameter("LABELS")){
            ezc3d::ParametersNS::GroupNS::Parameter p("LABELS", "");
            p.set(std::vector<std::string>()={});
            grp.parameter(p);
        }
        if (!grp.isParameter("DESCRIPTIONS")){
            ezc3d::ParametersNS::GroupNS::Parameter p("DESCRIPTIONS", "");
            p.set(std::vector<std::string>()={});
            grp.parameter(p);
        }
        if (!grp.isParameter("UNITS")){
            ezc3d::ParametersNS::GroupNS::Parameter p("UNITS", "");
            p.set(std::vector<std::string>()={});
            grp.parameter(p);
        }
    }
    {
        if (!isGroup("ANALOG")){
            group(ezc3d::ParametersNS::GroupNS::Group ("ANALOG"));
        }

        ezc3d::ParametersNS::GroupNS::Group& grp(group("ANALOG"));
        if (!grp.isParameter("USED")){
            ezc3d::ParametersNS::GroupNS::Parameter p("USED", "");
            p.set(0);
            p.lock();
            grp.parameter(p);
        }
        if (!grp.isParameter("LABELS")){
            ezc3d::ParametersNS::GroupNS::Parameter p("LABELS", "");
            p.set(std::vector<std::string>()={});
            grp.parameter(p);
        }
        if (!grp.isParameter("DESCRIPTIONS")){
            ezc3d::ParametersNS::GroupNS::Parameter p("DESCRIPTIONS", "");
            p.set(std::vector<std::string>()={});
            grp.parameter(p);
        }
        if (!grp.isParameter("GEN_SCALE")){
            ezc3d::ParametersNS::GroupNS::Parameter p("GEN_SCALE", "");
            p.set(1.0);
            grp.parameter(p);
        }
        if (!grp.isParameter("SCALE")){
            ezc3d::ParametersNS::GroupNS::Parameter p("SCALE", "");
            p.set(std::vector<double>()={});
            grp.parameter(p);
        }
        if (!grp.isParameter("OFFSET")){
            ezc3d::ParametersNS::GroupNS::Parameter p("OFFSET", "");
            p.set(std::vector<int>()={});
            grp.parameter(p);
        }
        if (!grp.isParameter("UNITS")){
            ezc3d::ParametersNS::GroupNS::Parameter p("UNITS", "");
            p.set(std::vector<std::string>()={});
            grp.parameter(p);
        }
        if (!grp.isParameter("RATE")){
            ezc3d::ParametersNS::GroupNS::Parameter p("RATE", "");
            p.set(0.0);
            p.lock();
            grp.parameter(p);
        }
        if (!grp.isParameter("FORMAT")){
            ezc3d::ParametersNS::GroupNS::Parameter p("FORMAT", "");
            p.set(std::vector<std::string>()={});
            grp.parameter(p);
        }
        if (!grp.isParameter("BITS")){
            ezc3d::ParametersNS::GroupNS::Parameter p("BITS", "");
            p.set(std::vector<int>()={});
            grp.parameter(p);
        }
    }
    {
        if (!isGroup("FORCE_PLATFORM")){
            group(ezc3d::ParametersNS::GroupNS::Group ("FORCE_PLATFORM"));
        }

        ezc3d::ParametersNS::GroupNS::Group& grp(group("FORCE_PLATFORM"));
        if (!grp.isParameter("USED")){
            ezc3d::ParametersNS::GroupNS::Parameter p("USED", "");
            p.set(0);
            grp.parameter(p);
        }
        if (!grp.isParameter("TYPE")){
            ezc3d::ParametersNS::GroupNS::Parameter p("TYPE", "");
            p.set(std::vector<int>()={});
            grp.parameter(p);
        }
        if (!grp.isParameter("ZERO")){
            ezc3d::ParametersNS::GroupNS::Parameter p("ZERO", "");
            p.set(std::vector<int>()={1,0});
            grp.parameter(p);
        }
        if (!grp.isParameter("CORNERS")){
            ezc3d::ParametersNS::GroupNS::Parameter p("CORNERS", "");
            p.set(std::vector<double>()={});
            grp.parameter(p);
        }
        if (!grp.isParameter("ORIGIN")){
            ezc3d::ParametersNS::GroupNS::Parameter p("ORIGIN", "");
            p.set(std::vector<double>()={});
            grp.parameter(p);
        }
        if (!grp.isParameter("CHANNEL")){
            ezc3d::ParametersNS::GroupNS::Parameter p("CHANNEL", "");
            p.set(std::vector<int>()={});
            grp.parameter(p);
        }
        if (!grp.isParameter("CAL_MATRIX")){
            ezc3d::ParametersNS::GroupNS::Parameter p("CAL_MATRIX", "");
            p.set(std::vector<double>()={});
            grp.parameter(p);
        }
    }
}

void ezc3d::ParametersNS::Parameters::print() const {
    std::cout << "Parameters header" << std::endl;
    std::cout << "parametersStart = " << parametersStart() << std::endl;
    std::cout << "nbParamBlock = " << nbParamBlock() << std::endl;
    std::cout << "processorType = " << processorType() << std::endl;

    for (size_t i = 0; i < nbGroups(); ++i){
        std::cout << "Group " << i << std::endl;
        group(i).print();
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void ezc3d::ParametersNS::Parameters::write(
        std::fstream &f,
        std::streampos &dataStartPosition) const {
    // Write the header of parameters
    f.write(reinterpret_cast<const char*>(&_parametersStart), ezc3d::BYTE);
    int checksum(0x50);
    f.write(reinterpret_cast<const char*>(&checksum), ezc3d::BYTE);
    // Leave a blank space which will be later fill
    // (number of block can't be known before writing them)
    std::streampos pos(f.tellg()); // remember where to input this value later
    int blankValue(0);
    f.write(reinterpret_cast<const char*>(&blankValue), ezc3d::BYTE);
    int processorType = PROCESSOR_TYPE::INTEL;
    f.write(reinterpret_cast<const char*>(&processorType), ezc3d::BYTE);

    // Write each groups
    for (size_t i=0; i < nbGroups(); ++i){
        const ezc3d::ParametersNS::GroupNS::Group& currentGroup(group(i));
        if (!currentGroup.isEmpty())
            currentGroup.write(f, -static_cast<int>(i+1), dataStartPosition);
    }

    // Move the cursor to a beginning of a block
    std::streampos currentPos(f.tellg());
    for (int i=0; i<512 - static_cast<int>(currentPos) % 512; ++i){
        f.write(reinterpret_cast<const char*>(&blankValue), ezc3d::BYTE);
    }
    // Go back at the left blank space and write the current position
    currentPos = f.tellg();
    f.seekg(pos);
    int nBlocksToNext = int(currentPos - pos-2)/512;
    if (int(currentPos - pos-2) % 512 > 0)
        ++nBlocksToNext;
    f.write(reinterpret_cast<const char*>(&nBlocksToNext), ezc3d::BYTE);
    f.seekg(currentPos);
}

size_t ezc3d::ParametersNS::Parameters::parametersStart() const {
    return _parametersStart;
}

size_t ezc3d::ParametersNS::Parameters::checksum() const {
    return _checksum;
}

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
        const std::string &groupName) const
{
    try {
        groupIdx(groupName);
        return true;
    } catch (std::invalid_argument) {
        return false;
    }
}

size_t ezc3d::ParametersNS::Parameters::groupIdx(
        const std::string &groupName) const {
    for (size_t i = 0; i < nbGroups(); ++i)
        if (!group(i).name().compare(groupName))
            return i;
    throw std::invalid_argument(
                "Parameters::groupIdx could not find " + groupName);
}

const ezc3d::ParametersNS::GroupNS::Group&
ezc3d::ParametersNS::Parameters::group(
        size_t idx) const {
    try {
        return _groups.at(idx);
    } catch(std::out_of_range) {
        throw std::out_of_range(
                    "Parameters::group method is trying to access the group "
                    + std::to_string(idx) +
                    " while the maximum number of groups is "
                    + std::to_string(nbGroups()) + ".");
    }
}

ezc3d::ParametersNS::GroupNS::Group&
ezc3d::ParametersNS::Parameters::group(
        size_t idx) {
    try {
        return _groups.at(idx);
    } catch(std::out_of_range) {
        throw std::out_of_range(
                    "Parameters::group method is trying to access the group "
                    + std::to_string(idx) +
                    " while the maximum number of groups is "
                    + std::to_string(nbGroups()) + ".");
    }
}

const ezc3d::ParametersNS::GroupNS::Group&
ezc3d::ParametersNS::Parameters::group(
        const std::string &groupName) const {
    return group(groupIdx(groupName));
}

ezc3d::ParametersNS::GroupNS::Group&
ezc3d::ParametersNS::Parameters::group(
        const std::string &groupName) {
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
        for (size_t i=0; i < g.nbParameters(); ++i)
            _groups[alreadyExtIdx].parameter(g.parameter(i));
    }
}

const std::vector<ezc3d::ParametersNS::GroupNS::Group>&
ezc3d::ParametersNS::Parameters::groups() const {
    return _groups;
}

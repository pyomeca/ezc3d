#define EZC3D_API_EXPORTS
#include "Parameters.h"
// Implementation of Parameters class


ezc3d::ParametersNS::Parameters::Parameters():
    _parametersStart(1),
    _checksum(0x50),
    _nbParamBlock(0),
    _processorType(84)
{
    // Mandatory groups
    {
        ezc3d::ParametersNS::GroupNS::Group grp("POINT", "");
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("USED", "");
            p.set(0);
            p.lock();
            grp.parameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("SCALE", "");
            p.set(-1.0);
            p.lock();
            grp.parameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("RATE", "");
            p.set(0.0);
            p.lock();
            grp.parameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("DATA_START", "");
            p.set(0);
            p.lock();
            grp.parameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("FRAMES", "");
            p.set(0);
            p.lock();
            grp.parameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("LABELS", "");
            p.set(std::vector<std::string>()={});
            grp.parameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("DESCRIPTIONS", "");
            p.set(std::vector<std::string>()={});
            grp.parameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("UNITS", "");
            p.set(std::vector<std::string>()={});
            grp.parameter(p);
        }
        group(grp);
    }
    {
        ezc3d::ParametersNS::GroupNS::Group grp("ANALOG", "");
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("USED", "");
            p.set(0);
            p.lock();
            grp.parameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("LABELS", "");
            p.set(std::vector<std::string>()={});
            grp.parameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("DESCRIPTIONS", "");
            p.set(std::vector<std::string>()={});
            grp.parameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("GEN_SCALE", "");
            p.set(1);
            grp.parameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("SCALE", "");
            p.set(std::vector<float>()={});
            grp.parameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("OFFSET", "");
            p.set(std::vector<int>()={});
            grp.parameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("UNITS", "");
            p.set(std::vector<std::string>()={});
            grp.parameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("RATE", "");
            p.set(0.0);
            p.lock();
            grp.parameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("FORMAT", "");
            p.set(std::vector<std::string>()={});
            grp.parameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("BITS", "");
            p.set(std::vector<int>()={});
            grp.parameter(p);
        }
        group(grp);
    }
    {
        ezc3d::ParametersNS::GroupNS::Group grp("FORCE_PLATFORM", "");
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("USED", "");
            p.set(0);
            grp.parameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("TYPE", "");
            p.set(std::vector<int>()={});
            grp.parameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("ZERO", "");
            p.set(std::vector<int>()={1,0});
            grp.parameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("CORNERS", "");
            p.set(std::vector<float>()={});
            grp.parameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("ORIGIN", "");
            p.set(std::vector<float>()={});
            grp.parameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("CHANNEL", "");
            p.set(std::vector<int>()={});
            grp.parameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("CAL_MATRIX", "");
            p.set(std::vector<float>()={});
            grp.parameter(p);
        }
        group(grp);
    }
}

ezc3d::ParametersNS::Parameters::Parameters(ezc3d::c3d &file) :
    _parametersStart(0),
    _checksum(0),
    _nbParamBlock(0),
    _processorType(0)
{
    // Read the Parameters Header
    _parametersStart = file.readUint(1*ezc3d::DATA_TYPE::BYTE, static_cast<int>(256*ezc3d::DATA_TYPE::WORD*(file.header().parametersAddress()-1)), std::ios::beg);
    _checksum = file.readUint(1*ezc3d::DATA_TYPE::BYTE);
    _nbParamBlock = file.readUint(1*ezc3d::DATA_TYPE::BYTE);
    _processorType = file.readUint(1*ezc3d::DATA_TYPE::BYTE);
    if (_checksum == 0 && _parametersStart == 0){
        // Theoritically, if this happens, this is a bad c3d formatting and should return an error, but for some reason
        // Qualisys decided that they would not comply to the standard. Therefore they put "_parameterStart" and "_checksum" to 0
        // This is a patch for Qualisys bad formatting c3d
        _parametersStart = 1;
        _checksum = 0x50;
    }
    if (_checksum != 0x50) // If checkbyte is wrong
        throw std::ios_base::failure("File must be a valid c3d file");

    // Read parameter or group
    std::streampos nextParamByteInFile(static_cast<int>(file.tellg()) + static_cast<int>(_parametersStart) - ezc3d::DATA_TYPE::BYTE);
    while (nextParamByteInFile)
    {
        // Check if we spontaneously got to the next parameter. Otherwise c3d is messed up
        if (file.tellg() != nextParamByteInFile)
            throw std::ios_base::failure("Bad c3d formatting");

        // Nb of char in the group name, locked if negative, 0 if we finished the section
        int nbCharInName(file.readInt(1*ezc3d::DATA_TYPE::BYTE));
        if (nbCharInName == 0)
            break;
        int id(file.readInt(1*ezc3d::DATA_TYPE::BYTE));

        // Make sure there at least enough group
        for (size_t i = _groups.size(); i < static_cast<size_t>(abs(id)); ++i)
            _groups.push_back(ezc3d::ParametersNS::GroupNS::Group());

        // Group ID always negative for groups and positive parameter of group ID
        if (id < 0)
            nextParamByteInFile = group_nonConst(static_cast<size_t>(abs(id)-1)).read(file, nbCharInName);
        else
            nextParamByteInFile = group_nonConst(static_cast<size_t>(id-1)).parameter(file, nbCharInName);
    }
}

void ezc3d::ParametersNS::Parameters::print() const
{
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

void ezc3d::ParametersNS::Parameters::write(std::fstream &f) const
{
    // Write the header of parameters
    f.write(reinterpret_cast<const char*>(&_parametersStart), ezc3d::BYTE);
    int checksum(0x50);
    f.write(reinterpret_cast<const char*>(&checksum), ezc3d::BYTE);
    // Leave a blank space which will be later fill
    // (number of block can't be known before writing them)
    std::streampos pos(f.tellg()); // remember where to input this value later
    int blankValue(0);
    f.write(reinterpret_cast<const char*>(&blankValue), ezc3d::BYTE);
    int processorType = 84;
    f.write(reinterpret_cast<const char*>(&processorType), ezc3d::BYTE);

    // Write each groups
    std::streampos dataStartPosition; // Special parameter in POINT group
    for (size_t i=0; i < nbGroups(); ++i)
        group(i).write(f, -static_cast<int>(i+1), dataStartPosition);

    // Move the cursor to a beginning of a block
    std::streampos actualPos(f.tellg());
    for (int i=0; i<512 - static_cast<int>(actualPos) % 512; ++i){
        f.write(reinterpret_cast<const char*>(&blankValue), ezc3d::BYTE);
    }
    // Go back at the left blank space and write the actual position
    actualPos = f.tellg();
    f.seekg(pos);
    int nBlocksToNext = int(actualPos - pos-2)/512;
    if (int(actualPos - pos-2) % 512 > 0)
        ++nBlocksToNext;
    f.write(reinterpret_cast<const char*>(&nBlocksToNext), ezc3d::BYTE);
    f.seekg(actualPos);

    // Go back to data start blank space and write the actual position
    actualPos = f.tellg();
    f.seekg(dataStartPosition);
    nBlocksToNext = int(actualPos)/512;
    if (int(actualPos) % 512 > 0)
        ++nBlocksToNext;
    f.write(reinterpret_cast<const char*>(&nBlocksToNext), ezc3d::BYTE);
    f.seekg(actualPos);
}

size_t ezc3d::ParametersNS::Parameters::parametersStart() const
{
    return _parametersStart;
}

size_t ezc3d::ParametersNS::Parameters::checksum() const
{
    return _checksum;
}

size_t ezc3d::ParametersNS::Parameters::nbParamBlock() const
{
    return _nbParamBlock;
}

size_t ezc3d::ParametersNS::Parameters::processorType() const
{
    return _processorType;
}

size_t ezc3d::ParametersNS::Parameters::nbGroups() const
{
    return _groups.size();
}

size_t ezc3d::ParametersNS::Parameters::groupIdx(const std::string &groupName) const
{
    for (size_t i = 0; i < nbGroups(); ++i)
        if (!group(i).name().compare(groupName))
            return i;
    throw std::invalid_argument("Parameters::groupIdx could not find " + groupName);
}

const ezc3d::ParametersNS::GroupNS::Group &ezc3d::ParametersNS::Parameters::group(size_t idx) const
{
    try {
        return _groups.at(idx);
    } catch(std::out_of_range) {
        throw std::out_of_range("Parameters::group method is trying to access the group "
                                + std::to_string(idx) +
                                " while the maximum number of groups is "
                                + std::to_string(nbGroups()) + ".");
    }
}

ezc3d::ParametersNS::GroupNS::Group &ezc3d::ParametersNS::Parameters::group_nonConst(size_t idx)
{
    try {
        return _groups.at(idx);
    } catch(std::out_of_range) {
        throw std::out_of_range("Parameters::group method is trying to access the group "
                                + std::to_string(idx) +
                                " while the maximum number of groups is "
                                + std::to_string(nbGroups()) + ".");
    }
}

const ezc3d::ParametersNS::GroupNS::Group &ezc3d::ParametersNS::Parameters::group(const std::string &groupName) const
{
    return group(groupIdx(groupName));
}

ezc3d::ParametersNS::GroupNS::Group &ezc3d::ParametersNS::Parameters::group_nonConst(const std::string &groupName)
{
    return group_nonConst(groupIdx(groupName));
}

void ezc3d::ParametersNS::Parameters::group(const ezc3d::ParametersNS::GroupNS::Group &g)
{
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

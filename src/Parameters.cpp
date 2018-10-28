#define EZC3D_API_EXPORTS
#include "Parameters.h"

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
            grp.addParameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("SCALE", "");
            p.set(-1.0);
            p.lock();
            grp.addParameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("RATE", "");
            p.set(0.0);
            p.lock();
            grp.addParameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("DATA_START", "");
            p.set(0);
            p.lock();
            grp.addParameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("FRAMES", "");
            p.set(0);
            p.lock();
            grp.addParameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("LABELS", "");
            p.set(std::vector<std::string>()={});
            grp.addParameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("DESCRIPTIONS", "");
            p.set(std::vector<std::string>()={});
            grp.addParameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("UNITS", "");
            p.set(std::vector<std::string>()={});
            grp.addParameter(p);
        }
        addGroup(grp);
    }
    {
        ezc3d::ParametersNS::GroupNS::Group grp("ANALOG", "");
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("USED", "");
            p.set(0);
            p.lock();
            grp.addParameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("LABELS", "");
            p.set(std::vector<std::string>()={});
            grp.addParameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("DESCRIPTIONS", "");
            p.set(std::vector<std::string>()={});
            grp.addParameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("GEN_SCALE", "");
            p.set(1);
            grp.addParameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("SCALE", "");
            p.set(std::vector<float>()={});
            grp.addParameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("OFFSET", "");
            p.set(std::vector<int>()={});
            grp.addParameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("UNITS", "");
            p.set(std::vector<std::string>()={});
            grp.addParameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("RATE", "");
            p.set(0.0);
            p.lock();
            grp.addParameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("FORMAT", "");
            p.set(std::vector<std::string>()={});
            grp.addParameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("BITS", "");
            p.set(std::vector<int>()={});
            grp.addParameter(p);
        }
        addGroup(grp);
    }
    {
        ezc3d::ParametersNS::GroupNS::Group grp("FORCE_PLATFORM", "");
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("USED", "");
            p.set(0);
            grp.addParameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("TYPE", "");
            p.set(std::vector<int>()={});
            grp.addParameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("ZERO", "");
            p.set(std::vector<int>()={1,0});
            grp.addParameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("CORNERS", "");
            p.set(std::vector<float>()={});
            grp.addParameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("ORIGIN", "");
            p.set(std::vector<float>()={});
            grp.addParameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("CHANNEL", "");
            p.set(std::vector<int>()={});
            grp.addParameter(p);
        }
        {
            ezc3d::ParametersNS::GroupNS::Parameter p("CAL_MATRIX", "");
            p.set(std::vector<float>()={});
            grp.addParameter(p);
        }
        addGroup(grp);
    }
}

ezc3d::ParametersNS::Parameters::Parameters(ezc3d::c3d &file) :
    _parametersStart(0),
    _checksum(0),
    _nbParamBlock(0),
    _processorType(0)
{
    // Read the Parameters Header
    _parametersStart = file.readInt(1*ezc3d::DATA_TYPE::BYTE, 256*ezc3d::DATA_TYPE::WORD*(file.header().parametersAddress()-1), std::ios::beg);
    _checksum = file.readInt(1*ezc3d::DATA_TYPE::BYTE);
    _nbParamBlock = file.readInt(1*ezc3d::DATA_TYPE::BYTE);
    _processorType = file.readInt(1*ezc3d::DATA_TYPE::BYTE);
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
    std::streampos nextParamByteInFile(static_cast<int>(file.tellg()) + _parametersStart - ezc3d::DATA_TYPE::BYTE);
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
        for (int i = static_cast<int>(_groups.size()); i < abs(id); ++i)
            _groups.push_back(ezc3d::ParametersNS::GroupNS::Group());

        // Group ID always negative for groups and positive parameter of group ID
        if (id < 0)
            nextParamByteInFile = group_nonConst(abs(id)-1).read(file, nbCharInName);
        else
            nextParamByteInFile = group_nonConst(id-1).addParameter(file, nbCharInName);
    }
}
int ezc3d::ParametersNS::Parameters::processorType() const
{
    return _processorType;
}
int ezc3d::ParametersNS::Parameters::nbParamBlock() const
{
    return _nbParamBlock;
}
int ezc3d::ParametersNS::Parameters::checksum() const
{
    return _checksum;
}
int ezc3d::ParametersNS::Parameters::parametersStart() const
{
    return _parametersStart;
}
void ezc3d::ParametersNS::Parameters::print() const
{
    std::cout << "Parameters header" << std::endl;
    std::cout << "parametersStart = " << parametersStart() << std::endl;
    std::cout << "nbParamBlock = " << nbParamBlock() << std::endl;
    std::cout << "processorType = " << processorType() << std::endl;

    for (int i = 0; i < static_cast<int>(groups().size()); ++i){
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
    for (int i=0; i<static_cast<int>(groups().size()); ++i)
        group(i).write(f, -(i+1), dataStartPosition);

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




ezc3d::ParametersNS::GroupNS::Group::Group(const std::string &name, const std::string &description) :
    _isLocked(false),
    _name(name),
    _description(description)
{

}
const std::vector<ezc3d::ParametersNS::GroupNS::Group>& ezc3d::ParametersNS::Parameters::groups() const
{
    return _groups;
}
ezc3d::ParametersNS::GroupNS::Group &ezc3d::ParametersNS::Parameters::group_nonConst(int group)
{
    return _groups[static_cast<unsigned int>(group)];
}

void ezc3d::ParametersNS::Parameters::addGroup(const ezc3d::ParametersNS::GroupNS::Group &g)
{
    // If the group already exist, override and merge
    int alreadyExtIdx(groupIdx(g.name()));
    if (alreadyExtIdx < 0)
        _groups.push_back(g);
    else {
        for (int i=0; i<static_cast<int>(g.parameters().size()); ++i)
            _groups[static_cast<unsigned int>(alreadyExtIdx)].addParameter(g.parameter(i));
    }

}
const ezc3d::ParametersNS::GroupNS::Group &ezc3d::ParametersNS::Parameters::group(int group) const
{
    return _groups[static_cast<unsigned int>(group)];
}
const ezc3d::ParametersNS::GroupNS::Group &ezc3d::ParametersNS::Parameters::group(const std::string &groupName) const
{
    int idx(groupIdx(groupName));
    if (idx < 0)
        throw std::invalid_argument("Group name was not found in parameters");
    return group(idx);
}

int ezc3d::ParametersNS::Parameters::groupIdx(const std::string &groupName) const
{
    for (int i = 0; i < static_cast<int>(groups().size()); ++i){
        if (!group(i).name().compare(groupName))
            return i;
    }
    return -1;
}
void ezc3d::ParametersNS::GroupNS::Group::lock()
{
    _isLocked = true;
}
void ezc3d::ParametersNS::GroupNS::Group::unlock()
{
    _isLocked = false;
}
bool ezc3d::ParametersNS::GroupNS::Group::isLocked() const
{
    return _isLocked;
}
const std::string& ezc3d::ParametersNS::GroupNS::Group::description() const
{
    return _description;
}
const std::string& ezc3d::ParametersNS::GroupNS::Group::name() const
{
    return _name;
}
int ezc3d::ParametersNS::GroupNS::Group::read(ezc3d::c3d &file, int nbCharInName)
{
    if (nbCharInName < 0)
        _isLocked = true;
    else
        _isLocked = false;

    // Read name of the group
    _name.assign(file.readString(static_cast<unsigned int>(abs(nbCharInName) * ezc3d::DATA_TYPE::BYTE)));

    // number of byte to the next group from here
    int offsetNext(static_cast<int>(file.readUint(2*ezc3d::DATA_TYPE::BYTE)));
    // Compute the position of the element in the file
    int nextParamByteInFile;
    if (offsetNext == 0)
        nextParamByteInFile = 0;
    else
        nextParamByteInFile = static_cast<int>(file.tellg()) + offsetNext - ezc3d::DATA_TYPE::WORD;

    // Byte 5+nbCharInName ==> Number of characters in group description
    int nbCharInDesc(file.readInt(1*ezc3d::DATA_TYPE::BYTE));
    // Byte 6+nbCharInName ==> Group description
    if (nbCharInDesc)
        _description = file.readString(static_cast<unsigned int>(nbCharInDesc));

    // Return how many bytes
    return nextParamByteInFile;
}
int ezc3d::ParametersNS::GroupNS::Group::addParameter(ezc3d::c3d &file, int nbCharInName)
{
    ezc3d::ParametersNS::GroupNS::Parameter p;
    int nextParamByteInFile = p.read(file, nbCharInName);
    addParameter(p);
    return nextParamByteInFile;
}

void ezc3d::ParametersNS::GroupNS::Group::addParameter(const ezc3d::ParametersNS::GroupNS::Parameter &p)
{
    if (p.type() == ezc3d::DATA_TYPE::NONE)
        throw std::runtime_error("Data type is not set");

    int alreadyExistIdx(-1);
    for (int i=0; i<static_cast<int>(_parameters.size()); ++i)
        if (!parameter(i).name().compare(p.name())){
            alreadyExistIdx = i;
            break;
        }
    if (alreadyExistIdx < 0)
        _parameters.push_back(p);
    else
        _parameters[static_cast<unsigned int>(alreadyExistIdx)] = p;
}
void ezc3d::ParametersNS::GroupNS::Group::print() const
{
    std::cout << "groupName = " << name() << std::endl;
    std::cout << "isLocked = " << isLocked() << std::endl;
    std::cout << "desc = " << description() << std::endl;

    for (int i=0; i<static_cast<int>(parameters().size()); ++i){
        std::cout << "Parameter " << i << std::endl;
        parameter(i).print();
    }
}
void ezc3d::ParametersNS::GroupNS::Group::write(std::fstream &f, int groupIdx, std::streampos &dataStartPosition) const
{
    int nCharName(static_cast<int>(name().size()));
    if (isLocked())
        nCharName *= -1;
    f.write(reinterpret_cast<const char*>(&nCharName), 1*ezc3d::DATA_TYPE::BYTE);
    if (isLocked())
        nCharName *= -1;
    f.write(reinterpret_cast<const char*>(&groupIdx), 1*ezc3d::DATA_TYPE::BYTE);
    f.write(ezc3d::toUpper(name()).c_str(), nCharName*ezc3d::DATA_TYPE::BYTE);

    // It is not possible already to know in how many bytes the next parameter is
    int blank(0);
    std::streampos pos(f.tellg());
    f.write(reinterpret_cast<const char*>(&blank), 2*ezc3d::DATA_TYPE::BYTE);

    int nCharGroupDescription(static_cast<int>(description().size()));
    f.write(reinterpret_cast<const char*>(&nCharGroupDescription), 1*ezc3d::DATA_TYPE::BYTE);
    f.write(description().c_str(), nCharGroupDescription*ezc3d::DATA_TYPE::BYTE);

    std::streampos actualPos(f.tellg());
    // Go back at the left blank space and write the actual position
    f.seekg(pos);
    int nCharToNext = int(actualPos - pos);
    f.write(reinterpret_cast<const char*>(&nCharToNext), 2*ezc3d::DATA_TYPE::BYTE);
    f.seekg(actualPos);

    for (int i=0; i<static_cast<int>(parameters().size()); ++i)
        parameter(i).write(f, -groupIdx, dataStartPosition);

}
const std::vector<ezc3d::ParametersNS::GroupNS::Parameter>& ezc3d::ParametersNS::GroupNS::Group::parameters() const
{
    return _parameters;
}

const ezc3d::ParametersNS::GroupNS::Parameter &ezc3d::ParametersNS::GroupNS::Group::parameter(int idx) const
{
    if (idx < 0 || idx >= static_cast<int>(_parameters.size()))
        throw std::out_of_range("Wrong number of parameter");
    return _parameters[static_cast<unsigned int>(idx)];
}
std::vector<ezc3d::ParametersNS::GroupNS::Parameter>& ezc3d::ParametersNS::GroupNS::Group::parameters_nonConst()
{
    return _parameters;
}
int ezc3d::ParametersNS::GroupNS::Group::parameterIdx(std::string parameterName) const
{
    for (int i = 0; i < static_cast<int>(parameters().size()); ++i){
        if (!parameter(i).name().compare(parameterName))
            return i;
    }
    return -1;
}
const ezc3d::ParametersNS::GroupNS::Parameter &ezc3d::ParametersNS::GroupNS::Group::parameter(std::string parameterName) const
{
    int idx(parameterIdx(parameterName));
    if (idx < 0)
        throw std::invalid_argument("Parameter name was not found within the group");
    return parameter(idx);
}





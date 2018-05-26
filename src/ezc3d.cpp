#define EZC3D_API_EXPORTS
#include "ezc3d.h"

ezc3d::c3d::c3d(const std::string &filePath):
    std::fstream(filePath, std::ios::in | std::ios::binary),
    _filePath(filePath)
{
    if (!is_open())
        throw std::ios_base::failure("Could not open the c3d file");

    // Read all the section
    _header = std::shared_ptr<ezc3d::Header>(new ezc3d::Header(*this));
    _parameters = std::shared_ptr<ezc3d::ParametersNS::Parameters>(new ezc3d::ParametersNS::Parameters(*this));
    _data = std::shared_ptr<ezc3d::DataNS::Data>(new ezc3d::DataNS::Data(*this));
}


ezc3d::c3d::~c3d()
{
    close();
}

void ezc3d::removeSpacesOfAString(std::string& s){
    // Remove the spaces at the end of the strings
    for (size_t i = s.size(); i >= 0; --i)
        if (s[s.size()-1] == ' ')
            s.pop_back();
        else
            break;
}


unsigned int ezc3d::c3d::hex2uint(const char * val){
    int ret(0);
    for (int i=0; i<strlen(val); i++)
        ret |= int((unsigned char)val[i]) * int(pow(0x100, i));
    return ret;
}

int ezc3d::c3d::hex2int(const char * val){
    unsigned int tp(hex2uint(val));

    // convert to signed int
    // Find max int value
    unsigned int max(0);
    for (int i=0; i<strlen(val); ++i)
        max |= 0xFF * int(pow(0x100, i));

    // If the value is over uint_max / 2 then it is a negative number
    int out;
    if (tp > max / 2)
        out = (int)(tp - max - 1);
    else
        out = tp;

    return out;
}

int ezc3d::c3d::hex2long(const char * val){
    long ret(0);
    for (int i=0; i<strlen(val); i++)
        ret |= long((unsigned char)val[i]) * long(pow(0x100, i));
    return ret;
}

void ezc3d::c3d::readFile(int nByteToRead, char * c, int nByteFromPrevious,
                     const  std::ios_base::seekdir &pos)
{
    if (pos != 1)
        this->seekg (nByteFromPrevious, pos); // Move to number analogs
    this->read (c, nByteToRead);
    c[nByteToRead] = '\0'; // Make sure last char is NULL
}
void ezc3d::c3d::readChar(int nByteToRead, char * c,int nByteFromPrevious,
                     const  std::ios_base::seekdir &pos)
{
    c = new char[nByteToRead + 1];
    readFile(nByteToRead, c, nByteFromPrevious, pos);
}

std::string ezc3d::c3d::readString(int nByteToRead, int nByteFromPrevious,
                              const std::ios_base::seekdir &pos)
{
    char* c = new char[nByteToRead + 1];
    readFile(nByteToRead, c, nByteFromPrevious, pos);
    return std::string(c);
}

int ezc3d::c3d::readInt(int nByteToRead, int nByteFromPrevious,
            const std::ios_base::seekdir &pos)
{
    char* c = new char[nByteToRead + 1];
    readFile(nByteToRead, c, nByteFromPrevious, pos);

    // make sure it is an int and not an unsigned int
    return hex2int(c);
}

int ezc3d::c3d::readUint(int nByteToRead, int nByteFromPrevious,
            const std::ios_base::seekdir &pos)
{
    char* c = new char[nByteToRead + 1];
    readFile(nByteToRead, c, nByteFromPrevious, pos);

    // make sure it is an int and not an unsigned int
    return hex2uint(c);
}

float ezc3d::c3d::readFloat(int nByteFromPrevious,
                const std::ios_base::seekdir &pos)
{
    int nByteToRead(4*ezc3d::READ_SIZE::BYTE);
    char* c = new char[nByteToRead + 1];
    readFile(nByteToRead, c, nByteFromPrevious, pos);
    return *reinterpret_cast<float*>(c);
}

long ezc3d::c3d::readLong(int nByteToRead,
              int nByteFromPrevious,
              const  std::ios_base::seekdir &pos)
{
    char* c = new char[nByteToRead + 1];
    readFile(nByteToRead, c, nByteFromPrevious, pos);
    return hex2long(c);
}

void ezc3d::c3d::readMatrix(int dataLenghtInBytes, std::vector<int> dimension,
                       std::vector<int> &param_data, int currentIdx)
{
    for (int i=0; i<dimension[currentIdx]; ++i)
        if (currentIdx == dimension.size()-1)
            param_data.push_back (readInt(dataLenghtInBytes*ezc3d::READ_SIZE::BYTE));
        else
            readMatrix(dataLenghtInBytes, dimension, param_data, currentIdx + 1);
}

void ezc3d::c3d::readMatrix(std::vector<int> dimension,
                       std::vector<float> &param_data, int currentIdx)
{
    for (int i=0; i<dimension[currentIdx]; ++i)
        if (currentIdx == dimension.size()-1)
            param_data.push_back (readFloat());
        else
            readMatrix(dimension, param_data, currentIdx + 1);
}

void ezc3d::c3d::readMatrix(std::vector<int> dimension,
                       std::vector<std::string> &param_data, int currentIdx)
{
    for (int i=0; i<dimension[currentIdx]; ++i)
        if (currentIdx == dimension.size()-1)
            param_data.push_back(readString(ezc3d::READ_SIZE::BYTE));
        else
            readMatrix(dimension, param_data, currentIdx + 1);
}

const ezc3d::Header& ezc3d::c3d::header() const
{
    return *_header;
}

const ezc3d::ParametersNS::Parameters& ezc3d::c3d::parameters() const
{
    return *_parameters;
}

const ezc3d::DataNS::Data& ezc3d::c3d::data() const
{
    return *_data;
}


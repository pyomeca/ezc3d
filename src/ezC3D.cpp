#include "ezC3D.h"

ezC3D::C3D::C3D(const std::string &filePath):
    std::fstream(filePath, std::ios::in | std::ios::binary),
    _filePath(filePath)
{
    if (!is_open())
        throw std::ios_base::failure("Could not open the C3D file");

    // Read all the section
    _header = std::shared_ptr<ezC3D::Header>(new ezC3D::Header(*this));
    _parameters = std::shared_ptr<ezC3D::ParametersNS::Parameters>(new ezC3D::ParametersNS::Parameters(*this));
    _data = std::shared_ptr<ezC3D::DataNS::Data>(new ezC3D::DataNS::Data(*this));
}


ezC3D::C3D::~C3D()
{
    close();
}




unsigned int ezC3D::C3D::hex2uint(const char * val){
    int ret(0);
    for (int i=0; i<strlen(val); i++)
        ret |= int((unsigned char)val[i]) * int(pow(0x100, i));
    return ret;
}

int ezC3D::C3D::hex2int(const char * val){
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

int ezC3D::C3D::hex2long(const char * val){
    long ret(0);
    for (int i=0; i<strlen(val); i++)
        ret |= long((unsigned char)val[i]) * long(pow(0x100, i));
    return ret;
}

void ezC3D::C3D::readFile(int nByteToRead, char * c, int nByteFromPrevious,
                     const  std::ios_base::seekdir &pos)
{
    this->seekg (nByteFromPrevious, pos); // Move to number analogs
    this->read (c, nByteToRead);
    c[nByteToRead] = '\0'; // Make sure last char is NULL
}

void ezC3D::C3D::readChar(int nByteToRead, char * c,int nByteFromPrevious,
                     const  std::ios_base::seekdir &pos)
{
    c = new char[nByteToRead + 1];
    readFile(nByteToRead, c, nByteFromPrevious, pos);
}

std::string ezC3D::C3D::readString(int nByteToRead, int nByteFromPrevious,
                              const std::ios_base::seekdir &pos)
{
    char c[nByteToRead + 1];
    readFile(nByteToRead, c, nByteFromPrevious, pos);
    return std::string(c);
}

int ezC3D::C3D::readInt(int nByteToRead, int nByteFromPrevious,
            const std::ios_base::seekdir &pos)
{
    char c[nByteToRead + 1];
    readFile(nByteToRead, c, nByteFromPrevious, pos);

    // make sure it is an int and not an unsigned int
    return hex2int(c);
}

int ezC3D::C3D::readUint(int nByteToRead, int nByteFromPrevious,
            const std::ios_base::seekdir &pos)
{
    char c[nByteToRead + 1];
    readFile(nByteToRead, c, nByteFromPrevious, pos);

    // make sure it is an int and not an unsigned int
    return hex2uint(c);
}

float ezC3D::C3D::readFloat(int nByteFromPrevious,
                const std::ios_base::seekdir &pos)
{
    int nByteToRead(4*ezC3D::READ_SIZE::BYTE);
    char c[nByteToRead + 1];
    readFile(nByteToRead, c, nByteFromPrevious, pos);
    float coucou = *reinterpret_cast<float*>(c);
    return coucou;
}

long ezC3D::C3D::readLong(int nByteToRead,
              int nByteFromPrevious,
              const  std::ios_base::seekdir &pos)
{
    char c[nByteToRead + 1];
    readFile(nByteToRead, c, nByteFromPrevious, pos);
    return hex2long(c);
}

void ezC3D::C3D::readMatrix(int dataLenghtInBytes, std::vector<int> dimension,
                       std::vector<int> &param_data, int currentIdx)
{
    for (int i=0; i<dimension[currentIdx]; ++i)
        if (currentIdx == dimension.size()-1)
            param_data.push_back (readInt(dataLenghtInBytes*ezC3D::READ_SIZE::BYTE));
        else
            readMatrix(dataLenghtInBytes, dimension, param_data, currentIdx + 1);
}

void ezC3D::C3D::readMatrix(std::vector<int> dimension,
                       std::vector<float> &param_data, int currentIdx)
{
    for (int i=0; i<dimension[currentIdx]; ++i)
        if (currentIdx == dimension.size()-1)
            param_data.push_back (readFloat());
        else
            readMatrix(dimension, param_data, currentIdx + 1);
}

void ezC3D::C3D::readMatrix(std::vector<int> dimension,
                       std::vector<std::string> &param_data, int currentIdx)
{
    for (int i=0; i<dimension[currentIdx]; ++i)
        if (currentIdx == dimension.size()-1)
            param_data.push_back(readString(ezC3D::READ_SIZE::BYTE));
        else
            readMatrix(dimension, param_data, currentIdx + 1);
}

const ezC3D::Header& ezC3D::C3D::header() const
{
    return *_header;
}

const ezC3D::ParametersNS::Parameters& ezC3D::C3D::parameters() const
{
    return *_parameters;
}

const ezC3D::DataNS::Data& ezC3D::C3D::data() const
{
    return *_data;
}


#ifndef __EZC3D_H__
#define __EZC3D_H__

#include <iostream>
#include <fstream>
#include <vector>
#include <string.h>
#include <cmath>
#include <stdexcept>
#include <memory>

class ezC3D : public std::fstream{
public:
    ezC3D(const std::string &filePath);


    // Byte reading functions
    void readChar(int nByteToRead,
                  char * c,
                  int nByteFromPrevious = 0,
                  const  std::ios_base::seekdir &pos = std::ios::cur);
    std::string readString(int nByteToRead, int nByteFromPrevious = 0,
                           const std::ios_base::seekdir &pos = std::ios::cur);
    int readInt(int nByteToRead,
                int nByteFromPrevious = 0,
                const std::ios_base::seekdir &pos = std::ios::cur);
    int readUint(int nByteToRead,
                int nByteFromPrevious = 0,
                const std::ios_base::seekdir &pos = std::ios::cur);
    float readFloat(int nByteFromPrevious = 0,
                    const std::ios_base::seekdir &pos = std::ios::cur);
    long readLong(int nByteToRead,
                  int nByteFromPrevious = 0,
                  const  std::ios_base::seekdir &pos = std::ios::cur);
    void readMatrix(int dataLenghtInBytes,
                    std::vector<int> dimension,
                    std::vector<int> &param_data,
                    int currentIdx = 0);
    void readMatrix(std::vector<int> dimension,
                    std::vector<std::string> &param_data,
                    int currentIdx = 0);

    class Header;
    class Parameter;

    class Point3d;
    class Analog;
    class Frame;
    // Size of some constant (in Byte)
    enum READ_SIZE{
        BYTE = 1,
        WORD = 2
    };
    const std::shared_ptr<Header>& header() const;
    const std::shared_ptr<Parameter>& parameters() const;
    const std::vector<Frame>& frames() const;

protected:
    std::string _filePath; // Remember the file path

    // Holder of data
    std::shared_ptr<Header> _header;
    std::shared_ptr<Parameter> _parameters;
    std::vector<Frame> _frames;

    // Internal reading function
    void readFile(int nByteToRead,
                  char * c,
                  int nByteFromPrevious = 0,
                  const  std::ios_base::seekdir &pos = std::ios::cur);

    // Converting functions
    unsigned int hex2uint(const char * val);
    int hex2int(const char * val);
    int hex2long(const char * val);
};
#include <DataHolder.h>
#include <SectionReader.h>

#endif

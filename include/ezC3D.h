#ifndef __EZC3D_H__
#define __EZC3D_H__

#include <iostream>
#include <fstream>
#include <vector>
#include <string.h>
#include <cmath>
#include <stdexcept>
#include <memory>


namespace ezC3D {
    // Size of some constant (in Byte)
    enum READ_SIZE{
        BYTE = 1,
        WORD = 2
    };

    class C3D;
    class Header;
    class Data;

    namespace Parameters {
        class Parameters;
        namespace Group {
            class Group;
            class Parameter;
        }
    }
}

class ezC3D::C3D : public std::fstream{
public:
    C3D(const std::string &filePath);
    C3D(const char* filePath);
    ~C3D();


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
    void readMatrix(std::vector<int> dimension,
                    std::vector<std::string> &param_data,
                    int currentIdx = 0);
    void readMatrix(int dataLenghtInBytes,
                    std::vector<int> dimension,
                    std::vector<int> &param_data,
                    int currentIdx = 0);
    void readMatrix(std::vector<int> dimension,
                    std::vector<float> &param_data,
                    int currentIdx = 0);

    const ezC3D::Header& header() const;
    const ezC3D::Parameters::Parameters& parameters() const;
    const ezC3D::Data& data() const;

protected:
    std::string _filePath; // Remember the file path

    // Holder of data
    std::shared_ptr<ezC3D::Header> _header;
    std::shared_ptr<ezC3D::Parameters::Parameters> _parameters;
    std::shared_ptr<ezC3D::Data> _data;

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
#include "Header.h"
#include "Data.h"
#include "Parameters.h"

#endif

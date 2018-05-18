#ifndef __EZC3D_H__
#define __EZC3D_H__

#include <iostream>
#include <fstream>
#include <vector>
#include <string.h>
#include <cmath>
#include <stdexcept>
#include <memory>

namespace ezc3d {
    // Size of some constant (in Byte)
    enum READ_SIZE{
        BYTE = 1,
        WORD = 2
    };
    void removeSpacesOfAString(std::string& s);

    class c3d;
    class Header;

    namespace ParametersNS {
        class Parameters;
        namespace GroupNS {
            class Group;
            class Parameter;
        }
    }

    namespace DataNS {
        class Data;

        class Frame;
        namespace Points3dNS {
            class Points;
            class Point;
        }
        namespace AnalogsNS {
            class Analogs;
            class SubFrame;
            class Channel;
        }
    }
}

class ezc3d::c3d : public std::fstream{
public:
    c3d(const std::string &filePath);
    ~c3d();


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

    const ezc3d::Header& header() const;
    const ezc3d::ParametersNS::Parameters& parameters() const;
    const ezc3d::DataNS::Data& data() const;

protected:
    std::string _filePath; // Remember the file path

    // Holder of data
    std::shared_ptr<ezc3d::Header> _header;
    std::shared_ptr<ezc3d::ParametersNS::Parameters> _parameters;
    std::shared_ptr<ezc3d::DataNS::Data> _data;

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

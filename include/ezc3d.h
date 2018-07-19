#ifndef __EZC3D_H__
#define __EZC3D_H__

#ifdef _WIN32
#  ifdef EZC3D_API_EXPORTS
#    define EZC3D_API __declspec(dllexport)
#  else
#    define EZC3D_API __declspec(dllimport)
#  endif
#else
#  define EZC3D_API
#endif


#include <iostream>
#include <fstream>
#include <vector>
#include <string.h>
#include <cmath>
#include <stdexcept>
#include <memory>
#ifdef _WIN32
#include <string>
#endif

namespace ezc3d {
    // Size of some constant (in Byte)
    enum READ_SIZE {
        BYTE = 1,
        WORD = 2
    };
    EZC3D_API void removeSpacesOfAString(std::string& s);

    class c3d;
    class EZC3D_API Header;

    namespace ParametersNS {
        class EZC3D_API Parameters;
        namespace GroupNS {
                class EZC3D_API Group;
                class EZC3D_API Parameter;
            }
    }

    namespace DataNS {
        class EZC3D_API Data;

        class Frame;
        namespace Points3dNS {
            class EZC3D_API Points;
            class EZC3D_API Point;
        }
        namespace AnalogsNS {
            class EZC3D_API Analogs;
            class EZC3D_API SubFrame;
            class EZC3D_API Channel;
        }
    }
}

class EZC3D_API ezc3d::c3d : public std::fstream {
public:
    c3d(const std::string &filePath);
    ~c3d();

    // Write the c3d to a file
    void write(const std::string &filePath) const;

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

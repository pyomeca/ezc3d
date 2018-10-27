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
#include <algorithm>
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
    enum DATA_TYPE{
        CHAR = -1,
        BYTE = 1,
        INT = 2,
        WORD = 2,
        FLOAT = 4,
        NONE = 10000
    };
    EZC3D_API void removeSpacesOfAString(std::string& s);
    EZC3D_API std::string toUpper(const std::string &str);

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

///
/// \brief Main class for C3D holder
///
class EZC3D_API ezc3d::c3d : public std::fstream {
public:
    c3d();
    c3d(const std::string &filePath);
    virtual ~c3d();
    void updateHeader();
    void updateParameters(const std::vector<std::string> &newMarkers = std::vector<std::string>(), const std::vector<std::string> &newAnalogs = std::vector<std::string>());

    // Write the c3d to a file
    void write(const std::string &filePath) const;
    void print() const;

    // Byte reading functions
    std::string readString(unsigned int nByteToRead, int nByteFromPrevious = 0,
                           const std::ios_base::seekdir &pos = std::ios::cur);
    int readInt(unsigned int nByteToRead,
                int nByteFromPrevious = 0,
                const std::ios_base::seekdir &pos = std::ios::cur);
    int readUint(unsigned int nByteToRead,
                 int nByteFromPrevious = 0,
                 const std::ios_base::seekdir &pos = std::ios::cur);
    float readFloat(int nByteFromPrevious = 0,
                    const std::ios_base::seekdir &pos = std::ios::cur);
    void readMatrix(const std::vector<int> &dimension,
                    std::vector<std::string> &param_data);
    void readMatrix(unsigned int dataLenghtInBytes,
                    const std::vector<int> &dimension,
                    std::vector<int> &param_data,
                    size_t currentIdx = 0);
    void readMatrix(const std::vector<int> &dimension,
                    std::vector<float> &param_data,
                    size_t currentIdx = 0);

    const ezc3d::Header& header() const;
    const ezc3d::ParametersNS::Parameters& parameters() const;
    const ezc3d::DataNS::Data& data() const;

    // Public C3D modifiation interface
    void lockGroup(const std::string &groupName);
    void unlockGroup(const std::string &groupName);
    void addParameter(const std::string &groupName, const ezc3d::ParametersNS::GroupNS::Parameter &p);
    void addFrame(const ezc3d::DataNS::Frame &f, int j = -1);
    void addPoint(const std::vector<ezc3d::DataNS::Frame> &frames);
    void addPoint(const std::string &name);
    void addAnalog(const std::vector<ezc3d::DataNS::Frame> &frames);
    void addAnalog(const std::string &name);

protected:
    std::string _filePath; // Remember the file path

    // Holder of data
    std::shared_ptr<ezc3d::Header> _header;
    std::shared_ptr<ezc3d::ParametersNS::Parameters> _parameters;
    std::shared_ptr<ezc3d::DataNS::Data> _data;

    // Internal reading function
    char * c_float;
    unsigned int m_nByteToRead_float;
    void readFile(unsigned int nByteToRead,
        char * c,
        int nByteFromPrevious = 0,
        const  std::ios_base::seekdir &pos = std::ios::cur);

    // Internal function for reading strings
    size_t _dispatchMatrix(const std::vector<int> &dimension,
                         const std::vector<std::string> &param_data_in,
                         std::vector<std::string> &param_data_out,
                         size_t idxInParam = 0,
                         size_t currentIdx = 1);
    void _readMatrix(const std::vector<int> &dimension,
                     std::vector<std::string> &param_data,
                     size_t currentIdx = 0);

    // Converting functions
    unsigned int hex2uint(const char * val, unsigned int len);
    int hex2int(const char * val, unsigned int len);
};
#include "Header.h"
#include "Data.h"
#include "Parameters.h"

#endif

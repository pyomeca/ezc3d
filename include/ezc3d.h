#ifndef EZC3D_H
#define EZC3D_H
///
/// \file ezc3d.h
/// \brief Declaration of ezc3d class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

// dllexport/import declaration
#ifdef _WIN32
#  ifdef EZC3D_API_EXPORTS
#    define EZC3D_API __declspec(dllexport)
#  else
#    define EZC3D_API __declspec(dllimport)
#  endif
#else
#  define EZC3D_API
#endif

// Includes for standard library
#include <sstream>
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

///
/// \brief Namespace ezc3d
///
/// Usefull functions, enum and misc useful for the ezc3d project
///
namespace ezc3d {
    ///
    /// \brief Enum that describes the size of different types
    ///
    enum DATA_TYPE{
        CHAR = -1,
        BYTE = 1,
        INT = 2,
        WORD = 2,
        FLOAT = 4,
        NONE = 10000
    };

    ///
    /// \brief Remove the spaces at the end of a string
    /// \param str The string to remove the trailing spaces from.
    ///
    /// The function receive a string and modify it by remove the trailing spaces
    ///
    EZC3D_API void removeTrailingSpaces(std::string& str);

    ///
    /// \brief Swap all characters of a string to capital letters
    /// \param str The string to capitalize
    ///
    EZC3D_API std::string toUpper(const std::string &str);

    // Forward declaration of the whole project structure
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
protected:
    std::string _filePath; ///< The file path if the C3D was opened from a file

    // ---- CONSTRUCTORS ---- //
public:
    ///
    /// \brief Create a valid minimalistic C3D structure
    ///
    c3d();

    ///
    /// \brief Read and store a C3D
    /// \param filePath The file path of the C3D file
    ///
    c3d(const std::string &filePath);

    ///
    /// \brief Destroy the class properly
    ///
    virtual ~c3d();


    //---- STREAM ----//
public:
    ///
    /// \brief Print the C3D by calling print method of header, parameter and data
    ///
    void print() const;

    ///
    /// \brief Write the C3D to an opened file by calling write method of header, parameter and data
    /// \param filePath Already opened fstream file with write access
    ///
    void write(const std::string &filePath) const;


    // ---- C3D MAIN STRUCTURE ---- //
protected:
    std::shared_ptr<ezc3d::Header> _header; ///< Pointer that holds the header of the C3D
    std::shared_ptr<ezc3d::ParametersNS::Parameters> _parameters; ///< Pointer that holds the parameters of the C3D
    std::shared_ptr<ezc3d::DataNS::Data> _data; ///< Pointer that holds the data of the C3D

public:
    const ezc3d::Header& header() const; ///< The header of the C3D
    const ezc3d::ParametersNS::Parameters& parameters() const; ///< The parameters of the C3D
    const ezc3d::DataNS::Data& data() const; ///< The points and analogous data of the C3D


    // ---- PUBLIC C3D MODIFICATION INTERFACE ---- //
public:
    ///
    /// \brief Add/replace a parameter to a group named groupName
    /// \param groupName The name of the group to add the parameter to
    /// \param parameter The parameter to add
    ///
    /// Add a parameter to a group. If the the group does not exist in the C3D, it is created. If the
    /// parameter already exists in the group, it is replaced.
    ///
    /// Throw a std::invalid_argument if the name of the parameter is not specified
    ///
    void parameter(const std::string &groupName, const ezc3d::ParametersNS::GroupNS::Parameter &parameter);

    ///
    /// \brief Lock a particular group named groupName
    /// \param groupName The name of the group to lock
    ///
    /// Throw a std::invalid_argument exception if the group name does not exist
    ///
    void lockGroup(const std::string &groupName);

    ///
    /// \brief Unlock a particular group named groupName
    /// \param groupName The name of the group to unlock
    ///
    /// Throw a std::invalid_argument exception if the group name does not exist
    ///
    void unlockGroup(const std::string &groupName);

    ///
    /// \brief Add/replace a frame to the data set
    /// \param frame The frame to copy to the data
    /// \param idx The index of the frame in the data set
    ///
    /// Add or replace a frame to the data set.
    ///
    /// If no idx is sent, then the frame is appended to the data set.
    /// If the idx correspond to a pre-existing frame, it replaces it.
    /// If idx is larger than the number of frames, it resize the frames accordingly and add the frame
    /// where it belongs but leaves the other created frames empty.
    ///
    /// Throw a std::runtime_error if the number of points defined in POINT:USED parameter doesn't correspond
    /// to the number of point in the frame.
    ///
    /// Throw a std::invalid_argument if the point names in the frame don't correspond to the name of the
    /// points as defined in POINT:LABELS group
    ///
    /// Throw a std::runtime_error if at least a point was added to the frame but POINT:RATE is equal to 0
    /// and/or if at least an analog data was added to the frame and ANALOG:RATE is equal to 0
    ///
    void frame(const ezc3d::DataNS::Frame &frame, size_t idx = SIZE_MAX);

    ///
    /// \brief Create a point to the data set of name name
    /// \param name The name of the point to create
    ///
    /// If, for some reason, you want to add a new point to a pre-existing data set, you must
    /// declare this point before, otherwise it rejects it because parameter POINT:LABELS doesn't fit.
    /// This function harmonize the parameter structure with the data structure in advance in order to
    /// add the point.
    ///
    /// Throw the same errors as updateParameter as it calls it after the point is created
    ///
    void point(const std::string &name);

    ///
    /// \brief Add a new point to the data set
    /// \param frames The array of frames to add
    ///
    /// Append a new point to the data set.
    ///
    /// Throw a std::invalid_argument if the size of the std::vector of frames is not equal to the number of frames
    /// already present in the data set. Obviously it throws the same error if no point were sent or if the
    /// point was already in the data set.
    ///
    /// Moreover it throws the same errors as updateParameter as it calls it after the point is added
    ///
    void point(const std::vector<ezc3d::DataNS::Frame> &frames);

    ///
    /// \brief Create a channel of analog data to the data set of name name
    /// \param name The name of the channel to create
    ///
    /// If, for some reason, you want to add a new channel to a pre-existing data set, you must
    /// declare this channel before, otherwise it rejects it because parameter ANALOG:LABELS doesn't fit.
    /// This function harmonize the parameter structure with the data structure in advance in order to
    /// add the channel.
    ///
    /// Throw the same errors as updateParameter as it calls it after the channel is created
    ///
    void analog(const std::string &name);

    ///
    /// \brief Add a new channel to the data set
    /// \param frames The array of frames to add
    ///
    /// Append a new channel to the data set.
    ///
    /// Throw a std::invalid_argument if the size of the std::vector of frames/subframes is not equal
    /// to the number of frames/subframes already present in the data set.
    /// Obviously it throws the same error if no channel were sent or if the
    /// channel was already in the data set.
    ///
    /// Moreover it throws the same errors as updateParameter as it calls it after the channel is added
    ///
    void analog(const std::vector<ezc3d::DataNS::Frame> &frames);


    // ---- UPDATER ---- //
protected:
    void updateHeader();
    void updateParameters(const std::vector<std::string> &newMarkers = std::vector<std::string>(), const std::vector<std::string> &newAnalogs = std::vector<std::string>());


public:
    // Byte reading functions
    std::string readString(unsigned int nByteToRead, int nByteFromPrevious = 0,
                           const std::ios_base::seekdir &pos = std::ios::cur);
    int readInt(unsigned int nByteToRead,
                int nByteFromPrevious = 0,
                const std::ios_base::seekdir &pos = std::ios::cur);
    size_t readUint(size_t nByteToRead,
                 int nByteFromPrevious = 0,
                 const std::ios_base::seekdir &pos = std::ios::cur);
    float readFloat(int nByteFromPrevious = 0,
                    const std::ios_base::seekdir &pos = std::ios::cur);
    void readMatrix(const std::vector<size_t> &dimension,
                    std::vector<std::string> &param_data);
    void readMatrix(unsigned int dataLenghtInBytes,
                    const std::vector<size_t> &dimension,
                    std::vector<int> &param_data,
                    size_t currentIdx = 0);
    void readMatrix(const std::vector<size_t> &dimension,
                    std::vector<float> &param_data,
                    size_t currentIdx = 0);





protected:
    // Internal reading function
    char * c_float;
    unsigned int m_nByteToRead_float;
    void readFile(unsigned int nByteToRead,
        char * c,
        int nByteFromPrevious = 0,
        const  std::ios_base::seekdir &pos = std::ios::cur);

    // Internal function for reading strings
    size_t _dispatchMatrix(const std::vector<size_t> &dimension,
                         const std::vector<std::string> &param_data_in,
                         std::vector<std::string> &param_data_out,
                         size_t idxInParam = 0,
                         size_t currentIdx = 1);
    void _readMatrix(const std::vector<size_t> &dimension,
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

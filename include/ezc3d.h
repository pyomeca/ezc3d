#ifndef EZC3D_H
#define EZC3D_H
///
/// \file ezc3d.h
/// \brief Declaration of ezc3d class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

///
/// \mainpage Documentation of ezc3d
///
/// \section intro_sec Introduction
///
/// This is the document for the library ezc3d
/// (<a href="http://github.com/pyomeca/ezc3d">http://github.com/pyomeca/ezc3d</a>). The main goal of
/// this library is to eazily create, read and modify c3d (<a href="http://c3d.org">http://c3d.org</a>)
/// files, largely used in biomechanics.
///
/// This documentation was automatically generated for the "Nostalgia" Release 1.1.0 on the 9th of August, 2019.
///
/// \section install_sec Installation
///
/// To install ezc3d, please refer to the README.md file accessible via the github repository or by following this
/// <a href="md__home_pariterre_programmation_ezc3d_README.html">link</a>.
///
/// \section contact_sec Contact
///
/// If you have any questions, comments or suggestions for future development, you are very welcomed to
/// send me an email at <a href="mailto:pariterre@gmail.com">pariterre@gmail.com</a>.
///
/// \section conclusion_sec Conclusion
///
/// Enjoy C3D files!
///

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

#include "ezc3dConfig.h"

///
/// \brief Namespace ezc3d
///
/// Useful functions, enum and misc useful for the ezc3d project
///
namespace ezc3d {
    // ---- UTILS ---- //
    ///
    /// \brief Enum that describes the size of different types
    ///
    enum DATA_TYPE{
        CHAR = -1,
        BYTE = 1,
        INT = 2,
        WORD = 2,
        FLOAT = 4,
        NO_DATA_TYPE = 10000
    };

    ///
    /// \brief The type of processor used to store the data
    ///
    enum PROCESSOR_TYPE{
        INTEL = 84,
        DEC = 85,
        MIPS = 86,
        NO_PROCESSOR_TYPE = INTEL
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


    // ---- FORWARD DECLARATION OF THE WHOLE PROJECT STRUCTURE ----//
    class c3d;
    class EZC3D_API Matrix;
    class EZC3D_API Matrix33;
    class EZC3D_API Matrix66;
    class EZC3D_API Vector3d;
    class EZC3D_API Vector6d;
    class EZC3D_API Header;

    ///
    /// \brief Namespace that holds the Parameters hierarchy
    ///
    namespace ParametersNS {
        class EZC3D_API Parameters;

        ///
        /// \brief Namespace that holds the Group and Parameter classes
        ///
        namespace GroupNS {
                class EZC3D_API Group;
                class EZC3D_API Parameter;
            }
    }

    ///
    /// \brief Namespace that holds the Data hierarchy
    ///
    namespace DataNS {
        class EZC3D_API Data;

        class Frame;
        ///
        /// \brief Namespace that holds the Points hierarchy
        ///
        namespace Points3dNS {
            class EZC3D_API Points;
            class EZC3D_API Point;
        }
        ///
        /// \brief Namespace that holds the Analogs hierarchy
        ///
        namespace AnalogsNS {
            class EZC3D_API Analogs;
            class EZC3D_API SubFrame;
            class EZC3D_API Channel;
        }
    }

    ///
    /// \brief Namespace for all the analysis modules
    ///
    namespace Modules {
        class EZC3D_API ForcePlatform;
        class EZC3D_API ForcePlatforms;
    }
}

///
/// \brief Main class for C3D holder
///
class EZC3D_API ezc3d::c3d {
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

protected:
    // Internal reading and writting function
    char * c_float; ///< Char to be used by the read function with the specific size of a float preventing to allocate it at each calls
    char * c_float_tp; ///< Char to be used by the read function with the specific size of a float preventing to allocate it at each calls (allow for copy of c_float)
    char * c_int; ///< Char to be used by the read function with the specific size of a int preventing to allocate it at each calls
    char * c_int_tp; ///< Char to be used by the read function with the specific size of a int preventing to allocate it at each calls  (allow for copy of c_int)
    unsigned int m_nByteToRead_float; ///< Declaration of the size of a float
    unsigned int m_nByteToReadMax_int; ///< Declaration of the max size of a int

    ///
    /// \brief Resize the too small char to read
    /// \param nByteToRead The number of bytes to read
    ///
    void resizeCharHolder(unsigned int nByteToRead);

    ///
    /// \brief The function that reads the file, it returns the value into a generic char pointer that must be pre-allocate
    /// \param file opened file stream to be read
    /// \param nByteToRead The number of bytes to read
    /// \param c The output char
    /// \param nByteFromPrevious The number of byte to skip from current position
    /// \param pos The position to start from
    ///
    void readFile(
        std::fstream &file,
        unsigned int nByteToRead,
        char * c,
        int nByteFromPrevious = 0,
        const  std::ios_base::seekdir &pos = std::ios::cur);

    ///
    /// \brief Convert an hexadecimal value to an unsigned integer
    /// \param val The value to convert
    /// \param len The number of bytes of the val parameter
    /// \return The unsigned integer value
    ///
    unsigned int hex2uint(const char * val, unsigned int len);

    ///
    /// \brief Convert an hexadecimal value to a integer
    /// \param val The value to convert
    /// \param len The number of bytes of the val parameter
    /// \return The integer value
    ///
    int hex2int(const char * val, unsigned int len);

    ///
    /// \brief Write the data_start parameter where demanded
    /// \param file opened file stream to be read
    /// \param dataStartPosition The position in block of the data
    /// \param type The type of data to write
    ///
    void writeDataStart(std::fstream &file,
                        const std::streampos& dataStartPosition,
                        const DATA_TYPE &type) const;

public:
    ///
    /// \brief Read an integer of nByteToRead bytes at the position current + nByteFromPrevious from a file
    /// \param processorType Convension processor type the file is following
    /// \param file opened file stream to be read
    /// \param nByteToRead The number of byte to read to be converted into integer
    /// \param nByteFromPrevious The number of bytes to skip from the current cursor position
    /// \param pos Where to reposition the cursor
    /// \return The integer value
    ///
    int readInt(PROCESSOR_TYPE processorType,
                std::fstream &file,
                unsigned int nByteToRead,
                int nByteFromPrevious = 0,
                const std::ios_base::seekdir &pos = std::ios::cur);

    ///
    /// \brief Read a unsigned integer of nByteToRead bytes at the position current + nByteFromPrevious from a file
    /// \param processorType Convension processor type the file is following
    /// \param file opened file stream to be read
    /// \param nByteToRead The number of byte to read to be converted into unsigned integer
    /// \param nByteFromPrevious The number of bytes to skip from the current cursor position
    /// \param pos Where to reposition the cursor
    /// \return The unsigned integer value
    ///
    size_t readUint(PROCESSOR_TYPE processorType,
                    std::fstream &file,
                    unsigned int nByteToRead,
                    int nByteFromPrevious = 0,
                    const std::ios_base::seekdir &pos = std::ios::cur);

    ///
    /// \brief Read a float at the position current + nByteFromPrevious from a file
    /// \param processorType Convension processor type the file is following
    /// \param file opened file stream to be read
    /// \param nByteFromPrevious The number of bytes to skip from the current cursor position
    /// \param pos Where to reposition the cursor
    /// \return The float value
    ///
    float readFloat(PROCESSOR_TYPE processorType,
                    std::fstream &file,
                    int nByteFromPrevious = 0,
                    const std::ios_base::seekdir &pos = std::ios::cur);

    ///
    /// \brief Read a string (array of char of nByteToRead bytes) at the position current + nByteFromPrevious from a file
    /// \param file opened file stream to be read
    /// \param nByteToRead The number of byte to read to be converted into float
    /// \param nByteFromPrevious The number of bytes to skip from the current cursor position
    /// \param pos Where to reposition the cursor
    /// \return The float value
    ///
    std::string readString(std::fstream &file,
                           unsigned int nByteToRead,
                           int nByteFromPrevious = 0,
                           const std::ios_base::seekdir &pos = std::ios::cur);

    ///
    /// \brief Read a matrix of integer parameters of dimensions dimension with each integer of length dataLengthInByte
    /// \param processorType Convension processor type the file is following
    /// \param file opened file stream to be read
    /// \param dataLenghtInBytes The number of bytes to read to be converted to int
    /// \param dimension The dimensions of the matrix up to 7-dimensions
    /// \param param_data The output of the function
    /// \param currentIdx Internal tracker of where the function is in the flow of the recursive calls
    ///
    void readParam(
            PROCESSOR_TYPE processorType,
            std::fstream &file,
            unsigned int dataLenghtInBytes,
            const std::vector<size_t> &dimension,
            std::vector<int> &param_data,
            size_t currentIdx = 0);

    ///
    /// \brief Read a matrix of float parameters of dimensions dimension
    /// \param processorType Convension processor type the file is following
    /// \param file opened file stream to be read
    /// \param dimension The dimensions of the matrix up to 7-dimensions
    /// \param param_data The output of the function
    /// \param currentIdx Internal tracker of where the function is in the flow of the recursive calls
    ///
    void readParam(
            PROCESSOR_TYPE processorType,
            std::fstream &file,
            const std::vector<size_t> &dimension,
            std::vector<double> &param_data,
            size_t currentIdx = 0);

    ///
    /// \brief Read a matrix of string of dimensions dimension with the first dimension being the length of the strings
    /// \param file opened file stream to be read
    /// \param dimension The dimensions of the matrix up to 7-dimensions. The first dimension is the length of the strings
    /// \param param_data The output of the function
    ///
    void readParam(
            std::fstream &file,
            const std::vector<size_t> &dimension,
            std::vector<std::string> &param_data);

protected:
    ///
    /// \brief Internal function to dispatch a string array to a matrix of strings
    /// \param dimension The dimensions of the matrix up to 7-dimensions
    /// \param param_data_in The input vector of strings
    /// \param param_data_out The output matrix of strings
    /// \param idxInParam Internal counter to keep track where the function is in its recursive calls
    /// \param currentIdx Internal counter to keep track where the function is in its recursive calls
    /// \return
    ///
    size_t _dispatchMatrix(
            const std::vector<size_t> &dimension,
            const std::vector<std::string> &param_data_in,
            std::vector<std::string> &param_data_out,
            size_t idxInParam = 0,
            size_t currentIdx = 1);

    ///
    /// \brief Internal function to read a string array to a matrix of strings
    /// \param file opened file stream to be read
    /// \param dimension The dimensions of the matrix up to 7-dimensions
    /// \param param_data The output matrix of strings
    /// \param currentIdx Internal counter to keep track where the function is in its recursive calls
    ///
    void _readMatrix(
            std::fstream &file,
            const std::vector<size_t> &dimension,
            std::vector<std::string> &param_data,
            size_t currentIdx = 0);


    // ---- C3D MAIN STRUCTURE ---- //
protected:
    std::shared_ptr<ezc3d::Header> _header; ///< Pointer that holds the header of the C3D
    std::shared_ptr<ezc3d::ParametersNS::Parameters> _parameters; ///< Pointer that holds the parameters of the C3D
    std::shared_ptr<ezc3d::DataNS::Data> _data; ///< Pointer that holds the data of the C3D

public:
    ///
    /// \brief The header of the C3D
    /// \return The header of the C3D
    ///
    const ezc3d::Header& header() const;

    ///
    /// \brief The parameters of the C3D
    /// \return The parameters of the C3D
    ///
    const ezc3d::ParametersNS::Parameters& parameters() const;

    ///
    /// \brief The points and analogous data of the C3D
    /// \return The points and analogous data of the C3D
    ///
    const ezc3d::DataNS::Data& data() const;

    // ---- PUBLIC GETTER INTERFACE ---- //
public:
    ///
    /// \brief Get a reference to the names of the points
    /// \return The reference to the names of the points
    ///
    const std::vector<std::string>& pointNames() const;

    ///
    /// \brief Get the index of a point in the points holder
    /// \param pointName Name of the point
    /// \return The index of the point
    ///
    /// Search for the index of a point into points data by the name of this point.
    ///
    /// Throw a std::invalid_argument if pointName is not found
    ///
    size_t pointIdx(
            const std::string& pointName) const;

    ///
    /// \brief Get a reference to the names of the analog channels
    /// \return The reference to the names of the analog channels
    ///
    const std::vector<std::string>& channelNames() const;

    ///
    /// \brief Get the index of a analog channel in the subframe
    /// \param channelName Name of the analog channel
    /// \return The index of the analog channel
    ///
    /// Search for the index of a analog channel into subframe by the name of this channel.
    ///
    /// Throw a std::invalid_argument if channelName is not found
    ///
    size_t channelIdx(
            const std::string& channelName) const;


    // ---- PUBLIC C3D MODIFICATION INTERFACE ---- //
public:

    ///
    /// \brief setFirstFrame Set the time stamp of the first frame in the header
    /// \param firstFrame The first frame time stamp
    ///
    void setFirstFrame(
            size_t firstFrame);

    ///
    /// \brief setGroupMetadata Set the metadata of a specific group. If group
    /// doesn't exist, it is created
    /// \param groupName The name of the group to set the metadata
    /// \param description The description of the group
    /// \param isLocked If the group is locked
    ///
    void setGroupMetadata(
            const std::string& groupName,
            const std::string& description,
            bool isLocked);

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
    void parameter(
            const std::string &groupName,
            const ezc3d::ParametersNS::GroupNS::Parameter &parameter);

    ///
    /// \brief Lock a particular group named groupName
    /// \param groupName The name of the group to lock
    ///
    /// Throw a std::invalid_argument exception if the group name does not exist
    ///
    void lockGroup(
            const std::string &groupName);

    ///
    /// \brief Unlock a particular group named groupName
    /// \param groupName The name of the group to unlock
    ///
    /// Throw a std::invalid_argument exception if the group name does not exist
    ///
    void unlockGroup(
            const std::string &groupName);

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
    void frame(
            const ezc3d::DataNS::Frame &frame,
            size_t idx = SIZE_MAX);

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
    void point(
            const std::string &name);

    ///
    /// \brief Add a new point to the data set
    /// \param pointName The name of the new point
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
    void point(
            const std::string &pointName,
            const std::vector<ezc3d::DataNS::Frame> &frames);

    ///
    /// \brief Add a new point to the data set
    /// \param pointNames The name vector of the new points
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
    void point(
            const std::vector<std::string> &pointNames,
            const std::vector<ezc3d::DataNS::Frame> &frames);

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
    void analog(
            const std::string &name);

    ///
    /// \brief Add a new channel to the data set
    /// \param channelName Name of the channel to add
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
    void analog(
            std::string channelName,
            const std::vector<ezc3d::DataNS::Frame> &frames);

    ///
    /// \brief Add a new channel to the data set
    /// \param channelNames Name of the channels to add
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
    void analog(
            const std::vector<std::string>& channelNames,
            const std::vector<ezc3d::DataNS::Frame> &frames);


    // ---- UPDATER ---- //
protected:
    ///
    /// \brief Update the header according to the parameters and the data
    ///
    void updateHeader();

    ///
    /// \brief Update parameters according to the data
    /// \param newPoints The names of the new poits
    /// \param newAnalogs The names of the new analogs
    ///
    /// Throw a std::runtime_error if newPoints or newAnalogs was added while the data set is not empty.
    /// If you want to add a new point after having data in the data set, you must use the frame method.
    ///
    void updateParameters(
            const std::vector<std::string> &newPoints = std::vector<std::string>(),
            const std::vector<std::string> &newAnalogs = std::vector<std::string>());

};

#endif


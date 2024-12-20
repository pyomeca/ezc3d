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
/// (<a
/// href="http://github.com/pyomeca/ezc3d">http://github.com/pyomeca/ezc3d</a>).
/// The main goal of this library is to eazily create, read and modify c3d (<a
/// href="http://c3d.org">http://c3d.org</a>) files, largely used in
/// biomechanics.
///
/// This documentation was automatically generated for the "Nostalgia"
/// Release 1.1.0 on the 9th of August, 2019.
///
/// \section install_sec Installation
///
/// To install ezc3d, please refer to the README.md file accessible via the
/// github repository or by following this <a
/// href="md__home_pariterre_programmation_ezc3d_README.html">link</a>.
///
/// \section contact_sec Contact
///
/// If you have any questions, comments or suggestions for future development,
/// you are very welcomed to send me an email at <a
/// href="mailto:pariterre@gmail.com">pariterre@gmail.com</a>.
///
/// \section conclusion_sec Conclusion
///
/// Enjoy C3D files!
///

// Includes for standard library
#include <fstream>
#include <vector>

#include "ezc3d/ezc3dConfig.h"
#include "ezc3d/ezc3dNamespace.h"

///
/// \brief Main class for C3D holder
///
/// Please note that a copy of a C3D class is a shallow copy, thanks to the
/// use of shared_ptr
///
class EZC3D_VISIBILITY ezc3d::c3d {
protected:
  std::string _filePath; ///< The file path if the C3D was opened from a file

  // ---- CONSTRUCTORS ---- //
public:
  ///
  /// \brief Create a valid minimalistic C3D structure
  ///
  EZC3D_API c3d();

  ///
  /// \brief Read and store a C3D
  /// \param filePath The file path of the C3D file
  /// \param ignoreBadFormatting If bad formatting of the c3d should be ignored,
  /// use with caution as it can results in a segmentation fault
  ///
  EZC3D_API c3d(const std::string &filePath, bool ignoreBadFormatting = false);

  //---- STREAM ----//
public:
  ///
  /// \brief Print the C3D by calling print method of header, parameter and data
  ///
  EZC3D_API void print() const;

  ///
  /// \brief Write the C3D to an opened file by calling write method of header,
  /// parameter and data \param filePath Already opened fstream file with write
  /// access \param format What order should the file has
  ///
  EZC3D_API void
  write(const std::string &filePath,
        const WRITE_FORMAT &format = WRITE_FORMAT::DEFAULT) const;

  ///
  /// \brief Write the C3D to an opened file by calling write method of header,
  /// parameter and data. The default parametrization will produce a valid and
  /// standard c3d. However, changing these value will definitely produce a
  /// non-standard c3d which may or may not work on another software.
  ///
  /// \param filePath Already opened fstream file with write access
  /// \param format What order should the file has
  /// \param forceZeroBasedOnFrameCount According to the standard, the first and
  /// last frame are stored as a one-based value. But some software requires it
  /// to be zero. Leave the user the capability to do so.
  ///
  EZC3D_API void
  parametrizedWrite(const std::string &filePath,
                    const WRITE_FORMAT &format = WRITE_FORMAT::DEFAULT,
                    bool forceZeroBasedOnFrameCount = false) const;

protected:
  // Internal reading and writting function
  std::vector<char>
      c_float; ///< Char to be used by the read function with the specific size
               ///< of a float preventing to allocate it at each calls
  std::vector<char>
      c_float_tp; ///< Char to be used by the read function with the specific
                  ///< size of a float preventing to allocate it at each calls
                  ///< (allow for copy of c_float)
  std::vector<char>
      c_int; ///< Char to be used by the read function with the specific size of
             ///< a int preventing to allocate it at each calls
  std::vector<char> c_int_tp; ///< Char to be used by the read function with the
                              ///< specific size of a int preventing to allocate
                              ///< it at each calls  (allow for copy of c_int)
  unsigned int m_nByteToRead_float;  ///< Declaration of the size of a float
  unsigned int m_nByteToReadMax_int; ///< Declaration of the max size of a int

  ///
  /// \brief Resize the too small char to read
  /// \param nByteToRead The number of bytes to read
  ///
  void resizeCharHolder(unsigned int nByteToRead);

  ///
  /// \brief The function that reads the file, it returns the value into a
  /// generic char pointer that must be pre-allocate \param file opened file
  /// stream to be read \param nByteToRead The number of bytes to read \param c
  /// The output char \param nByteFromPrevious The number of byte to skip from
  /// current position \param pos The position to start from
  ///
  void readFile(std::fstream &file, unsigned int nByteToRead,
                std::vector<char> &c, int nByteFromPrevious = 0,
                const std::ios_base::seekdir &pos = std::ios::cur);

  ///
  /// \brief Convert an hexadecimal value to an unsigned integer
  /// \param val The value to convert
  /// \param len The number of bytes of the val parameter
  /// \return The unsigned integer value
  ///
  unsigned int hex2uint(const std::vector<char> &val, unsigned int len);

  ///
  /// \brief Convert an hexadecimal value to a integer
  /// \param val The value to convert
  /// \param len The number of bytes of the val parameter
  /// \return The integer value
  ///
  int hex2int(const std::vector<char> &val, unsigned int len);

  ///
  /// \brief Write the data_start parameter where demanded
  /// \param file opened file stream to be read
  /// \param dataStartPosition The position in block of the data
  ///
  void writeDataStart(std::fstream &file,
                      const ezc3d::DataStartInfo &dataStartPosition) const;

public:
  ///
  /// \brief Read an integer of nByteToRead bytes at the position current +
  /// nByteFromPrevious from a file \param processorType Convension processor
  /// type the file is following \param file opened file stream to be read
  /// \param nByteToRead The number of byte to read to be converted into integer
  /// \param nByteFromPrevious The number of bytes to skip from the current
  /// cursor position \param pos Where to reposition the cursor \return The
  /// integer value
  ///
  EZC3D_API int readInt(PROCESSOR_TYPE processorType, std::fstream &file,
                        unsigned int nByteToRead, int nByteFromPrevious = 0,
                        const std::ios_base::seekdir &pos = std::ios::cur);

  ///
  /// \brief Read a unsigned integer of nByteToRead bytes at the position
  /// current + nByteFromPrevious from a file \param processorType Convension
  /// processor type the file is following \param file opened file stream to be
  /// read \param nByteToRead The number of byte to read to be converted into
  /// unsigned integer \param nByteFromPrevious The number of bytes to skip from
  /// the current cursor position \param pos Where to reposition the cursor
  /// \return The unsigned integer value
  ///
  EZC3D_API size_t readUint(PROCESSOR_TYPE processorType, std::fstream &file,
                            unsigned int nByteToRead, int nByteFromPrevious = 0,
                            const std::ios_base::seekdir &pos = std::ios::cur);

  ///
  /// \brief Read a float at the position current + nByteFromPrevious from a
  /// file \param processorType Convension processor type the file is following
  /// \param file opened file stream to be read
  /// \param nByteFromPrevious The number of bytes to skip from the current
  /// cursor position \param pos Where to reposition the cursor \return The
  /// float value
  ///
  EZC3D_API float readFloat(PROCESSOR_TYPE processorType, std::fstream &file,
                            int nByteFromPrevious = 0,
                            const std::ios_base::seekdir &pos = std::ios::cur);

  ///
  /// \brief Read a string (array of char of nByteToRead bytes) at the position
  /// current + nByteFromPrevious from a file \param file opened file stream to
  /// be read \param nByteToRead The number of byte to read to be converted into
  /// float \param nByteFromPrevious The number of bytes to skip from the
  /// current cursor position \param pos Where to reposition the cursor \return
  /// The float value
  ///
  EZC3D_API std::string
  readString(std::fstream &file, unsigned int nByteToRead,
             int nByteFromPrevious = 0,
             const std::ios_base::seekdir &pos = std::ios::cur);

  ///
  /// \brief Read a matrix of integer parameters of dimensions dimension with
  /// each integer of length dataLengthInByte \param processorType Convension
  /// processor type the file is following \param file opened file stream to be
  /// read \param dataLenghtInBytes The number of bytes to read to be converted
  /// to int \param dimension The dimensions of the matrix up to 7-dimensions
  /// \param param_data The output of the function
  /// \param currentIdx Internal tracker of where the function is in the flow of
  /// the recursive calls
  ///
  EZC3D_API void readParam(PROCESSOR_TYPE processorType, std::fstream &file,
                           unsigned int dataLenghtInBytes,
                           const std::vector<size_t> &dimension,
                           std::vector<int> &param_data, size_t currentIdx = 0);

  ///
  /// \brief Read a matrix of float parameters of dimensions dimension
  /// \param processorType Convension processor type the file is following
  /// \param file opened file stream to be read
  /// \param dimension The dimensions of the matrix up to 7-dimensions
  /// \param param_data The output of the function
  /// \param currentIdx Internal tracker of where the function is in the flow of
  /// the recursive calls
  ///
  EZC3D_API void readParam(PROCESSOR_TYPE processorType, std::fstream &file,
                           const std::vector<size_t> &dimension,
                           std::vector<double> &param_data,
                           size_t currentIdx = 0);

  ///
  /// \brief Read a matrix of string of dimensions dimension with the first
  /// dimension being the length of the strings \param file opened file stream
  /// to be read \param dimension The dimensions of the matrix up to
  /// 7-dimensions. The first dimension is the length of the strings \param
  /// param_data The output of the function
  ///
  EZC3D_API void readParam(std::fstream &file,
                           const std::vector<size_t> &dimension,
                           std::vector<std::string> &param_data);

  ///
  /// \brief Advance the cursor in a file to a new 512 bytes block
  /// \param file The file stream
  ///
  EZC3D_API static void moveCursorToANewBlock(std::fstream &file);

protected:
  ///
  /// \brief Internal function to dispatch a string array to a matrix of strings
  /// \param dimension The dimensions of the matrix up to 7-dimensions
  /// \param param_data_in The input vector of strings
  /// \param param_data_out The output matrix of strings
  /// \param idxInParam Internal counter to keep track where the function is in
  /// its recursive calls \param currentIdx Internal counter to keep track where
  /// the function is in its recursive calls \return
  ///
  size_t _dispatchMatrix(const std::vector<size_t> &dimension,
                         const std::vector<std::string> &param_data_in,
                         std::vector<std::string> &param_data_out,
                         size_t idxInParam = 0, size_t currentIdx = 1);

  ///
  /// \brief Internal function to read a string array to a matrix of strings
  /// \param file opened file stream to be read
  /// \param dimension The dimensions of the matrix up to 7-dimensions
  /// \param param_data The output matrix of strings
  /// \param currentIdx Internal counter to keep track where the function is in
  /// its recursive calls
  ///
  void _readMatrix(std::fstream &file, const std::vector<size_t> &dimension,
                   std::vector<std::string> &param_data, size_t currentIdx = 0);

  // ---- C3D MAIN STRUCTURE ---- //
protected:
  std::shared_ptr<ezc3d::Header>
      _header; ///< Pointer that holds the header of the C3D
  std::shared_ptr<ezc3d::ParametersNS::Parameters>
      _parameters; ///< Pointer that holds the parameters of the C3D
  std::shared_ptr<ezc3d::DataNS::Data>
      _data; ///< Pointer that holds the data of the C3D

public:
  ///
  /// \brief The header of the C3D
  /// \return The header of the C3D
  ///
  EZC3D_API const ezc3d::Header &header() const;

  ///
  /// \brief The parameters of the C3D
  /// \return The parameters of the C3D
  ///
  EZC3D_API const ezc3d::ParametersNS::Parameters &parameters() const;

  ///
  /// \brief The points and analogous data of the C3D
  /// \return The points and analogous data of the C3D
  ///
  EZC3D_API const ezc3d::DataNS::Data &data() const;

  // ---- PUBLIC GETTER INTERFACE ---- //
public:
  ///
  /// \brief Get a copy of the names of the points
  /// \return The reference to the names of the points
  ///
  EZC3D_API const std::vector<std::string> pointNames() const;

  ///
  /// \brief Get a copy of the scales of the points
  /// \return The reference to the scales of the points
  ///
  EZC3D_API const std::vector<double> pointScales() const;

  ///
  /// \brief Get the index of a point in the points holder
  /// \param pointName Name of the point
  /// \return The index of the point
  ///
  /// Search for the index of a point into points data by the name of this
  /// point.
  ///
  /// Throw a std::invalid_argument if pointName is not found
  ///
  EZC3D_API size_t pointIdx(const std::string &pointName) const;

  ///
  /// \brief Get the names of the analog channels
  /// \return The names of the analog channels
  ///
  EZC3D_API const std::vector<std::string> channelNames() const;

  ///
  /// \brief Get a copy of the scales of the channels
  /// \return The reference to the scales of the channels
  ///
  EZC3D_API const std::vector<double> channelScales() const;

  ///
  /// \brief Get a copy of the offsets of the channels
  /// \return The reference to the offsets of the channels
  ///
  EZC3D_API const std::vector<int> channelOffsets() const;

  ///
  /// \brief Get the index of a analog channel in the subframe
  /// \param channelName Name of the analog channel
  /// \return The index of the analog channel
  ///
  /// Search for the index of a analog channel into subframe by the name of this
  /// channel.
  ///
  /// Throw a std::invalid_argument if channelName is not found
  ///
  EZC3D_API size_t channelIdx(const std::string &channelName) const;

  // ---- PUBLIC C3D MODIFICATION INTERFACE ---- //
public:
  ///
  /// \brief setFirstFrame Set the time stamp of the first frame in the header
  /// \param firstFrame The first frame time stamp
  ///
  EZC3D_API void setFirstFrame(size_t firstFrame);

  ///
  /// \brief setGroupMetadata Set the metadata of a specific group. If group
  /// doesn't exist, it is created
  /// \param groupName The name of the group to set the metadata
  /// \param description The description of the group
  /// \param isLocked If the group is locked
  ///
  EZC3D_API void setGroupMetadata(const std::string &groupName,
                                  const std::string &description,
                                  bool isLocked);

  ///
  /// \brief Add/replace a parameter to a group named groupName
  /// \param groupName The name of the group to add the parameter to
  /// \param parameter The parameter to add
  ///
  /// Add a parameter to a group. If the the group does not exist in the C3D, it
  /// is created. If the parameter already exists in the group, it is replaced.
  ///
  /// Throw a std::invalid_argument if the name of the parameter is not
  /// specified
  ///
  EZC3D_API void
  parameter(const std::string &groupName,
            const ezc3d::ParametersNS::GroupNS::Parameter &parameter);

  ///
  /// \brief Remove a parameter from a group
  /// \param groupName The name of the group to remove the parameter from
  /// \param parameterName The name of the parameter in the group
  ///
  EZC3D_API void remove(const std::string &groupName,
                        const std::string &parameterName);

  ///
  /// \brief Remove a group
  /// \param groupName The name of the group to remove
  ///
  EZC3D_API void remove(const std::string &groupName);

  ///
  /// \brief Lock a particular group named groupName
  /// \param groupName The name of the group to lock
  ///
  /// Throw a std::invalid_argument exception if the group name does not exist
  ///
  EZC3D_API void lockGroup(const std::string &groupName);

  ///
  /// \brief Unlock a particular group named groupName
  /// \param groupName The name of the group to unlock
  ///
  /// Throw a std::invalid_argument exception if the group name does not exist
  ///
  EZC3D_API void unlockGroup(const std::string &groupName);

  ///
  /// \brief Add/replace a frame to the data set
  /// \param frame The frame to copy to the data
  /// \param idx The index of the frame in the data set
  /// \param skipInternalUpdates If the updates of Parameters and Headers should
  /// be skipped
  ///
  /// Add or replace a frame to the data set.
  ///
  /// If no idx is sent, then the frame is appended to the data set.
  /// If the idx correspond to a pre-existing frame, it replaces it.
  /// If idx is larger than the number of frames, it resize the frames
  /// accordingly and add the frame where it belongs but leaves the other
  /// created frames empty.
  ///
  /// If [skipInternalUpdates] is set to true, then no checks or updates are
  /// done to the parameters and the headers. This greatly improves the speed of
  /// creating a new file, but conversely it can create corrupted C3D.
  ///
  /// Throw a std::runtime_error if the number of points defined in POINT:USED
  /// parameter doesn't correspond to the number of point in the frame.
  ///
  /// Throw a std::invalid_argument if the point names in the frame don't
  /// correspond to the name of the points as defined in POINT:LABELS group
  ///
  /// Throw a std::runtime_error if at least a point was added to the frame but
  /// POINT:RATE is equal to 0 and/or if at least an analog data was added to
  /// the frame and ANALOG:RATE is equal to 0
  ///
  EZC3D_API void frame(const ezc3d::DataNS::Frame &frame, size_t idx = SIZE_MAX,
                       bool skipInternalUpdates = false);

  ///
  /// \brief Add/replace a batch of frames, calling the frame() method
  /// repeateadly but skipping the updateParameter \param frames All the frame
  /// to add \param firstFrameIdx The index of the frame of frames[0]. The
  /// others are assumed to be increments of one
  ///
  /// See frame() for the description of the function with [skipInternalUpdates]
  /// set to true for all but the first and last frames. If no [firstFrameIdx]
  /// is sent, then all the frames are appended to the existing values.
  ///
  /// WARNING: since no checks are performed on the frames (apart from the first
  /// and the last), all the frames must have the same number of subframes,
  /// points and analogs. Failing to do so will not throw an error, but will
  /// create a corrupted C3D file.
  ///
  EZC3D_API void frames(const std::vector<ezc3d::DataNS::Frame> frames,
                        size_t firstFrameidx = SIZE_MAX);

  ///
  /// \brief Create a point to the data set of name pointName
  /// \param pointName The name of the point to create
  ///
  /// If, for some reason, you want to add a new point to a pre-existing data
  /// set, you must declare this point before, otherwise it rejects it because
  /// parameter POINT:LABELS doesn't fit. This function harmonize the parameter
  /// structure with the data structure in advance in order to add the point.
  ///
  /// Throw the same errors as updateParameter as it calls it after the point is
  /// created
  ///
  EZC3D_API void point(const std::string &pointName);

  ///
  /// \brief Add a new point to the data set
  /// \param pointName The name of the new point
  /// \param frames The array of frames to add
  ///
  /// Append a new point to the data set.
  ///
  /// Throw a std::invalid_argument if the size of the std::vector of frames is
  /// not equal to the number of frames already present in the data set.
  /// Obviously it throws the same error if no point were sent or if the point
  /// was already in the data set.
  ///
  /// Moreover it throws the same errors as updateParameter as it calls it after
  /// the point is added
  ///
  EZC3D_API void point(const std::string &pointName,
                       const std::vector<ezc3d::DataNS::Frame> &frames);

  ///
  /// \brief Create points to the data set of name pointNames
  /// \param pointNames The name of the points to create
  ///
  /// If, for some reason, you want to add a new point to a pre-existing data
  /// set, you must declare this point before, otherwise it rejects it because
  /// parameter POINT:LABELS doesn't fit. This function harmonize the parameter
  /// structure with the data structure in advance in order to add the point.
  ///
  /// Throw the same errors as updateParameter as it calls it after the point is
  /// created
  ///
  EZC3D_API void point(const std::vector<std::string> &pointNames);

  ///
  /// \brief Add a new point to the data set
  /// \param pointNames The name vector of the new points
  /// \param frames The array of frames to add
  ///
  /// Append a new point to the data set.
  ///
  /// Throw a std::invalid_argument if the size of the std::vector of frames is
  /// not equal to the number of frames already present in the data set.
  /// Obviously it throws the same error if no point were sent or if the point
  /// was already in the data set.
  ///
  /// Moreover it throws the same errors as updateParameter as it calls it after
  /// the point is added
  ///
  EZC3D_API void point(const std::vector<std::string> &pointNames,
                       const std::vector<ezc3d::DataNS::Frame> &frames);

  ///
  /// \brief Create a channel of analog data to the data set of name channelName
  /// \param channelName The name of the channel to create
  ///
  /// If, for some reason, you want to add a new channel to a pre-existing data
  /// set, you must declare this channel before, otherwise it rejects it because
  /// parameter ANALOG:LABELS doesn't fit. This function harmonize the parameter
  /// structure with the data structure in advance in order to add the channel.
  ///
  /// Throw the same errors as updateParameter as it calls it after the channel
  /// is created
  ///
  EZC3D_API void analog(const std::string &channelName);

  ///
  /// \brief Add a new channel to the data set
  /// \param channelName Name of the channel to add
  /// \param frames The array of frames to add
  ///
  /// Append a new channel to the data set.
  ///
  /// Throw a std::invalid_argument if the size of the std::vector of
  /// frames/subframes is not equal to the number of frames/subframes already
  /// present in the data set. Obviously it throws the same error if no channel
  /// were sent or if the channel was already in the data set.
  ///
  /// Moreover it throws the same errors as updateParameter as it calls it after
  /// the channel is added
  ///
  EZC3D_API void analog(std::string channelName,
                        const std::vector<ezc3d::DataNS::Frame> &frames);

  ///
  /// \brief Create channels of analog data to the data set of name channelNames
  /// \param channelNames The name of the channel to create
  ///
  /// If, for some reason, you want to add a new channel to a pre-existing data
  /// set, you must declare this channel before, otherwise it rejects it because
  /// parameter ANALOG:LABELS doesn't fit. This function harmonize the parameter
  /// structure with the data structure in advance in order to add the channel.
  ///
  /// Throw the same errors as updateParameter as it calls it after the channel
  /// is created
  ///
  EZC3D_API void analog(const std::vector<std::string> &channelName);

  ///
  /// \brief Add a new channel to the data set
  /// \param channelNames Name of the channels to add
  /// \param frames The array of frames to add
  ///
  /// Append a new channel to the data set.
  ///
  /// Throw a std::invalid_argument if the size of the std::vector of
  /// frames/subframes is not equal to the number of frames/subframes already
  /// present in the data set. Obviously it throws the same error if no channel
  /// were sent or if the channel was already in the data set.
  ///
  /// Moreover it throws the same errors as updateParameter as it calls it after
  /// the channel is added
  ///
  EZC3D_API void analog(const std::vector<std::string> &channelNames,
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
  /// Throw a std::runtime_error if newPoints or newAnalogs was added while the
  /// data set is not empty. If you want to add a new point after having data in
  /// the data set, you must use the frame method.
  ///
  void updateParameters(
      const std::vector<std::string> &newPoints = std::vector<std::string>(),
      const std::vector<std::string> &newAnalogs = std::vector<std::string>());
};

#endif

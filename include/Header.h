#ifndef HEADER_H
#define HEADER_H
///
/// \file Header.h
/// \brief Declaration of Header class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "ezc3d.h"

///
/// \brief Header of a C3D file
///
class EZC3D_API ezc3d::Header{
    //---- CONSTRUCTORS ----//
public:
    ///
    /// \brief Create a valid header with minimal informations
    ///
    Header();

    ///
    /// \brief Read and store the header of an opened C3D file
    /// \param c3d C3D reference to copy the data in
    /// \param file Already opened fstream file with read access
    ///
    Header(
            c3d &c3d,
            std::fstream &file);


    //---- STREAM ----//
public:
    ///
    /// \brief Print the header
    ///
    void print() const;

    ///
    /// \brief Write the header to an opened file
    /// \param f Already opened fstream file with write access
    /// \param dataStartPosition Returns the byte where to put the data start parameter
    ///
    void write(
            std::fstream &f,
            std::streampos &dataStartPosition) const;

    ///
    /// \brief Read and store a header from an opened C3D file
    /// \param c3d C3D reference to copy the data in
    /// \param file The file stream already opened with read access
    ///
    void read(
            c3d &c3d,
            std::fstream &file);


    //---- HEADER ----//
protected:
    size_t _nbOfZerosBeforeHeader; ///< If the header doesn't start at the begining of the file
                                   ///< remember the zero counts

    size_t _parametersAddress;  ///< Byte 1.1
                                ///<
                                ///< The byte at which the parameters start in the file

public:
    ///
    /// \brief Get the number of zeros before the header starts in the file
    /// \return The number of zeros before the header starts in the file
    ///
    size_t nbOfZerosBeforeHeader() const;

    ///
    /// \brief Get the byte at which the parameters start in the file
    /// \return The byte at which the parameters start in the file
    ///
    size_t parametersAddress() const;

protected:
    ///
    /// \brief Reads the processor type in the parameter section, returns the file pointer where it was at the beggining of the function
    /// \param c3d C3D reference to copy the data in
    /// \param file opened file stream to be read
    /// \return The processor type as specified in the c3d file (83-Intel, 84-DEC, 85-MIPS)
    ///
    PROCESSOR_TYPE readProcessorType(
            c3d &c3d,
            std::fstream &file);

protected:
    size_t _checksum;   ///< Byte 1.2
                        ///<
                        ///< The checksum should be equals to 0x50 for a valid a c3d

public:
    ///
    /// \brief Get the checksum of the header
    /// \return
    ///
    /// The checksum of the header should be equals to 0x50 for a valid a c3d
    ///
    size_t checksum() const;

protected:
    size_t _nb3dPoints; ///< Byte 2
                        ///<
                        ///< The number of stored 3D points

public:
    ///
    /// \brief Get the number 3D points
    /// \return The number of stored 3D points
    ///
    size_t nb3dPoints() const;

    ///
    /// \brief Set the number 3D points
    /// \param numberOfPoints The number of points
    ///
    void nb3dPoints(size_t numberOfPoints);

protected:
    size_t _nbAnalogsMeasurement;   ///< Byte 3
                                    ///<
                                    ///< The total number of analogous data per point frame

public:
    ///
    /// \brief Get the number of analogs
    /// \return The number of analogs
    ///
    size_t nbAnalogs() const;

    ///
    /// \brief Set the number of analogs
    /// \param nbOfAnalogs The number of analogs
    ///
    void nbAnalogs(
            size_t nbOfAnalogs);

    ///
    /// \brief Get the number of recorded analogs
    /// \return The number of recorded analogs
    ///
    size_t nbAnalogsMeasurement() const;

protected:
    size_t _firstFrame; ///< Byte 4
                        ///<
                        ///< The first frame in the file.
                        ///< Note, contrary to the way it is stored in the file,
                        ///< it is 0-based, i.e. the first frame is 0.
    size_t _lastFrame;  ///< Byte 5
                        ///<
                        ///< The last frame in the file.
                        ///< Note, contrary to the way it is stored in the file,
                        ///< it is 0-based, i.e. the first frame is 0.

public:
    ///
    /// \brief Get the number of frames
    /// \return The number of frames
    ///
    size_t nbFrames() const;

    ///
    /// \brief Get the first frame
    /// \return The first frame
    ///
    size_t firstFrame() const;

    ///
    /// \brief Set the first frame
    /// \param frame
    ///
    void firstFrame(
            size_t frame);

    ///
    /// \brief Get the last frame
    /// \return The last frame
    ///
    size_t lastFrame() const;

    ///
    /// \brief Set the last frame
    /// \param frame
    ///
    void lastFrame(
            size_t frame);

protected:
    size_t _nbMaxInterpGap; ///< Byte 6
                            ///<
                            ///< The maximal gap used for interpolation

public:
    ///
    /// \brief Get the maximal gap used for interpolation
    /// \return The maximal gap used for interpolation
    ///
    size_t nbMaxInterpGap() const;

protected:
    float _scaleFactor;   ///< Byte 7-8
                          ///<
                          ///< The scaling factor to convert the 3D point.
                          ///< If the points are floats, then the scaling factor is if negative

public:
    ///
    /// \brief Get the scaling factor to convert the 3D point
    /// \return The scaling factor to convert the 3D point
    ///
    /// If the points are floats, then the scaling factor is if negative. Otherwise it is a integer
    ///
    float scaleFactor() const;

protected:
    size_t _dataStart; ///< Byte 9
                    ///<
                    ///< The number of 256-byte blocks to get to the points and analogous data in the file

public:
    ///
    /// \brief Get the number of 256-byte blocks to get to the points and analogous data in the file
    /// \return The number of 256-byte blocks to get to the points and analogous data in the file
    ///
    size_t dataStart() const;

protected:
    size_t _nbAnalogByFrame;    ///< Byte 10
                                ///<
                                ///< The number of analog by frame
public:
    ///
    /// \brief Get the number of analog by frame
    /// \return The number of analog by frame
    ///
    size_t nbAnalogByFrame() const;

    ///
    /// \brief Set the number of analog by frame
    /// \param nbOfAnalogsByFrame The number of analog by frame
    ///
    void nbAnalogByFrame(
            size_t nbOfAnalogsByFrame);

protected:
    float _frameRate;   ///< Byte 11-12
                        ///<
                        ///< The points frame rate in Hz

public:
    ///
    /// \brief Get the points frame rate in Hz
    /// \return The points frame rate in Hz
    ///
    float frameRate() const;

    ///
    /// \brief Set the points frame rate in Hz
    /// \param pointFrameRate The points frame rate in Hz
    ///
    void frameRate(
            float pointFrameRate);

protected:
    int _emptyBlock1;   ///< Byte 13-147
                        ///<
                        ///< A fixed section of 0, this is defined by the standard for future use

    int _emptyBlock2;   ///< Byte 152
                        ///<
                        ///< A fixed section of 0, this is defined by the standard for future use

    int _emptyBlock3;   ///< Byte 198
                         ///<
                         ///< A fixed section of 0, this is defined by the standard for future use

    int _emptyBlock4;   ///< Byte 235-256
                        ///<
                        ///< A fixed section of 0, this is defined by the standard for future use

public:
    ///
    /// \brief Get the empty block 1
    /// \return The empty block 1
    ///
    int emptyBlock1() const;

    ///
    /// \brief Get the empty block 2
    /// \return The empty block 2
    ///
    int emptyBlock2() const;

    ///
    /// \brief Get the empty block 3
    /// \return The empty block 3
    ///
    int emptyBlock3() const;

    ///
    /// \brief Get the empty block 4
    /// \return The empty block 4
    ///
    int emptyBlock4() const;

protected:
    size_t _keyLabelPresent;    ///< Byte 148
                                ///<
                                ///< The present label flag. If it is equal to 12345,
                                ///< then label and range are present.

public:
    ///
    /// \brief Get the present label flag
    /// \return The present label flag
    ///
    /// The present label flag.
    /// If it is equal to 12345, then label and range are present.
    ///
    size_t keyLabelPresent() const;

protected:
    size_t _firstBlockKeyLabel;     ///< Byte 149
                                    ///<
                                    ///< The first block of key labels (if present)
    size_t _fourCharPresent;    ///< Byte 150
                                ///<
                                ///< The four characters flag. If it is equal to 12345,
                                ///< then event labels are represented as 4 characters. Otherwise it is 2 characters.

    size_t _nbEvents;   ///< Byte 151
                        ///<
                        ///< The number of defined time events (0 to 18)

    std::vector<float> _eventsTime; ///< Byte 153-188
                                    ///<
                                    ///< The event times in seconds

    std::vector<size_t> _eventsDisplay;     ///< Byte 189-197
                                            ///<
                                            ///< The display flag. If it is 0x00, then it is ON.
                                            ///< If it is 0x01, then it is OFF.

    std::vector<std::string> _eventsLabel;  ///< Byte 199-234
                                            ///<
                                            ///< The event labels (4 characters by label)

public:
    ///
    /// \brief Get the first block of key labels (if present)
    /// \return The first block of key labels (if present)
    ///
    size_t firstBlockKeyLabel() const;

    ///
    /// \brief fourCharPresent
    /// \return
    ///
    size_t fourCharPresent() const;

    ///
    /// \brief Get the number of defined time events (0 to 18)
    /// \return The number of defined time events (0 to 18)
    ///
    size_t nbEvents() const;

    ///
    /// \brief Get the event times in seconds
    /// \return The event times in seconds
    ///
    const std::vector<float>& eventsTime() const;

    ///
    /// \brief Get a particular event time of index idx in seconds
    /// \param idx The index of the event
    /// \return The event time in seconds
    ///
    /// Throw a std::out_of_range exception if idx is larger than the number of events
    ///
    float eventsTime(
            size_t idx) const;

    ///
    /// \brief Get the display flags
    /// \return The display flags
    ///
    /// If it is 0x00, then it is ON.
    /// If it is 0x01, then it is OFF.
    ///
    std::vector<size_t> eventsDisplay() const;

    ///
    /// \brief Get a particular display flag of index idx
    /// \param idx The index of the event
    /// \return The display flag
    ///
    /// If it is 0x00, then it is ON.
    /// If it is 0x01, then it is OFF.
    ///
    /// Throw a std::out_of_range exception if idx is larger than the number of events
    ///
    size_t eventsDisplay(
            size_t idx) const;

    ///
    /// \brief Get the event labels
    /// \return The event labels
    ///
    const std::vector<std::string>& eventsLabel() const;

    ///
    /// \brief Get a particular event label of index idx
    /// \param idx The index of the event
    /// \return The event label in seconds
    ///
    /// Throw a std::out_of_range exception if idx is larger than the number of events
    ///
    const std::string& eventsLabel(
            size_t idx) const;

};

#endif

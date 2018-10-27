#ifndef DATA_H
#define DATA_H
///
/// \file Data.h
/// \brief Declaration of data class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include <sstream>
#include <memory>
#include <ezc3d.h>
#include <Frame.h>

///
/// \brief Actual data of the C3D file
///
/// The class stores all the data frames of a given or create C3D into a STL vector of frame.
///
class EZC3D_API ezc3d::DataNS::Data{
    //---- CONSTRUCTORS ----//
public:
    ///
    /// \brief Create a ready to fill Data class
    ///
    Data();

    ///
    /// \brief Create a filled Data class from a given file
    /// \param file File to copy the data from
    ///
    Data(ezc3d::c3d &file);


    //---- STREAM ----//
public:
    ///
    ///
    /// \brief Print the data
    ///
    /// Print all the data to the console by calling sequentially all the print method for all the frames
    ///
    void print() const;

    ///
    /// \brief Write all the data to an opened file
    /// \param f Already opened fstream file with write access
    ///
    /// Write all the data to a file by calling sequentially all the write method for all the frames
    ///
    void write(std::fstream &f) const;


    //---- FRAME ----//
protected:
    std::vector<ezc3d::DataNS::Frame> _frames; ///< Storage of the data
public:
    ///
    /// \brief Get the number of frames in the data structure
    /// \return The number of frames
    ///
    size_t nbFrames() const;

    ///
    /// \brief Return a frame
    /// \return A const reference to a particular data frame
    ///
    const ezc3d::DataNS::Frame& frame(size_t idx) const;

    ///
    /// \brief Return a frame in order to be modified by the caller
    /// \return A non-const reference to a particular data frame
    ///
    /// This method returns a frame in the form of a non-const reference.
    /// The user can thereafter modify this frame at will, but with the caution it requires.
    ///
    ezc3d::DataNS::Frame& frame_nonConst(size_t idx);

    ///
    /// \brief Add/replace a frame to the data set
    /// \param frame the actual frame
    /// \param idx the index of the frame
    ///
    /// Add or replace a particular frame to the data set.
    ///
    /// If no idx is sent, then the frame is append to the data set.
    /// If the idx correspond to a specific frame, it replaces it.
    /// If idx is outside the data set, it resize the data set accordingly and add the frame where it belongs
    /// but leaves the other created frames empty.
    ///
    void frame(const ezc3d::DataNS::Frame& frame, size_t idx = SIZE_MAX);

};

class EZC3D_API ezc3d::DataNS::Points3dNS::Point{
public:
    void print() const;
    Point();
    Point(const ezc3d::DataNS::Points3dNS::Point&);
    void write(std::fstream &f) const;

    float x() const;
    void x(float x);

    float y() const;
    void y(float y);

    float z() const;
    void z(float z);

	const std::vector<float> data() const;

    float residual() const;
    void residual(float residual);
    const std::string& name() const;
    void name(const std::string &name);

protected:
	std::vector<float> _data;
    std::string _name;
};

class EZC3D_API ezc3d::DataNS::AnalogsNS::Analogs{
public:
    Analogs();
    Analogs(int nSubframes);
    void print() const;
    void write(std::fstream &f) const;

    const std::vector<ezc3d::DataNS::AnalogsNS::SubFrame>& subframes() const;
    std::vector<ezc3d::DataNS::AnalogsNS::SubFrame>& subframes_nonConst();
    const ezc3d::DataNS::AnalogsNS::SubFrame& subframe(int idx) const;
    void addSubframe(const ezc3d::DataNS::AnalogsNS::SubFrame& subframe);
    void replaceSubframe(int idx, const SubFrame& subframe);

protected:
    std::vector<ezc3d::DataNS::AnalogsNS::SubFrame> _subframe;
};

class EZC3D_API ezc3d::DataNS::AnalogsNS::SubFrame{
public:
    SubFrame();
    SubFrame(int nChannels);
    void print() const;
    void write(std::fstream &f) const;

    void addChannel(const ezc3d::DataNS::AnalogsNS::Channel& channel);
    void replaceChannel(int idx, const ezc3d::DataNS::AnalogsNS::Channel& channel);
    void addChannels(const std::vector<ezc3d::DataNS::AnalogsNS::Channel>& allChannelsData);
    std::vector<ezc3d::DataNS::AnalogsNS::Channel>& channels_nonConst();
    const std::vector<ezc3d::DataNS::AnalogsNS::Channel>& channels() const;
    const ezc3d::DataNS::AnalogsNS::Channel& channel(int idx) const;
    const ezc3d::DataNS::AnalogsNS::Channel& channel(std::string channelName) const;
protected:
    std::vector<ezc3d::DataNS::AnalogsNS::Channel> _channels;
};

class EZC3D_API ezc3d::DataNS::AnalogsNS::Channel{
public:
    void print() const;
    void write(std::fstream &f) const;

    float value() const;
    void value(float v);

    const std::string& name() const;
    void name(const std::string &name);

protected:
    std::string _name;
    float _value;
};




#endif

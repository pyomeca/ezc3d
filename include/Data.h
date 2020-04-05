#ifndef DATA_H
#define DATA_H
///
/// \file Data.h
/// \brief Declaration of data class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "Frame.h"

///
/// \brief Data of the C3D file
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
    /// \param c3d Reference to the c3d to copy the data in
    /// \param file File to copy the data from
    ///
    Data(
            ezc3d::c3d &c3d,
            std::fstream &file);


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
    /// \param pointScaleFactor The factor to scale the point data with
    /// \param analogScaleFactors The factors to scale the analog data with
    ///
    /// Write all the data to a file by calling sequentially all the write method for all the frames
    ///
    void write(
            std::fstream &f,
            float pointScaleFactor,
            std::vector<double> analogScaleFactors) const;


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
    /// \brief Get the frame of index idx
    /// \param idx The index of the frame
    /// \return The frame of index idx
    ///
    /// Get the frame of index idx.
    ///
    /// Throw a std::out_of_range exception if idx is larger than the number of frames
    ///
    const ezc3d::DataNS::Frame& frame(
            size_t idx) const;

    ///
    /// \brief Get the frame of index idx in order to be modified by the caller
    /// \param idx The index of the frame
    /// \return A non-const reference to the frame of index idx
    ///
    /// Return a frame in the form of a non-const reference.
    /// The user can thereafter modify this frame at will, but with the caution it requires.
    ///
    /// Throw a std::out_of_range exception if idx is larger than the number of frames
    ///
    ///
    ezc3d::DataNS::Frame& frame(
            size_t idx);

    ///
    /// \brief Add/replace a frame to the data set
    /// \param frame the frame to add/replace
    /// \param idx the index of the frame
    ///
    /// Add or replace a particular frame to the data set.
    ///
    /// If no idx is sent, then the frame is appended to the data set.
    /// If the idx correspond to a specific frame, it replaces it.
    /// If idx is outside the data set, it resize the data set accordingly and add the frame where it belongs
    /// but leaves the other created frames empty.
    ///
    void frame(
            const ezc3d::DataNS::Frame& frame,
            size_t idx = SIZE_MAX);

    ///
    /// \brief Get all the frames from the data set
    /// \return The frames
    ///
    const std::vector<ezc3d::DataNS::Frame>& frames() const;

};

#endif

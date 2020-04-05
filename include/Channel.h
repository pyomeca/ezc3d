#ifndef CHANNEL_H
#define CHANNEL_H
///
/// \file Channel.h
/// \brief Declaration of Channel class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "ezc3d.h"

///
/// \brief Channel of an analogous data
///
class EZC3D_API ezc3d::DataNS::AnalogsNS::Channel{
    //---- CONSTRUCTORS ----//
public:
    ///
    /// \brief Create an empty analogous data channel
    ///
    Channel();

    ///
    /// \brief Copy an analog channel
    /// \param channel The channel to copy
    ///
    Channel(
            const ezc3d::DataNS::AnalogsNS::Channel &channel);


    //---- STREAM ----//
public:
    ///
    ///
    /// \brief Print the channel
    ///
    /// Print the value of the analog data to the console
    ///
    void print() const;

    ///
    /// \brief Write the channel to an opened file
    /// \param f Already opened fstream file with write access
    /// \param scaleFactor The factor to scale the data with
    ///
    /// Write the value of the analog data to a file
    ///
    void write(
            std::fstream &f,
            double scaleFactor) const;


    //---- DATA ----//
protected:
    double _data; ///< Value of the analog data
public:
    ///
    /// \brief Get the value of the analog data
    /// \return The value of the analog data
    ///
    double data() const;

    ///
    /// \brief Set the value of the analog data
    /// \param value The value of the analog data
    ///
    void data(
            double value);

    ///
    /// \brief Return if the channel is empty
    /// \return if the channel is empty
    ///
    bool isEmpty() const;
};


#endif

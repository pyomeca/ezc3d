#ifndef CHANNEL_H
#define CHANNEL_H
///
/// \file Channel.h
/// \brief Declaration of Channel class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include <sstream>
#include <memory>
#include <ezc3d.h>
#include <Subframe.h>

///
/// \brief Channel of an analogous data
///
class EZC3D_API ezc3d::DataNS::AnalogsNS::Channel{
    //---- CONSTRUCTORS ----//
public:
    ///
    /// \brief Create an empty analogous data channel
    /// \param name The name of the channel
    ///
    Channel(const std::string& name = "");

    ///
    /// \brief Copy an analog channel
    /// \param channel The channel to copy
    ///
    Channel(const ezc3d::DataNS::AnalogsNS::Channel &channel);


    //---- STREAM ----//
public:
    ///
    ///
    /// \brief Print the channel
    ///
    /// Print the actual value of the analog data to the console
    ///
    void print() const;

    ///
    /// \brief Write the channel to an opened file
    /// \param f Already opened fstream file with write access
    ///
    /// Write the actual value of the analog data to a file
    ///
    void write(std::fstream &f) const;


    //---- METADATA ----//
protected:
    std::string _name; ///< Name of the channel
public:
    ///
    /// \brief Get the name of the channel
    /// \return The name of a channel
    ///
    const std::string& name() const;

    ///
    /// \brief Set the name of the channel
    /// \param name The name of the channel
    ///
    void name(const std::string &name);


    //---- ACTUAL DATA ----//
protected:
    float _data; ///< Value of the analog data
public:
    ///
    /// \brief Get the value of the analog data
    /// \return The value of the analog data
    ///
    float data() const;

    ///
    /// \brief Set the value of the analog data
    /// \param value The value of the analog data
    ///
    void data(float value);

};


#endif

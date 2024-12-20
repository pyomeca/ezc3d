#ifndef CHANNEL_H
#define CHANNEL_H
///
/// \file Channel.h
/// \brief Declaration of Channel class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "ezc3d/ezc3dNamespace.h"

///
/// \brief Channel of an analogous data
///
class EZC3D_VISIBILITY ezc3d::DataNS::AnalogsNS::Channel {
  //---- CONSTRUCTORS ----//
public:
  ///
  /// \brief Create an empty analogous data channel
  ///
  EZC3D_API Channel();

  ///
  /// \brief Copy an analog channel
  /// \param channel The channel to copy
  ///
  EZC3D_API Channel(const ezc3d::DataNS::AnalogsNS::Channel &channel);

  ///
  /// \brief Create a filled Channel class at a given subframe from a given file
  /// \param c3d Reference to the c3d to copy the data in
  /// \param file File to copy the data from
  /// \param info The information about the analogs
  /// \param channelIndex The index of the channel currently created
  ///
  EZC3D_API Channel(ezc3d::c3d &c3d, std::fstream &file,
                    const AnalogsNS::Info &info, size_t channelIndex);

  //---- STREAM ----//
public:
  ///
  ///
  /// \brief Print the channel
  ///
  /// Print the value of the analog data to the console
  ///
  EZC3D_API void print() const;

  ///
  /// \brief Write the channel to an opened file
  /// \param f Already opened fstream file with write access
  /// \param scaleFactor The factor to scale the data with
  ///
  /// Write the value of the analog data to a file
  ///
  EZC3D_API void write(std::fstream &f, double scaleFactor) const;

  //---- DATA ----//
protected:
  double _data; ///< Value of the analog data
public:
  ///
  /// \brief Get the value of the analog data
  /// \return The value of the analog data
  ///
  EZC3D_API double data() const;

  ///
  /// \brief Set the value of the analog data
  /// \param value The value of the analog data
  ///
  EZC3D_API void data(double value);

  ///
  /// \brief Return if the channel is empty
  /// \return if the channel is empty
  ///
  EZC3D_API bool isEmpty() const;
};

#endif

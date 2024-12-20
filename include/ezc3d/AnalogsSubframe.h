#ifndef ANALOGS_SUBFRAME_H
#define ANALOGS_SUBFRAME_H
///
/// \file AnalogsSubframe.h
/// \brief Declaration of Subframe class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "ezc3d/Channel.h"
#include <vector>

///
/// \brief Subframe for the analogous data
///
class EZC3D_VISIBILITY ezc3d::DataNS::AnalogsNS::SubFrame {
  //---- CONSTRUCTORS ----//
public:
  ///
  /// \brief Create an empty subframe for analogous data
  ///
  EZC3D_API SubFrame();

  ///
  /// \brief Create a filled SubFrame class at a given frame from a given file
  /// \param c3d Reference to the c3d to copy the data in
  /// \param file File to copy the data from
  /// \param info The information about the analogs
  ///
  EZC3D_API SubFrame(ezc3d::c3d &c3d, std::fstream &file,
                     const AnalogsNS::Info &info);

  //---- STREAM ----//
public:
  ///
  ///
  /// \brief Print the subframe
  ///
  /// Print the subframe to the console by calling sequentially the print method
  /// of all of the analog channels
  ///
  EZC3D_API void print() const;

  ///
  /// \brief Write the subframe to an opened file
  /// \param f Already opened fstream file with write access
  /// \param scaleFactors The factor to scale the data with
  ///
  /// Write the subframe to a file by calling sequentially the write method of
  /// all of the analog channels
  ///
  EZC3D_API void write(std::fstream &f, std::vector<double> scaleFactors) const;

  //---- CHANNELS ----//
protected:
  std::vector<ezc3d::DataNS::AnalogsNS::Channel>
      _channels; ///< Holder for the channels
public:
  ///
  /// \brief Get the number of analog channels
  /// \return The number of channels
  ///
  EZC3D_API size_t nbChannels() const;

  ///
  /// \brief Resize the number of channels. Warning, this function drops data if
  /// channels are downsized. \param nChannels Number of channels in the
  /// subframe
  ///
  EZC3D_API void nbChannels(size_t nChannels);

  ///
  /// \brief Get a particular analog channel of index idx from the analogous
  /// data \param idx Index of the analog channel \return The analog channel
  ///
  /// Get a particular analog channel of index idx from the analogous data.
  ///
  /// Throw a std::out_of_range exception if idx is larger than the number of
  /// channels
  ///
  EZC3D_API const ezc3d::DataNS::AnalogsNS::Channel &channel(size_t idx) const;

  ///
  /// \brief Get a particular analog channel of index idx from the analogous
  /// data with write access \param idx Index of the analog channel \return The
  /// analog channel
  ///
  /// Get a particular analog channel of index idx from the analogous in the
  /// form of a non-const reference. The user can thereafter modify this analog
  /// channel at will, but with the caution it requires.
  ///
  /// Throw a std::out_of_range exception if idx is larger than the number of
  /// channels
  ///
  EZC3D_API ezc3d::DataNS::AnalogsNS::Channel &channel(size_t idx);

  ///
  /// \brief Add/replace a channel to the analog subframe data set
  /// \param channel the channel to add
  /// \param idx the index of the channel in the subframe data set
  ///
  /// Add or replace a particular channel to the subframe data set.
  ///
  /// If no idx is sent, then the channel is appended to the points data set.
  /// If the idx correspond to a pre-existing channel, it replaces it.
  /// If idx is larger than the number of channels, it resize the subframe
  /// accordingly and add the channel where it belongs but leaves the other
  /// created channels empty.
  ///
  EZC3D_API void channel(const ezc3d::DataNS::AnalogsNS::Channel &channel,
                         size_t idx = SIZE_MAX);

  ///
  /// \brief Get all the analog channels from the analogous data
  /// \return The analog channels
  ///
  EZC3D_API const std::vector<ezc3d::DataNS::AnalogsNS::Channel> &
  channels() const;

  ///
  /// \brief Return if the subframe is empty
  /// \return if the subframe is empty
  ///
  EZC3D_API bool isEmpty() const;
};

#endif

#ifndef ANALOGS_H
#define ANALOGS_H
///
/// \file Analogs.h
/// \brief Declaration of Analogs class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "ezc3d/AnalogsSubframe.h"

///
/// \brief Analog holder for C3D analogous data
///
class EZC3D_VISIBILITY ezc3d::DataNS::AnalogsNS::Analogs {
  //---- CONSTRUCTORS ----//
public:
  ///
  /// \brief Create an empty holder for the analogous data
  ///
  EZC3D_API Analogs();

  ///
  /// \brief Create a filled Analogs class at a given frame from a given file
  /// \param c3d Reference to the c3d to copy the data in
  /// \param file File to copy the data from
  /// \param info The information about the analogs
  ///
  EZC3D_API Analogs(ezc3d::c3d &c3d, std::fstream &file,
                    const AnalogsNS::Info &info);

  //---- STREAM ----//
public:
  ///
  ///
  /// \brief Print the subframes
  ///
  /// Print the subframes to the console by calling sequentially the print
  /// method of each subframes
  ///
  EZC3D_API void print() const;

  ///
  /// \brief Write the subframes to an opened file
  /// \param f Already opened fstream file with write access
  /// \param scaleFactors The factor to scale the data with
  ///
  /// Write all the subframes to a file by calling sequentially the write method
  /// of each subframe
  ///
  EZC3D_API void write(std::fstream &f, std::vector<double> scaleFactors) const;

  //---- SUBFRAME ----//
protected:
  std::vector<ezc3d::DataNS::AnalogsNS::SubFrame>
      _subframe; ///< Holder for the subframes
public:
  ///
  /// \brief Get the number of subframes
  /// \return The number of subframes
  ///
  EZC3D_API size_t nbSubframes() const;

  ///
  /// \brief Resize the number of subframes. Warning, this function drops data
  /// if subframes is downsized \param nbSubframes The number of subframes to be
  /// in the holder
  ///
  EZC3D_API void nbSubframes(size_t nbSubframes);

  ///
  /// \brief Get a particular subframe of index idx from the analogous data set
  /// \param idx The index of the subframe
  /// \return The Subframe
  ///
  /// Get a particular subframe of index idx from the analogous data set.
  ///
  /// Throw a std::out_of_range exception if idx is larger than the number of
  /// subframes
  ///
  EZC3D_API const ezc3d::DataNS::AnalogsNS::SubFrame &
  subframe(size_t idx) const;

  ///
  /// \brief Get a particular subframe of index idx from the analogous data set
  /// in order to be modified by the caller \param idx The index of the subframe
  /// \return A non-const reference to the subframe
  ///
  /// Get a particular subframe of index idx from the analogous data in the form
  /// of a non-const reference. The user can thereafter modify these points at
  /// will, but with the caution it requires.
  ///
  /// Throw a std::out_of_range exception if idx is larger than the number of
  /// subframes
  ///
  EZC3D_API ezc3d::DataNS::AnalogsNS::SubFrame &subframe(size_t idx);

  ///
  /// \brief Add/replace a subframe to the analogous data set
  /// \param subframe The subframe to add
  /// \param idx The index of the subframe in the analogous data set
  ///
  /// Add or replace a subframe to the analogous data set.
  ///
  /// If no idx is sent, then the subframe is appended to the analogous data
  /// set. If the idx correspond to a pre-existing subframe, it replaces it. If
  /// idx is larger than the number of subframes, it resize the analogous data
  /// set accordingly and add the subframe where it belongs but leaves the other
  /// created subframes empty.
  ///
  EZC3D_API void subframe(const ezc3d::DataNS::AnalogsNS::SubFrame &subframe,
                          size_t idx = SIZE_MAX);

  ///
  /// \brief Get all the subframes from the analogous data set
  /// \return The subframes
  ///
  EZC3D_API const std::vector<ezc3d::DataNS::AnalogsNS::SubFrame> &
  subframes() const;

  ///
  /// \brief Return if the analogs are empty
  /// \return if the analogs are empty
  ///
  EZC3D_API bool isEmpty() const;
};

#endif

#ifndef PARAMETERS_H
#define PARAMETERS_H
///
/// \file Parameters.h
/// \brief Declaration of Parameters class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "ezc3d/Group.h"

///
/// \brief Group holder of C3D parameters
///
class EZC3D_VISIBILITY ezc3d::ParametersNS::Parameters {
  //---- CONSTRUCTOR ----//
public:
  ///
  /// \brief Create a default group holder with minimal groups to have a valid
  /// c3d
  ///
  EZC3D_API Parameters();

  ///
  /// \brief Construct group holder from a C3D file
  /// \param c3d C3D reference to copy the data in
  /// \param file Already opened fstream file with read access
  /// \param ignoreBadFormatting If bad formatting of the c3d should be ignored,
  /// use with caution as it can results in a segmentation fault
  ///
  EZC3D_API Parameters(c3d &c3d, std::fstream &file,
                       bool ignoreBadFormatting = false);

public:
  ///
  /// \brief Test if a given group is mandatory or not
  /// \param groupName The name of the group to test
  /// \return If a given group is mandatory or not
  ///
  EZC3D_API static bool isMandatory(const std::string &groupName);

  ///
  /// \brief Test if a given parameter is mandatory or not
  /// \param groupName The name of the group that contains the parameter
  /// \param parameterName The name of the parameter to test
  /// \return If a given parameter is mandatory or not
  ///
  EZC3D_API static bool isMandatory(const std::string &groupName,
                                    const std::string &parameterName);

protected:
  ///
  /// \brief Add all required parameters for a c3d to be valid
  ///
  void setMandatoryParameters();

  ///
  /// \brief Set the required parameters for some specific but non-mandatory
  /// groups \param groupName The name of the group to add to the c3d
  ///
  void setMandatoryParametersForSpecialGroup(const std::string &groupName);

  //---- STREAM ----//
public:
  ///
  /// \brief Print the groups by calling the print method of all the groups
  ///
  EZC3D_API void print() const;

  ///
  /// \brief Write the groups to an opened file by calling the write method of
  /// all the groups \param f Already opened fstream file with write access
  /// \param dataStartPositionToFill Returns the byte where to put the data
  /// start parameter \param header A reference to the header section \param
  /// format What order should the file has
  ///
  EZC3D_API ezc3d::ParametersNS::Parameters
  write(std::fstream &f, ezc3d::DataStartInfo &dataStartPositionToFill,
        const ezc3d::Header &header,
        const ezc3d::WRITE_FORMAT &format = ezc3d::WRITE_FORMAT::DEFAULT) const;

  //---- PARAMETER METADATA ----//
protected:
  ///
  /// \brief Prepare a copy of all parameters that will be used to write
  /// \param header A reference to the header section
  /// \param format What order should the file has
  /// \return
  ///
  Parameters prepareCopyForWriting(
      const ezc3d::Header &header,
      const ezc3d::WRITE_FORMAT &format = ezc3d::WRITE_FORMAT::DEFAULT) const;

  // Read the Parameters Header
  size_t
      _parametersStart; ///< Byte 1 of the parameter's section of the C3D file.
                        ///<
                        ///< If the value is 1 then it starts at byte 3
                        ///< otherwise, it starts at byte 512*parametersStart
  size_t _checksum;     ///< Byte 2 of the C3D file
                        ///<
                        ///< It should be equals to 0x50 for a valid a c3d
  size_t _nbParamBlock; ///< Byte 3 of the C3D file
                        ///<
                        ///< Number of 256-bytes blocks the paramertes fits in.
                        ///< It defines the starting position of the data
  PROCESSOR_TYPE
  _processorType; ///< Byte 4 of the C3D file
                  ///<
                  ///< Processor type (83 + [1 Inter, 2 DEC, 3 MIPS])

public:
  ///
  /// \brief Get the block index (bytes = block_index * 256) in the file where
  /// the data starts \return The block in the file where the data starts
  ///
  EZC3D_API size_t parametersStart() const;

  ///
  /// \brief Get the checksum of the parameters
  /// \return The checksum of the parameters
  ///
  /// The chechsum, according to C3D.org documentation, should be equals to 0x50
  /// for a valid C3D
  ///
  EZC3D_API size_t checksum() const;

  ///
  /// \brief Get the number of 256-bytes the parameters need in the file
  /// \return The number of 256-bytes the parameters need in the file
  ///
  EZC3D_API size_t nbParamBlock() const;

  ///
  /// \brief Get the processor type the file was writen on
  /// \return The processor type the file was writen on
  ///
  /// The processor type is defined by the value 83 + index. Where index is 1
  /// for Intel, 2 for DEC and 3 for MIPS
  ///
  EZC3D_API PROCESSOR_TYPE processorType() const;

  //---- GROUPS ----//
protected:
  std::vector<ezc3d::ParametersNS::GroupNS::Group>
      _groups; ///< Holder for the group of parameters

public:
  /// \brief Get the number of groups
  /// \return The number of groups
  ///
  EZC3D_API size_t nbGroups() const;

  ///
  /// \brief Return if a group of a specific name exists
  /// \param groupName The group name to return
  /// \return If the group exists (true) or not (false)
  ///
  EZC3D_API bool isGroup(const std::string &groupName) const;

  ///
  /// \brief Get the index of a group in the group holder
  /// \param groupName Name of the group
  /// \return The index of the group
  ///
  /// Search for the index of a group into the group holder by the name of this
  /// group.
  ///
  /// Throw a std::invalid_argument if groupName is not found
  ///
  EZC3D_API size_t groupIdx(const std::string &groupName) const;

  ///
  /// \brief Get a particular group of index idx from the group holder
  /// \param idx The index of the group
  /// \return The group
  ///
  /// Get a particular group of index idx from the group holder.
  ///
  /// Throw a std::out_of_range exception if idx is larger than the number of
  /// groups
  ///
  EZC3D_API const ezc3d::ParametersNS::GroupNS::Group &group(size_t idx) const;

  ///
  /// \brief Get a particular group of index idx from the group holder in order
  /// to be modified by the caller \param idx The index of the group \return The
  /// group
  ///
  /// Get a particular group of index idx from the group holder in the form of a
  /// non-const reference. The user can thereafter modify the parameter at will,
  /// but with the caution it requires.
  ///
  /// Throw a std::out_of_range exception if idx is larger than the number of
  /// groups
  ///
  EZC3D_API ezc3d::ParametersNS::GroupNS::Group &group(size_t idx);

  ///
  /// \brief Get a particular group with the name groupName from the group
  /// holder \param groupName The name of the group \return The group
  ///
  /// Throw a std::invalid_argument if groupName is not found
  ///
  EZC3D_API const ezc3d::ParametersNS::GroupNS::Group &
  group(const std::string &groupName) const;

  ///
  /// \brief Get a particular group with the name groupName from the group
  /// holder \param groupName The name of the group \return The group
  ///
  /// Throw a std::invalid_argument if groupName is not found
  ///
  EZC3D_API ezc3d::ParametersNS::GroupNS::Group &
  group(const std::string &groupName);

  ///
  /// \brief Add/replace a group in the group holder
  /// \param group The group to copy
  ///
  /// If the group sent does not exist in the group holder, it is appended.
  /// Otherwise it is replaced
  ///
  EZC3D_API void group(const ezc3d::ParametersNS::GroupNS::Group &group);

  ///
  /// \brief Remove a group
  /// \param name The name of the group to remove
  ///
  EZC3D_API void remove(const std::string &name);

  ///
  /// \brief Remove a group
  /// \param idx The index of the group to remove
  ///
  EZC3D_API void remove(size_t idx);

  ///
  /// \brief Get all groups the group holder with read-only access
  /// \return The groups
  ///
  EZC3D_API const std::vector<ezc3d::ParametersNS::GroupNS::Group> &
  groups() const;
};

#endif

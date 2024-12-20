#ifndef GROUP_H
#define GROUP_H
///
/// \file Group.h
/// \brief Declaration of Group class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "ezc3d/Parameter.h"

///
/// \brief Group of parameter of a C3D file
///
class EZC3D_VISIBILITY ezc3d::ParametersNS::GroupNS::Group {
  //---- CONSTRUCTOR ----//
public:
  ///
  /// \brief Create an empty group of parameter
  /// \param name The name of the group of parameter
  /// \param description The description of the group of parameter
  ///
  EZC3D_API Group(const std::string &name = "",
                  const std::string &description = "");

  //---- STREAM ----//
public:
  ///
  /// \brief Print the group by calling the print method of all the parameters
  ///
  EZC3D_API void print() const;

  ///
  /// \brief Write the group to an opened file by calling the write method of
  /// all the parameters \param f Already opened fstream file with write access
  /// \param groupIdx Index of the group that this particular parameter is in
  /// \param dataStartPositionToFill The position in the file where the data
  /// start (special case for POINT:DATA_START and ROTATION:DATA_START
  /// parameters)
  ///
  EZC3D_API void write(std::fstream &f, int groupIdx,
                       ezc3d::DataStartInfo &dataStartPositionToFill) const;

  ///
  /// \brief Read and store a group of parameter from an opened C3D file
  /// \param c3d C3D reference to copy the data in
  /// \param params Reference to a valid parameter
  /// \param file The file stream already opened with read access
  /// \param nbCharInName The number of character of the group name
  /// \return The position in the file of the next Group/Parameter
  ///
  EZC3D_API int read(ezc3d::c3d &c3d, const Parameters &params,
                     std::fstream &file, int nbCharInName);

  ///
  /// \brief isEmpty If the group has no name and no parameter, it is considered
  /// empty \return If the group is empty
  ///
  EZC3D_API bool isEmpty() const;

  //---- METADATA ----//
protected:
  std::string _name;        ///< The name of the group
  std::string _description; ///< The description of the group
  bool _isLocked;           ///< The lock status of the group

public:
  ///
  /// \brief Get the name of the group
  /// \return The name of the group
  ///
  EZC3D_API const std::string &name() const;

  ///
  /// \brief Set the name of the group
  /// \param name The name of the group
  ///
  EZC3D_API void name(const std::string &name);

  ///
  /// \brief Get the description of the group
  /// \return The description of the group
  ///
  EZC3D_API const std::string &description() const;

  ///
  /// \brief Set the description of the group
  /// \param description The description of the group
  ///
  EZC3D_API void description(const std::string &description);

  ///
  /// \brief Get the locking status of the group
  /// \return The locking status of the group
  ///
  EZC3D_API bool isLocked() const;

  ///
  /// \brief Set the locking status of the group to true
  ///
  EZC3D_API void lock();

  ///
  /// \brief Set the locking status of the group to false
  ///
  EZC3D_API void unlock();

  //---- PARAMETERS ----/////
protected:
  std::vector<ezc3d::ParametersNS::GroupNS::Parameter>
      _parameters; ///< Holder for the parameters of the group

public:
  /// \brief Get the number of parameters
  /// \return The number of parameters
  ///
  EZC3D_API size_t nbParameters() const;

  ///
  /// \brief Return if a parameter of a specific name exists
  /// \param parameterName The parameter name to return
  /// \return If the parameter exists (true) or not (false)
  ///
  EZC3D_API bool isParameter(const std::string &parameterName) const;

  ///
  /// \brief Get the index of a parameter in the group
  /// \param parameterName Name of the parameter
  /// \return The index of the parameter
  ///
  /// Search for the index of a parameter into the group by the name of this
  /// parameter.
  ///
  /// Throw a std::invalid_argument if parameterName is not found
  ///
  EZC3D_API size_t parameterIdx(const std::string &parameterName) const;

  ///
  /// \brief Get a particular parameter of index idx from the group
  /// \param idx The index of the parameter
  /// \return The parameter
  ///
  /// Get a particular point of index idx from the group.
  ///
  /// Throw a std::out_of_range exception if idx is larger than the number of
  /// parameters
  ///
  EZC3D_API const ezc3d::ParametersNS::GroupNS::Parameter &
  parameter(size_t idx) const;

  ///
  /// \brief Get a particular parameter of index idx from the group in order to
  /// be modified by the caller \param idx The index of the parameter \return
  /// The parameter
  ///
  /// Get a particular parameter of index idx from the group in the form of a
  /// non-const reference. The user can thereafter modify the parameter at will,
  /// but with the caution it requires.
  ///
  /// Throw a std::out_of_range exception if idx is larger than the number of
  /// parameters
  ///
  EZC3D_API ezc3d::ParametersNS::GroupNS::Parameter &parameter(size_t idx);

  ///
  /// \brief Get a particular parameter with the name parameterName from the
  /// group \param parameterName The name of the parameter \return The parameter
  ///
  /// Throw a std::invalid_argument if parameterName is not found
  ///
  EZC3D_API const ezc3d::ParametersNS::GroupNS::Parameter &
  parameter(const std::string &parameterName) const;

  ///
  /// \brief Get a particular parameter with the name parameterName from the
  /// group in the form of a non-const reference. \param parameterName The name
  /// of the parameter \return The parameter
  ///
  /// Get a particular parameterwith the name parameterName from the group in
  /// the form of a non-const reference. The user can thereafter modify the
  /// parameter at will, but with the caution it requires.
  ///
  /// Throw a std::invalid_argument if parameterName is not found
  ///
  EZC3D_API ezc3d::ParametersNS::GroupNS::Parameter &
  parameter(const std::string &parameterName);

  ///
  /// \brief Add a parameter to the group from a C3D file
  /// \param c3d C3D reference to copy the data in
  /// \param params Reference to a valid parameter
  /// \param file The file stream already opened with read access
  /// \param nbCharInName The number of character of the parameter name
  /// \return
  ///
  EZC3D_API int parameter(c3d &c3d, const Parameters &params,
                          std::fstream &file, int nbCharInName);

  ///
  /// \brief Add/replace a parameter to the group
  /// \param parameter The parameter to add
  ///
  /// If the parameter sent does not exist in the group, it is appended.
  /// Otherwise it is replaced
  ///
  EZC3D_API void
  parameter(const ezc3d::ParametersNS::GroupNS::Parameter &parameter);

  ///
  /// \brief Remove a parameter from a group
  /// \param name The name of the parameter to remove
  ///
  EZC3D_API void remove(const std::string &name);

  ///
  /// \brief Remove a parameter from a group
  /// \param idx The index of the parameter to remove
  ///
  EZC3D_API void remove(size_t idx);

  ///
  /// \brief Get all the parameter from the group
  /// \return The parameters
  ///
  EZC3D_API const std::vector<ezc3d::ParametersNS::GroupNS::Parameter> &
  parameters() const;
};

#endif

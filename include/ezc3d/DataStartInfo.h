#ifndef DATA_START_INFO_H
#define DATA_START_INFO_H
///
/// \file DataStartInfo.h
/// \brief Declaration of DataStartInfo class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "ezc3d/ezc3dNamespace.h"
#include <sstream>

///
/// \brief Placeholder for stocking the position and value of data start
///
class EZC3D_VISIBILITY ezc3d::DataStartInfo {

protected:
  bool m_hasPointDataStart = false; ///< If the point data start is set
  std::streampos m_pointDataStart;  ///< The data start for the points

public:
  ///
  /// \brief Returns if the point data start is set
  /// \return If the point data start is set
  ///
  EZC3D_API bool hasPointDataStart() const;

  ///
  /// \brief The point data start
  /// \param value The starting position of the points data in the c3d file
  ///
  EZC3D_API void setPointDataStart(const std::streampos &value);

  ///
  /// \brief Get the point data start
  /// \return The point data start
  ///
  EZC3D_API const std::streampos &pointDataStart() const;

protected:
  bool m_hasHeaderPointDataStart =
      false; ///< If the point data start is set for the header
  std::streampos m_headerPointDataStart; ///< Position in the c3d to put the
                                         ///< point start start in the header
  DATA_TYPE m_headerPointDataStartSize =
      DATA_TYPE::WORD; ///< The size of the value in the c3d file

public:
  ///
  /// \brief Returns if the point data start is set for the header
  /// \return If the point data start is set for the header
  ///
  EZC3D_API bool hasHeaderPointDataStart() const;

  ///
  /// \brief The position in the c3d where to put the header point data start
  /// \param position
  ///
  EZC3D_API void
  setHeaderPositionInC3dForPointDataStart(const std::streampos &position);

  ///
  /// \brief Get the position in the c3d to put the point start start in the
  /// header \return The position in the c3d to put the point start start in the
  /// header
  ///
  EZC3D_API const std::streampos &headerPointDataStart() const;

  ///
  /// \brief Get the size of the value in the c3d file header
  /// \return The size of the value in the c3d file header
  ///
  EZC3D_API DATA_TYPE headerPointDataStartSize() const;

protected:
  bool m_hasParameterPointDataStart =
      false; ///< If the point data start is set for the parameters
  std::streampos
      m_parameterPointDataStart; ///< Position in the c3d to put the point start
                                 ///< start in the parameters
  DATA_TYPE m_parameterPointDataStartSize =
      DATA_TYPE::BYTE; ///< The size of the value in the c3d file

public:
  ///
  /// \brief Returns if the point data start is set for the parameters
  /// \return If the point data start is set for the parameters
  ///
  EZC3D_API bool hasParameterPointDataStart() const;
  ///
  /// \brief The position in the c3d where to put the parameter point data start
  /// \param position
  ///
  EZC3D_API void
  setParameterPositionInC3dForPointDataStart(const std::streampos &position);

  ///
  /// \brief Get the position in the c3d to put the point start start in the
  /// parameters \return The position in the c3d to put the point start start in
  /// the parameters
  ///
  EZC3D_API const std::streampos &parameterPointDataStart() const;

  ///
  /// \brief Get the size of the value in the c3d file parameter
  /// \return The size of the value in the c3d file parameter
  ///
  DATA_TYPE parameterPointDataStartSize() const;

protected:
  bool m_hasRotationDataStart = false; ///< If the rotation data start is set
  std::streampos m_rotationsDataStart; ///< The data start for the rotations

public:
  ///
  /// \brief Returns if the rotations data start is set
  /// \return If the rotations data start is set
  ///
  EZC3D_API bool hasRotationsDataStart() const;

  ///
  /// \brief The rotations data start
  /// \param value The starting position of the rotations data in the c3d file
  ///
  EZC3D_API void setRotationsDataStart(const std::streampos &value);

  ///
  /// \brief Get the rotations data start
  /// \return The rotations data start
  ///
  EZC3D_API const std::streampos &rotationsDataStart() const;

protected:
  bool m_hasParameterRotationsDataStart =
      false; ///< If the rotations data start is set for the parameters
  std::streampos m_parameterRotationsDataStart; ///< Position in the c3d to put
                                                ///< the rotations start start
                                                ///< in the parameters
  DATA_TYPE m_parameterRotationsDataStartSize =
      DATA_TYPE::WORD; ///< The size of the value in the c3d file

public:
  ///
  /// \brief Returns if the rotations data start is set for the parameters
  /// \return If the rotations data start is set for the parameters
  ///
  EZC3D_API bool hasParameterRotationsDataStart() const;

  ///
  /// \brief The position in the c3d where to put the parameter rotations data
  /// start \param position
  ///
  EZC3D_API void setParameterPositionInC3dForRotationsDataStart(
      const std::streampos &position);

  ///
  /// \brief Get the position in the c3d to put the rotations start start in the
  /// parameters \return The position in the c3d to put the rotations start
  /// start in the parameters
  ///
  EZC3D_API const std::streampos &parameterRotationsDataStart() const;

  ///
  /// \brief Get the size of the value in the c3d file parameter
  /// \return The size of the value in the c3d file parameter
  ///
  EZC3D_API DATA_TYPE parameterRotationsDataStartSize() const;
};

#endif

#ifndef DATA_START_INFO_H
#define DATA_START_INFO_H
///
/// \file DataStartInfo.h
/// \brief Declaration of DataStartInfo class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "ezc3dNamespace.h"
///
/// \brief Placeholder for stocking the position and value of data start
///
class EZC3D_API ezc3d::DataStartInfo{

protected:
    std::streampos m_pointDataStart = -1;  ///< The data start for the points

public:
    ///
    /// \brief The point data start
    /// \param value The starting position of the points data in the c3d file
    ///
    void setPointDataStart(const std::streampos& value);

    ///
    /// \brief Get the point data start
    /// \return The point data start
    ///
    const std::streampos& pointDataStart() const;


protected:
    std::streampos m_headerPointDataStart = -1;  ///< Position in the c3d to put the point start start in the header
    DATA_TYPE m_headerPointDataStartSize = DATA_TYPE::WORD;  ///< The size of the value in the c3d file

public:
    ///
    /// \brief The position in the c3d where to put the header point data start
    /// \param position
    ///
    void setHeaderPositionInC3dForPointDataStart(const std::streampos& position);

    ///
    /// \brief Get the position in the c3d to put the point start start in the header
    /// \return The position in the c3d to put the point start start in the header
    ///
    const std::streampos& headerPointDataStart() const;

    ///
    /// \brief Get the size of the value in the c3d file header
    /// \return The size of the value in the c3d file header
    ///
    DATA_TYPE headerPointDataStartSize() const;

protected:
    std::streampos m_parameterPointDataStart = -1;  ///< Position in the c3d to put the point start start in the parameters
    DATA_TYPE m_parameterPointDataStartSize = DATA_TYPE::BYTE;  ///< The size of the value in the c3d file

public:
    ///
    /// \brief The position in the c3d where to put the parameter point data start
    /// \param position
    ///
    void setParameterPositionInC3dForPointDataStart(const std::streampos& position);

    ///
    /// \brief Get the position in the c3d to put the point start start in the parameters
    /// \return The position in the c3d to put the point start start in the parameters
    ///
    const std::streampos& parameterPointDataStart() const;

    ///
    /// \brief Get the size of the value in the c3d file parameter
    /// \return The size of the value in the c3d file parameter
    ///
    DATA_TYPE parameterPointDataStartSize() const;

protected:
    std::streampos m_rotationsDataStart = -1;  ///< The data start for the rotations

public:
    ///
    /// \brief The rotations data start
    /// \param value The starting position of the rotations data in the c3d file
    ///
    void setRotationsDataStart(const std::streampos& value);

    ///
    /// \brief Get the rotations data start
    /// \return The rotations data start
    ///
    const std::streampos& rotationsDataStart() const;

protected:
    std::streampos m_parameterRotationsDataStart = -1;  ///< Position in the c3d to put the rotations start start in the parameters
    DATA_TYPE m_parameterRotationsDataStartSize = DATA_TYPE::BYTE;  ///< The size of the value in the c3d file

public:
    ///
    /// \brief The position in the c3d where to put the parameter rotations data start
    /// \param position
    ///
    void setParameterPositionInC3dForRotationsDataStart(const std::streampos& position);

    ///
    /// \brief Get the position in the c3d to put the rotations start start in the parameters
    /// \return The position in the c3d to put the rotations start start in the parameters
    ///
    const std::streampos& parameterRotationsDataStart() const;

    ///
    /// \brief Get the size of the value in the c3d file parameter
    /// \return The size of the value in the c3d file parameter
    ///
    DATA_TYPE parameterRotationsDataStartSize() const;
};

#endif

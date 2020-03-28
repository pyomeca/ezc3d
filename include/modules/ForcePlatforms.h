#ifndef MODULES__FORCE_PLATFORM_H
#define MODULES__FORCE_PLATFORM_H
///
/// \file ForcePlatform.h
/// \brief Analyses of the analogous data in order to extract the platform
/// \author Pariterre
/// \version 1.0
/// \date March 25th, 2020
///

#include "ezc3d.h"
#include "Vector3d.h"

///
/// \brief Force Platform analyse
///
class EZC3D_API ezc3d::Modules::ForcePlatform{
    //---- CONSTRUCTORS ----//   
public:
    ///
    /// \brief Declare a ForcePlatForm analyse
    /// \param idx Index of the force platform
    /// \param c3d A reference to the c3d class
    ///
    ForcePlatform(
            size_t idx,
            const ezc3d::c3d& c3d);

    //---- DATA ----//
protected:
    size_t _type;  ///< The type of force platform (see C3D documentation)
    std::vector<ezc3d::Vector3d> _corners;  ///< Position of the 4 corners of the force platform
    ezc3d::Vector3d _meanCorners;  ///< Mean position of the corners of the force platform
    ezc3d::Vector3d _origin;  ///< Position of the origin of the force platform
    ezc3d::Matrix _refFrame;  ///< The reference frame of the force plate in the global reference frame
    std::vector<ezc3d::Vector3d> _F;  ///< Force vectors for all instants (including subframes) in global reference frame
    std::vector<ezc3d::Vector3d> _M;  ///< Moment vectors for all instants (including subframes) in global reference frame
    std::vector<ezc3d::Vector3d> _CoP;  ///< Center of Pressure vectors for all instants (including subframes) in global reference frame
    std::vector<ezc3d::Vector3d> _Tz;  ///< Moment [0, 0, Tz] vectors for all instants (including subframes) expressed at the CoP

public:
    ///
    /// \brief Returns the force vectors at each frame in the global reference frame
    /// \return The force vectors at each frame in the global reference frame
    ///
    const std::vector<ezc3d::Vector3d>& forces() const;

    ///
    /// \brief Returns the moment vectors at each frame in the global reference frame at origin
    /// \return The moment vectors at each frame in the global reference frame at origin
    ///
    const std::vector<ezc3d::Vector3d>& moments() const;

    ///
    /// \brief Returns the center of pressure at each frame in the global reference frame
    /// \return The center of pressure at each frame in the global reference frame at origin
    ///
    const std::vector<ezc3d::Vector3d>& CoP() const;

    ///
    /// \brief Returns the moments at each frame in the global reference frame at center of pressure
    /// \return The moments at each frame in the global reference frame at center of pressure
    ///
    const std::vector<ezc3d::Vector3d>& Tz() const;

private:
    ///
    /// \brief Extract the force platform's type from the parameters
    /// \param idx Index of the platform
    /// \param c3d A reference to the c3d
    ///
    void extractType(
            size_t idx,
            const ezc3d::c3d &c3d);

    ///
    /// \brief Extract the corner positions from the parameters
    /// \param idx Index of the platform
    /// \param c3d A reference to the c3d
    ///
    void extractCorners(
            size_t idx,
            const ezc3d::c3d &c3d);

    ///
    /// \brief Extract the origin position from the parameters
    /// \param idx Index of the platform
    /// \param c3d A reference to the c3d
    ///
    void extractOrigin(
            size_t idx,
            const ezc3d::c3d &c3d);

    ///
    /// \brief From the position of the force platform, construct the reference
    /// frame
    ///
    void computePfReferenceFrame();

    ///
    /// \brief Extract the platform data from the c3d
    /// \param idx Index of the platform
    /// \param c3d A reference to the c3d
    ///
    void extractData(
            size_t idx,
            const ezc3d::c3d &c3d);
};

///
/// \brief Force Platform analyse holder
///
class EZC3D_API ezc3d::Modules::ForcePlatforms{
    //---- CONSTRUCTORS ----//
public:
    ///
    /// \brief Declare a ForcePlatForm analyse holder
    ///
    ForcePlatforms(const ezc3d::c3d& c3d);

    //---- DATA ----//
private:
    std::vector<ezc3d::Modules::ForcePlatform> _platforms;  ///< All the platforms

public:
    ///
    /// \brief forcePlatform Returns the data from all the force plateforms
    /// \return The data from all the force plateforms
    ///
    const std::vector<ezc3d::Modules::ForcePlatform>& forcePlatforms() const;

    ///
    /// \brief forcePlatform Returns the data from a specified force plateform
    /// \param idx The index of the platform
    /// \return The data from a specified force plateform
    ///
    const ForcePlatform& forcePlatform(
            size_t idx) const;

};

#endif
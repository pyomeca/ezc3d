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

#include "math/Matrix33.h"
#include "math/Matrix66.h"
#include "math/Vector3d.h"

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

    //---- UNITS ----//
protected:
    std::string _unitsForce; ///< Units for the forces
    std::string _unitsMoment; ///< Units for the moments
    std::string _unitsPosition; ///< Units for the positions

public:
    ///
    /// \brief Return the units for the forces
    /// \return
    ///
    const std::string& forceUnit() const;

    ///
    /// \brief Return the units for the moments
    /// \return
    ///
    const std::string& momentUnit() const;

    ///
    /// \brief Return the units for the center of pressure, corners, origin
    /// \return
    ///
    const std::string& positionUnit() const;

protected:
    ///
    /// \brief Extract the units for the platform
    /// \param idx Index of the platform
    /// \param c3d A reference to the c3d
    ///
    void extractUnits(
            const ezc3d::c3d &c3d);

    //---- DATA ----//
protected:
    size_t _type;  ///< The type of force platform (see C3D documentation)
    ezc3d::Matrix66 _calMatrix; ///< The calibration matrix
    std::vector<ezc3d::Vector3d> _corners;  ///< Position of the 4 corners of the force platform
    ezc3d::Vector3d _meanCorners;  ///< Mean position of the corners of the force platform
    ezc3d::Vector3d _origin;  ///< Position of the origin of the force platform
    ezc3d::Matrix33 _refFrame;  ///< The reference frame of the force plate in the global reference frame
    std::vector<ezc3d::Vector3d> _F;  ///< Force vectors for all instants (including subframes) in global reference frame
    std::vector<ezc3d::Vector3d> _M;  ///< Moment vectors for all instants (including subframes) in global reference frame
    std::vector<ezc3d::Vector3d> _CoP;  ///< Center of Pressure vectors for all instants (including subframes) in global reference frame
    std::vector<ezc3d::Vector3d> _Tz;  ///< Moment [0, 0, Tz] vectors for all instants (including subframes) expressed at the CoP

public:
    ///
    /// \brief Returns the number of frame recorded
    /// \return The number of frame recorded
    ///
    size_t nbFrames() const;

    ///
    /// \brief Returns the type of the force platform
    /// \return The type of the force platform
    ///
    size_t type() const;

    ///
    /// \brief Returns the calibration matrix
    /// \return The calibration matrix
    ///
    const ezc3d::Matrix66& calMatrix() const;

    ///
    /// \brief Returns the corners of the force platform
    /// \return The corners of the force platform
    ///
    const std::vector<ezc3d::Vector3d>& corners() const;

    ///
    /// \brief Returns the geometrical center of the force platform
    /// \return The geometrical center of the force platform
    ///
    const ezc3d::Vector3d& meanCorners() const;

    ///
    /// \brief Returns the origin of the force platform
    /// \return The origin of the force platform
    ///
    const ezc3d::Vector3d& origin() const;

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

protected:
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
    /// \brief Extract the calibration matrix from the parameters
    /// \param idx Index of the platform
    /// \param c3d A reference to the c3d
    ///
    void extractCalMatrix(
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

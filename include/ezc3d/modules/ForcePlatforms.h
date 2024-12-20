#ifndef MODULES__FORCE_PLATFORM_H
#define MODULES__FORCE_PLATFORM_H
///
/// \file ForcePlatform.h
/// \brief Analyses of the analogous data in order to extract the platform
/// \author Pariterre
/// \version 1.0
/// \date March 25th, 2020
///

#include "ezc3d/ezc3dNamespace.h"

#include "ezc3d/math/Matrix33.h"
#include "ezc3d/math/Matrix66.h"
#include "ezc3d/math/Vector3d.h"
#include <iostream>

///
/// \brief Force Platform analyse
///
/// Warning: Something important to remember is that there is no easy way to
/// detect what is the upward vector. Consequently, `ezc3d` has to assume one.
/// The most common being Z-axis pointing upward, this is what is assume.
/// If one has a C3D with the Y-axis pointing upward, they must transform
/// their data accordingly in order to use the force platform filter.
///
class EZC3D_VISIBILITY ezc3d::Modules::ForcePlatform {
  //---- CONSTRUCTORS ----//
public:
  ///
  /// \brief Default constructor
  ///
  EZC3D_API ForcePlatform();

  ///
  /// \brief Declare a ForcePlatForm analyse
  /// \param idx Index of the force platform
  /// \param c3d A reference to the c3d class
  ///
  EZC3D_API ForcePlatform(size_t idx, const ezc3d::c3d &c3d);

  //---- UNITS ----//
protected:
  std::string _unitsForce;    ///< Units for the forces
  std::string _unitsMoment;   ///< Units for the moments
  std::string _unitsPosition; ///< Units for the positions

public:
  ///
  /// \brief Return the units for the forces
  /// \return
  ///
  EZC3D_API const std::string &forceUnit() const;

  ///
  /// \brief Return the units for the moments
  /// \return
  ///
  EZC3D_API const std::string &momentUnit() const;

  ///
  /// \brief Return the units for the center of pressure, corners, origin
  /// \return
  ///
  EZC3D_API const std::string &positionUnit() const;

protected:
  ///
  /// \brief Extract the units for the platform
  /// \param idx Index of the platform
  /// \param c3d A reference to the c3d
  ///
  void extractUnits(const ezc3d::c3d &c3d);

  //---- DATA ----//
protected:
  size_t _type; ///< The type of force platform (see C3D documentation)
  ezc3d::Matrix66 _calMatrix; ///< The calibration matrix
  std::vector<ezc3d::Vector3d>
      _corners; ///< Position of the 4 corners of the force platform
  ezc3d::Vector3d
      _meanCorners; ///< Mean position of the corners of the force platform
  ezc3d::Vector3d _origin;   ///< Position of the origin of the force platform
  ezc3d::Matrix33 _refFrame; ///< The reference frame of the force plate in the
                             ///< global reference frame
  std::vector<ezc3d::Vector3d>
      _F; ///< Force vectors for all instants (including subframes) in global
          ///< reference frame
  std::vector<ezc3d::Vector3d>
      _M; ///< Moment vectors for all instants (including subframes) in global
          ///< reference frame
  std::vector<ezc3d::Vector3d>
      _CoP; ///< Center of Pressure vectors for all instants (including
            ///< subframes) in global reference frame
  std::vector<ezc3d::Vector3d>
      _Tz; ///< Moment [0, 0, Tz] vectors for all instants (including subframes)
           ///< expressed at the CoP

  std::vector<double>
      _type3copPoly; ///< The polynomial coefficients of the Kistler platforms

public:
  ///
  /// \brief Returns the number of frame recorded
  /// \return The number of frame recorded
  ///
  EZC3D_API size_t nbFrames() const;

  ///
  /// \brief Returns the type of the force platform
  /// \return The type of the force platform
  ///
  EZC3D_API size_t type() const;

  ///
  /// \brief Returns the calibration matrix
  /// \return The calibration matrix
  ///
  EZC3D_API const ezc3d::Matrix66 &calMatrix() const;

  ///
  /// \brief Returns the corners of the force platform
  /// \return The corners of the force platform
  ///
  EZC3D_API const std::vector<ezc3d::Vector3d> &corners() const;

  ///
  /// \brief Returns the geometrical center of the force platform
  /// \return The geometrical center of the force platform
  ///
  EZC3D_API const ezc3d::Vector3d &meanCorners() const;

  ///
  /// \brief Returns the origin of the force platform
  /// \return The origin of the force platform
  ///
  EZC3D_API const ezc3d::Vector3d &origin() const;

  ///
  /// \brief Returns the force vectors at each frame in the global reference
  /// frame \return The force vectors at each frame in the global reference
  /// frame
  ///
  EZC3D_API const std::vector<ezc3d::Vector3d> &forces() const;

  ///
  /// \brief Returns the moment vectors at each frame in the global reference
  /// frame at origin \return The moment vectors at each frame in the global
  /// reference frame at origin
  ///
  EZC3D_API const std::vector<ezc3d::Vector3d> &moments() const;

  ///
  /// \brief Returns the center of pressure at each frame in the global
  /// reference frame \return The center of pressure at each frame in the global
  /// reference frame at origin
  ///
  EZC3D_API const std::vector<ezc3d::Vector3d> &CoP() const;

  ///
  /// \brief Returns the moments at each frame in the global reference frame at
  /// center of pressure \return The moments at each frame in the global
  /// reference frame at center of pressure
  ///
  EZC3D_API const std::vector<ezc3d::Vector3d> &Tz() const;

protected:
  ///
  /// \brief Extract the force platform's type from the parameters
  /// \param idx Index of the platform
  /// \param c3d A reference to the c3d
  ///
  void extractType(size_t idx, const ezc3d::c3d &c3d);

  ///
  /// \brief Extract the corner positions from the parameters
  /// \param idx Index of the platform
  /// \param c3d A reference to the c3d
  ///
  void extractCorners(size_t idx, const ezc3d::c3d &c3d);

  ///
  /// \brief Extract the origin position from the parameters
  /// \param idx Index of the platform
  /// \param c3d A reference to the c3d
  ///
  void extractOrigin(size_t idx, const ezc3d::c3d &c3d);

  ///
  /// \brief Extract the calibration matrix from the parameters
  /// \param idx Index of the platform
  /// \param c3d A reference to the c3d
  ///
  void extractCalMatrix(size_t idx, const ezc3d::c3d &c3d);

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
  void extractData(size_t idx, const ezc3d::c3d &c3d);
};

///
/// \brief Force Platform analyse holder
///
class EZC3D_VISIBILITY ezc3d::Modules::ForcePlatforms {
  //---- CONSTRUCTORS ----//
public:
  ///
  /// \brief Declare a ForcePlatForm analyse holder
  ///
  EZC3D_API ForcePlatforms(const ezc3d::c3d &c3d);

  //---- DATA ----//
private:
  std::vector<ezc3d::Modules::ForcePlatform> _platforms; ///< All the platforms

public:
  ///
  /// \brief forcePlatform Returns the data from all the force plateforms
  /// \return The data from all the force plateforms
  ///
  EZC3D_API const std::vector<ezc3d::Modules::ForcePlatform> &
  forcePlatforms() const;

  ///
  /// \brief forcePlatform Returns the data from a specified force plateform
  /// \param idx The index of the platform
  /// \return The data from a specified force plateform
  ///
  EZC3D_API const ForcePlatform &forcePlatform(size_t idx) const;
};

#endif

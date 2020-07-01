
#include <vector>
#include "ezc3d_all.h"

int main()
{
    ezc3d::c3d c3d("c3dExampleFiles/Qualisys.c3d");
    ezc3d::Modules::ForcePlatforms pf(c3d);
    auto& pf0(pf.forcePlatform(0));

    // Show some metadata
    std::cout << "Number of force platform (pf) = " << pf.forcePlatforms().size() << std::endl;
    std::cout << std::endl;
    std::cout << "Information for pf 0:" << std::endl;
    std::cout << "Number of frames = " << pf0.nbFrames() << std::endl;
    std::cout << "Type = " << pf0.type() << std::endl;
    std::cout << "Force units = " << pf0.forceUnit() << std::endl;
    std::cout << "Moment units = " << pf0.momentUnit() << std::endl;
    std::cout << "Position units = " << pf0.positionUnit() << std::endl;
    std::cout << "Calibration matrix = " << std::endl << pf0.calMatrix() << std::endl;
    std::cout << "Corners = " << std::endl << pf0.corners() << std::endl;
    std::cout << "Origin = " << std::endl << pf0.origin() << std::endl;
    std::cout << std::endl;

    // Show some data
    std::cout << "Data for 1st frame:" << std::endl;
    // Values
    std::cout << "Forces = " << pf0.forces()[0].T() << std::endl;
    std::cout << "Moments = " << pf0.moments()[0].T() << std::endl;
    std::cout << "CoP = " << pf0.CoP()[0].T() << std::endl;
    std::cout << "Tz = " << pf0.Tz()[0].T() << std::endl;

    return 0;
}

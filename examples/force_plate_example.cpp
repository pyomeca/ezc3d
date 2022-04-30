#include <vector>
#include "ezc3d_all.h"

int main()
{
    ezc3d::c3d c3d("c3dExampleFiles/Qualisys.c3d");
    ezc3d::Modules::ForcePlatforms pf(c3d);
    auto& pf0(pf.forcePlatform(0));

    // Show some metadata
    std::cout << "Number of force platform (pf) = " << pf.forcePlatforms().size() << "\n";
    std::cout << "\n";
    std::cout << "Information for pf 0:" << "\n";
    std::cout << "Number of frames = " << pf0.nbFrames() << "\n";
    std::cout << "Type = " << pf0.type() << "\n";
    std::cout << "Force units = " << pf0.forceUnit() << "\n";
    std::cout << "Moment units = " << pf0.momentUnit() << "\n";
    std::cout << "Position units = " << pf0.positionUnit() << "\n";
    std::cout << "Calibration matrix = " << "\n" << pf0.calMatrix() << "\n";
    std::cout << "Corners = " << "\n" << pf0.corners() << "\n";
    std::cout << "Origin = " << "\n" << pf0.origin() << "\n";
    std::cout << "\n";

    // Show some data
    std::cout << "Data for 1st frame:" << "\n";
    // Values
    std::cout << "Forces = " << pf0.forces()[0].T() << "\n";
    std::cout << "Moments = " << pf0.moments()[0].T() << "\n";
    std::cout << "CoP = " << pf0.CoP()[0].T() << "\n";
    std::cout << "Tz = " << pf0.Tz()[0].T() << "\n";

    return 0;
}

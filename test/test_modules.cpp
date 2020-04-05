#include <iostream>
#include <gtest/gtest.h>

#include "ezc3d_all.h"

TEST(ForcePlatForm, NoPlatForm){
    ezc3d::c3d c3d("c3dTestFiles/Vicon.c3d");
    ezc3d::Modules::ForcePlatforms pf(c3d);
    EXPECT_EQ(pf.forcePlatforms().size(), 0);
}

TEST(ForcePlatForm, Type2){
    ezc3d::c3d c3d("c3dTestFiles/Qualisys.c3d");
    ezc3d::Modules::ForcePlatforms pf(c3d);
    EXPECT_EQ(pf.forcePlatforms().size(), 2);
    EXPECT_THROW(pf.forcePlatform(2), std::out_of_range);

    // Frames
    EXPECT_EQ(pf.forcePlatform(0).nbFrames(),
              c3d.header().nbFrames() * c3d.header().nbAnalogByFrame());
    EXPECT_EQ(pf.forcePlatform(1).nbFrames(),
              c3d.header().nbFrames() * c3d.header().nbAnalogByFrame());

    // Type
    EXPECT_EQ(pf.forcePlatform(0).type(), 2);
    EXPECT_EQ(pf.forcePlatform(1).type(), 2);

    // Units
    EXPECT_STREQ(pf.forcePlatform(0).forceUnit().c_str(), "N");
    EXPECT_STREQ(pf.forcePlatform(0).momentUnit().c_str(), "Nmm");
    EXPECT_STREQ(pf.forcePlatform(0).positionUnit().c_str(), "mm");
    EXPECT_STREQ(pf.forcePlatform(1).forceUnit().c_str(), "N");
    EXPECT_STREQ(pf.forcePlatform(1).momentUnit().c_str(), "Nmm");
    EXPECT_STREQ(pf.forcePlatform(1).positionUnit().c_str(), "mm");

    // Values
    const std::vector<ezc3d::Vector3d>& forces(pf.forcePlatform(0).forces());
    const std::vector<ezc3d::Vector3d>& moments(pf.forcePlatform(0).moments());
    const std::vector<ezc3d::Vector3d>& cop(pf.forcePlatform(0).CoP());
    const std::vector<ezc3d::Vector3d>& Tz(pf.forcePlatform(0).Tz());

    EXPECT_DOUBLE_EQ(forces[0](0), 0.13992118835449219);
    EXPECT_DOUBLE_EQ(forces[0](1), 0.046148300170898438);
    EXPECT_DOUBLE_EQ(forces[0](2), -0.18352508544921872);

    EXPECT_DOUBLE_EQ(moments[0](0), 20.867615272954936);
    EXPECT_DOUBLE_EQ(moments[0](1), -4.622511359985765);
    EXPECT_DOUBLE_EQ(moments[0](2), -29.393223381101276);

    EXPECT_DOUBLE_EQ(cop[0](0), 228.81266090518048);
    EXPECT_DOUBLE_EQ(cop[0](1), 118.29556977523387);
    EXPECT_DOUBLE_EQ(cop[0](2), 0.0);

    EXPECT_DOUBLE_EQ(Tz[0](0), 0.0);
    EXPECT_DOUBLE_EQ(Tz[0](1), 0.0);
    EXPECT_DOUBLE_EQ(Tz[0](2), -44.140528790099872);

    // CAL_MATRIX
    for (size_t i=0; i<2; ++i){
        const auto& calMatrix(pf.forcePlatform(i).calMatrix());
        EXPECT_EQ(calMatrix.nbRows(), 6);
        EXPECT_EQ(calMatrix.nbCols(), 6);
        for (size_t j=0; j<6; ++j){
            for (size_t k=0; k<6; ++k){
                EXPECT_EQ(calMatrix(j, k), 0.0);
            }
        }
    }

    // CORNERS
    const auto& corners(pf.forcePlatform(1).corners());
    EXPECT_EQ(corners.size(), 4);
    EXPECT_FLOAT_EQ(corners[0].x(), 1017.0);
    EXPECT_FLOAT_EQ(corners[0].y(), 464.0);
    EXPECT_FLOAT_EQ(corners[0].z(), 0.0);

    EXPECT_FLOAT_EQ(corners[1].x(), 1017.0);
    EXPECT_FLOAT_EQ(corners[1].y(), 0.0);
    EXPECT_FLOAT_EQ(corners[1].z(), 0.0);

    EXPECT_FLOAT_EQ(corners[2].x(), 509.0);
    EXPECT_FLOAT_EQ(corners[2].y(), 0.0);
    EXPECT_FLOAT_EQ(corners[2].z(), 0.0);

    EXPECT_FLOAT_EQ(corners[3].x(), 509.0);
    EXPECT_FLOAT_EQ(corners[3].y(), 464.0);
    EXPECT_FLOAT_EQ(corners[3].z(), 0.0);

    const auto& meanCorners(pf.forcePlatform(1).meanCorners());
    EXPECT_FLOAT_EQ(meanCorners.x(), 763.0);
    EXPECT_FLOAT_EQ(meanCorners.y(), 232.0);
    EXPECT_FLOAT_EQ(meanCorners.z(), 0.0);

    const auto& origin(pf.forcePlatform(1).origin());
    EXPECT_FLOAT_EQ(origin.x(), 1.016);
    EXPECT_FLOAT_EQ(origin.y(), 0);
    EXPECT_FLOAT_EQ(origin.z(),  -36.322);
}

TEST(ForcePlatForm, Type4) {
    ezc3d::c3d c3d("c3dTestFiles/Cortex.c3d");
    ezc3d::Modules::ForcePlatforms pf(c3d);
    EXPECT_EQ(pf.forcePlatforms().size(), 2);
    EXPECT_THROW(pf.forcePlatform(2), std::out_of_range);

    // Frames
    EXPECT_EQ(pf.forcePlatform(0).nbFrames(),
              c3d.header().nbFrames() * c3d.header().nbAnalogByFrame());
    EXPECT_EQ(pf.forcePlatform(1).nbFrames(),
              c3d.header().nbFrames() * c3d.header().nbAnalogByFrame());

    // Type
    EXPECT_EQ(pf.forcePlatform(0).type(), 4);
    EXPECT_EQ(pf.forcePlatform(1).type(), 4);

    // Units
    EXPECT_STREQ(pf.forcePlatform(0).forceUnit().c_str(), "N");
    EXPECT_STREQ(pf.forcePlatform(0).momentUnit().c_str(), "Nmm");
    EXPECT_STREQ(pf.forcePlatform(0).positionUnit().c_str(), "mm");
    EXPECT_STREQ(pf.forcePlatform(1).forceUnit().c_str(), "N");
    EXPECT_STREQ(pf.forcePlatform(1).momentUnit().c_str(), "Nmm");
    EXPECT_STREQ(pf.forcePlatform(1).positionUnit().c_str(), "mm");

    // Values
    const std::vector<ezc3d::Vector3d>& forces(pf.forcePlatform(0).forces());
    const std::vector<ezc3d::Vector3d>& moments(pf.forcePlatform(0).moments());
    const std::vector<ezc3d::Vector3d>& cop(pf.forcePlatform(0).CoP());
    const std::vector<ezc3d::Vector3d>& Tz(pf.forcePlatform(0).Tz());

    EXPECT_DOUBLE_EQ(forces[0](0), -32.238374685103523);
    EXPECT_DOUBLE_EQ(forces[0](1), 158.77370163491972);
    EXPECT_DOUBLE_EQ(forces[0](2), 584.26677922098179);

    EXPECT_DOUBLE_EQ(moments[0](0), -227478.16512061583);
    EXPECT_DOUBLE_EQ(moments[0](1), 121251.84987341597);
    EXPECT_DOUBLE_EQ(moments[0](2), -45517.209036652457);

    EXPECT_DOUBLE_EQ(cop[0](0), 341.33145223639389);
    EXPECT_DOUBLE_EQ(cop[0](1), 124.03836138403432);
    EXPECT_DOUBLE_EQ(cop[0](2), 2.1286785066892508);

    EXPECT_DOUBLE_EQ(Tz[0](0), 0.061895476251777826);
    EXPECT_DOUBLE_EQ(Tz[0](1), -0.06201938977928554);
    EXPECT_DOUBLE_EQ(Tz[0](2), -15.489373202823376);

    // CAL_MATRIX
    for (size_t i=0; i<2; ++i){
        const auto& calMatrix(pf.forcePlatform(i).calMatrix());
        EXPECT_EQ(calMatrix.nbRows(), 6);
        EXPECT_EQ(calMatrix.nbCols(), 6);
        for (size_t j=0; j<6; ++j){
            for (size_t k=0; k<6; ++k){
                if (j == k){
                    if (j == 0 || j == 1){
                        EXPECT_EQ(calMatrix(j, k), 500.0);
                    }
                    else if (j == 2){
                        EXPECT_EQ(calMatrix(j, k), 1000.0);
                    }
                    else if (j == 3){
                        EXPECT_EQ(calMatrix(j, k), 800000.0);
                    }
                    else if (j == 4 || j == 5){
                        EXPECT_EQ(calMatrix(j, k), 400000.0);
                    }
                }
                else {
                    EXPECT_EQ(calMatrix(j, k), 0.0);
                }
            }
        }
    }

    // CORNERS
    const auto& corners(pf.forcePlatform(1).corners());
    EXPECT_EQ(corners.size(), 4);
    EXPECT_FLOAT_EQ(corners[0].x(), -293.892);
    EXPECT_FLOAT_EQ(corners[0].y(), 1403.719);
    EXPECT_FLOAT_EQ(corners[0].z(), -3.9355);

    EXPECT_FLOAT_EQ(corners[1].x(), 265.108);
    EXPECT_FLOAT_EQ(corners[1].y(), 1402.601);
    EXPECT_FLOAT_EQ(corners[1].z(), -3.3765);

    EXPECT_FLOAT_EQ(corners[2].x(), 261.552);
    EXPECT_FLOAT_EQ(corners[2].y(), -375.399);
    EXPECT_FLOAT_EQ(corners[2].z(), 3.7355);

    EXPECT_FLOAT_EQ(corners[3].x(), -297.448);
    EXPECT_FLOAT_EQ(corners[3].y(), -374.281);
    EXPECT_FLOAT_EQ(corners[3].z(), 3.1765);

    const auto& meanCorners(pf.forcePlatform(1).meanCorners());
    EXPECT_FLOAT_EQ(meanCorners.x(), -16.17);
    EXPECT_FLOAT_EQ(meanCorners.y(), 514.16);
    EXPECT_FLOAT_EQ(meanCorners.z(), -0.0999999);

    const auto& origin(pf.forcePlatform(1).origin());
    EXPECT_FLOAT_EQ(origin.x(), -279.0);
    EXPECT_FLOAT_EQ(origin.y(), 889.0);
    EXPECT_FLOAT_EQ(origin.z(),  -2.0);
}

#include <iostream>
#include "gtest/gtest.h"

#include "ezc3d.h"

TEST(c3dModifier, addPoints) {
    // Create an empty c3d and load a new one
    ezc3d::c3d new_c3d;

    // Test before adding anything
    EXPECT_EQ(new_c3d.parameters().checksum(), 80);

    // Setup some variables
    ezc3d::c3d c3d("example/markers_analogs.c3d"); // load a reference c3d
    int nMarkers = 3;
    int nFrames = 10;
    std::vector<std::string> markerNames;
    for (int m = 0; m < nMarkers; ++m)
        markerNames.push_back(c3d.parameters().group("POINT").parameter("LABELS").valuesAsString()[m]);

    // Add markers to the new c3d
    for (int m = 0; m < nMarkers; ++m)
        new_c3d.addMarker(markerNames[m]);
    for (int f = 0; f < nFrames; ++f){
        ezc3d::DataNS::Frame frame;
        ezc3d::DataNS::Points3dNS::Points pts;
        for (int m = 0; m < nMarkers; ++m){
            ezc3d::DataNS::Points3dNS::Point pt(c3d.data().frame(f).points().point(markerNames[m]));
            pts.add(pt);
        }
        frame.add(pts);
        new_c3d.addFrame(frame);
    }


    // Test if the header have been properly adjusted


    // Test if the parameters have been properly adjusted
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("USED").valuesAsInt().size(), 1);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("USED").valuesAsInt()[0], nMarkers);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("SCALE").valuesAsFloat().size(), 1);
    EXPECT_FLOAT_EQ(new_c3d.parameters().group("POINT").parameter("SCALE").valuesAsFloat()[0], -1);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("RATE").valuesAsFloat().size(), 1);
    EXPECT_FLOAT_EQ(new_c3d.parameters().group("POINT").parameter("RATE").valuesAsFloat()[0], 0);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("FRAMES").valuesAsInt().size(), 1);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("FRAMES").valuesAsInt()[0], nFrames);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("LABELS").valuesAsString().size(), nMarkers);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("DESCRIPTIONS").valuesAsString().size(), nMarkers);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("UNITS").valuesAsString().size(), nMarkers);
    for (int m = 0; m < nMarkers; ++m){
        EXPECT_STREQ(new_c3d.parameters().group("POINT").parameter("LABELS").valuesAsString()[m].c_str(), markerNames[m].c_str());
        EXPECT_STREQ(new_c3d.parameters().group("POINT").parameter("DESCRIPTIONS").valuesAsString()[m].c_str(), "");
        EXPECT_STREQ(new_c3d.parameters().group("POINT").parameter("UNITS").valuesAsString()[m].c_str(), "mm");
    }

//    // Test if the data has effectively been added
//    EXPECT_FLOAT_EQ(new_c3d.data().frame(0).points().point(markerName).x(), frame.points().point(markerName).x());
//    EXPECT_FLOAT_EQ(new_c3d.data().frame(0).points().point(markerName).y(), frame.points().point(markerName).y());
//    EXPECT_FLOAT_EQ(new_c3d.data().frame(0).points().point(markerName).z(), frame.points().point(markerName).z());
}


#include <iostream>
#include "gtest/gtest.h"

#include "ezc3d.h"

enum HEADER_TYPE{
    ALL,
    POINT_ONLY,
    ANALOG_ONLY,
    EVENT_ONLY,
    POINT_AND_ANALOG,
    POINT_AND_EVENT,
    ANALOG_AND_EVENT
};


enum PARAMETER_TYPE{
    HEADER,
    POINT,
    ANALOG,
    FORCE_PLATFORM
};

void defaultHeaderTest(const ezc3d::c3d& new_c3d, HEADER_TYPE type = HEADER_TYPE::ALL){

    // Generic stuff
    EXPECT_EQ(new_c3d.header().checksum(), 80);
    EXPECT_EQ(new_c3d.header().keyLabelPresent(), 0);
    EXPECT_EQ(new_c3d.header().firstBlockKeyLabel(), 0);
    EXPECT_EQ(new_c3d.header().fourCharPresent(), 12345);
    EXPECT_EQ(new_c3d.header().emptyBlock1(), 0);
    EXPECT_EQ(new_c3d.header().emptyBlock2(), 0);
    EXPECT_EQ(new_c3d.header().emptyBlock3(), 0);
    EXPECT_EQ(new_c3d.header().emptyBlock4(), 0);

    // Point stuff
    if (    type == HEADER_TYPE::POINT_ONLY ||
            type == HEADER_TYPE::POINT_AND_ANALOG ||
            type == HEADER_TYPE::POINT_AND_EVENT ||
            type == HEADER_TYPE::ALL){
        EXPECT_EQ(new_c3d.header().nb3dPoints(), 0);
        EXPECT_EQ(new_c3d.header().nbMaxInterpGap(), 10);
        EXPECT_EQ(new_c3d.header().scaleFactor(), -1);
        EXPECT_FLOAT_EQ(new_c3d.header().frameRate(), 0);
    }

    // Analog stuff
    if (    type == HEADER_TYPE::ANALOG_ONLY ||
            type == HEADER_TYPE::POINT_AND_ANALOG ||
            type == HEADER_TYPE::ANALOG_AND_EVENT ||
            type == HEADER_TYPE::ALL){
        EXPECT_EQ(new_c3d.header().nbAnalogsMeasurement(), 0);
        EXPECT_EQ(new_c3d.header().nbAnalogByFrame(), 0);
        EXPECT_EQ(new_c3d.header().nbAnalogs(), 0);
    }

    // Event stuff
    if (    type == HEADER_TYPE::EVENT_ONLY ||
            type == HEADER_TYPE::POINT_AND_EVENT ||
            type == HEADER_TYPE::ANALOG_AND_EVENT ||
            type == HEADER_TYPE::ALL){
        EXPECT_EQ(new_c3d.header().nbEvents(), 0);
        EXPECT_EQ(new_c3d.header().eventsTime().size(), 18);
        for (int e = 0; e < static_cast<int>(new_c3d.header().eventsTime().size()); ++e)
            EXPECT_FLOAT_EQ(new_c3d.header().eventsTime(e), 0);
        EXPECT_EQ(new_c3d.header().eventsLabel().size(), 18);
        for (int e = 0; e < static_cast<int>(new_c3d.header().eventsDisplay().size()); ++e)
            EXPECT_STREQ(new_c3d.header().eventsLabel(e).c_str(), "");
        EXPECT_EQ(new_c3d.header().eventsDisplay().size(), 9);
        for (int e = 0; e < static_cast<int>(new_c3d.header().eventsDisplay().size()); ++e)
            EXPECT_EQ(new_c3d.header().eventsDisplay(e), 0);
    }

    if (type == HEADER_TYPE::ALL){
        EXPECT_EQ(new_c3d.header().firstFrame(), 0);
        EXPECT_EQ(new_c3d.header().lastFrame(), 0);
        EXPECT_EQ(new_c3d.header().nbFrames(), 0);
    }
}


void defaultParametersTest(const ezc3d::c3d& new_c3d, PARAMETER_TYPE type){
    if (type == PARAMETER_TYPE::HEADER){
        EXPECT_EQ(new_c3d.parameters().checksum(), 80);
        EXPECT_EQ(new_c3d.parameters().groups().size(), 3);
    } else if (type == PARAMETER_TYPE::POINT){
        EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("USED").type(), ezc3d::INT);
        EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("USED").valuesAsInt().size(), 1);
        EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("USED").valuesAsInt()[0], 0);
        EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("SCALE").type(), ezc3d::FLOAT);
        EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("SCALE").valuesAsFloat().size(), 1);
        EXPECT_FLOAT_EQ(new_c3d.parameters().group("POINT").parameter("SCALE").valuesAsFloat()[0], -1);
        EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("RATE").type(), ezc3d::FLOAT);
        EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("RATE").valuesAsFloat().size(), 1);
        EXPECT_FLOAT_EQ(new_c3d.parameters().group("POINT").parameter("RATE").valuesAsFloat()[0], 0);
        // EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("FRAMES").valuesAsInt()[0], 0); // ignore because it changes if analog is present
        EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("FRAMES").type(), ezc3d::INT);
        EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("LABELS").type(), ezc3d::CHAR);
        EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("LABELS").valuesAsString().size(), 0);
        EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("DESCRIPTIONS").type(), ezc3d::CHAR);
        EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("DESCRIPTIONS").valuesAsString().size(), 0);
        EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("UNITS").type(), ezc3d::CHAR);
        EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("UNITS").valuesAsString().size(), 0);
    } else if (type == PARAMETER_TYPE::ANALOG) {
        EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("USED").type(), ezc3d::INT);
        EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("USED").valuesAsInt().size(), 1);
        EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("USED").valuesAsInt()[0], 0);
        EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("LABELS").type(), ezc3d::CHAR);
        EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("LABELS").valuesAsString().size(), 0);
        EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("DESCRIPTIONS").type(), ezc3d::CHAR);
        EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("DESCRIPTIONS").valuesAsString().size(), 0);
        EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("GEN_SCALE").type(), ezc3d::INT);
        EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("GEN_SCALE").valuesAsInt().size(), 1);
        EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("GEN_SCALE").valuesAsInt()[0], 1);
        EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("SCALE").type(), ezc3d::FLOAT);
        EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("SCALE").valuesAsFloat().size(), 0);
        EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("OFFSET").type(), ezc3d::INT);
        EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("OFFSET").valuesAsInt().size(), 0);
        EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("UNITS").type(), ezc3d::CHAR);
        EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("UNITS").valuesAsString().size(), 0);
        EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("RATE").type(), ezc3d::FLOAT);
        EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("RATE").valuesAsFloat().size(), 1);
        EXPECT_FLOAT_EQ(new_c3d.parameters().group("ANALOG").parameter("RATE").valuesAsFloat()[0], 0);
        EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("FORMAT").type(), ezc3d::CHAR);
        EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("FORMAT").valuesAsString().size(), 0);
        EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("BITS").type(), ezc3d::INT);
        EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("BITS").valuesAsInt().size(), 0);
    } else if (type == PARAMETER_TYPE::FORCE_PLATFORM){
        EXPECT_EQ(new_c3d.parameters().group("FORCE_PLATFORM").parameter("USED").type(), ezc3d::INT);
        EXPECT_EQ(new_c3d.parameters().group("FORCE_PLATFORM").parameter("USED").valuesAsInt().size(), 1);
        EXPECT_EQ(new_c3d.parameters().group("FORCE_PLATFORM").parameter("USED").valuesAsInt()[0], 0);
        EXPECT_EQ(new_c3d.parameters().group("FORCE_PLATFORM").parameter("TYPE").type(), ezc3d::INT);
        EXPECT_EQ(new_c3d.parameters().group("FORCE_PLATFORM").parameter("TYPE").valuesAsInt().size(), 0);
        EXPECT_EQ(new_c3d.parameters().group("FORCE_PLATFORM").parameter("ZERO").type(), ezc3d::INT);
        EXPECT_EQ(new_c3d.parameters().group("FORCE_PLATFORM").parameter("ZERO").valuesAsInt().size(), 2);
        EXPECT_EQ(new_c3d.parameters().group("FORCE_PLATFORM").parameter("ZERO").valuesAsInt()[0], 1);
        EXPECT_EQ(new_c3d.parameters().group("FORCE_PLATFORM").parameter("ZERO").valuesAsInt()[1], 0);
        EXPECT_EQ(new_c3d.parameters().group("FORCE_PLATFORM").parameter("CORNERS").type(), ezc3d::FLOAT);
        EXPECT_EQ(new_c3d.parameters().group("FORCE_PLATFORM").parameter("CORNERS").valuesAsFloat().size(), 0);
        EXPECT_EQ(new_c3d.parameters().group("FORCE_PLATFORM").parameter("ORIGIN").type(), ezc3d::FLOAT);
        EXPECT_EQ(new_c3d.parameters().group("FORCE_PLATFORM").parameter("ORIGIN").valuesAsFloat().size(), 0);
        EXPECT_EQ(new_c3d.parameters().group("FORCE_PLATFORM").parameter("CHANNEL").type(), ezc3d::INT);
        EXPECT_EQ(new_c3d.parameters().group("FORCE_PLATFORM").parameter("CHANNEL").valuesAsInt().size(), 0);
        EXPECT_EQ(new_c3d.parameters().group("FORCE_PLATFORM").parameter("CAL_MATRIX").type(), ezc3d::FLOAT);
        EXPECT_EQ(new_c3d.parameters().group("FORCE_PLATFORM").parameter("CAL_MATRIX").valuesAsFloat().size(), 0);
    }
}

TEST(initialize, newC3D){
    // Create an empty c3d and load a new one
    ezc3d::c3d new_c3d;

    // HEADER
    defaultHeaderTest(new_c3d);

    // PARAMETERS
    defaultParametersTest(new_c3d, PARAMETER_TYPE::HEADER);
    defaultParametersTest(new_c3d, PARAMETER_TYPE::POINT);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("FRAMES").valuesAsInt()[0], 0);
    defaultParametersTest(new_c3d, PARAMETER_TYPE::ANALOG);
    defaultParametersTest(new_c3d, PARAMETER_TYPE::FORCE_PLATFORM);

    // DATA
    EXPECT_EQ(new_c3d.data().frames().size(), 0);
}

TEST(c3dModifier, addPoints) {
    // Create an empty c3d and load a new one
    ezc3d::c3d new_c3d;

    // Setup some variables
    std::vector<std::string> markerNames = {"marker1", "marker2", "marker3"};
    size_t nMarkers = markerNames.size();
    // Add markers to the new c3d
    for (size_t m = 0; m < nMarkers; ++m)
        new_c3d.addMarker(markerNames[m]);

    size_t nFrames = 10;
    float frameRate = 100;
    ezc3d::ParametersNS::GroupNS::Parameter pointRate("RATE");
    pointRate.set(std::vector<float>()={frameRate}, {1});
    new_c3d.addParameter("POINT", pointRate);
    for (size_t f = 0; f < nFrames; ++f){
        ezc3d::DataNS::Frame frame;
        ezc3d::DataNS::Points3dNS::Points pts;
        for (size_t m = 0; m < nMarkers; ++m){
            ezc3d::DataNS::Points3dNS::Point pt;
            pt.name(markerNames[m]);
            // Generate some random data
            pt.x(static_cast<float>(2*f+3*m+1) / static_cast<float>(7.0));
            pt.y(static_cast<float>(2*f+3*m+2) / static_cast<float>(7.0));
            pt.z(static_cast<float>(2*f+3*m+3) / static_cast<float>(7.0));
            pts.add(pt);
        }
        frame.add(pts);
        new_c3d.addFrame(frame);
    }


    // HEADER
    // Things that should remain as default
    defaultHeaderTest(new_c3d, HEADER_TYPE::ANALOG_AND_EVENT);

    // Things that should have change
    EXPECT_EQ(new_c3d.header().nb3dPoints(), nMarkers);
    EXPECT_EQ(new_c3d.header().firstFrame(), 0);
    EXPECT_EQ(new_c3d.header().lastFrame(), nFrames - 1);
    EXPECT_EQ(new_c3d.header().nbMaxInterpGap(), 10);
    EXPECT_EQ(new_c3d.header().scaleFactor(), -1);
    EXPECT_FLOAT_EQ(new_c3d.header().frameRate(), frameRate);
    EXPECT_EQ(new_c3d.header().nbFrames(), nFrames);

    // PARAMETERS
    // Things that should remain as default
    defaultParametersTest(new_c3d, PARAMETER_TYPE::HEADER);
    defaultParametersTest(new_c3d, PARAMETER_TYPE::ANALOG);
    defaultParametersTest(new_c3d, PARAMETER_TYPE::FORCE_PLATFORM);

    // Things that should have change
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("USED").type(), ezc3d::INT);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("USED").valuesAsInt().size(), 1);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("USED").valuesAsInt()[0], nMarkers);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("SCALE").valuesAsFloat().size(), 1);
    EXPECT_FLOAT_EQ(new_c3d.parameters().group("POINT").parameter("SCALE").valuesAsFloat()[0], -1);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("RATE").type(), ezc3d::FLOAT);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("RATE").valuesAsFloat().size(), 1);
    EXPECT_FLOAT_EQ(new_c3d.parameters().group("POINT").parameter("RATE").valuesAsFloat()[0], frameRate);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("FRAMES").type(), ezc3d::INT);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("FRAMES").valuesAsInt().size(), 1);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("FRAMES").valuesAsInt()[0], nFrames);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("LABELS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("LABELS").valuesAsString().size(), nMarkers);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("DESCRIPTIONS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("DESCRIPTIONS").valuesAsString().size(), nMarkers);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("UNITS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("UNITS").valuesAsString().size(), nMarkers);
    for (size_t m = 0; m < nMarkers; ++m){
        EXPECT_STREQ(new_c3d.parameters().group("POINT").parameter("LABELS").valuesAsString()[m].c_str(), markerNames[m].c_str());
        EXPECT_STREQ(new_c3d.parameters().group("POINT").parameter("DESCRIPTIONS").valuesAsString()[m].c_str(), "");
        EXPECT_STREQ(new_c3d.parameters().group("POINT").parameter("UNITS").valuesAsString()[m].c_str(), "mm");
    }

    // DATA
    for (size_t f = 0; f < nFrames; ++f){
        for (size_t m = 0; m < nMarkers; ++m){
            EXPECT_FLOAT_EQ(new_c3d.data().frame(static_cast<int>(f)).points().point(markerNames[m]).x(), static_cast<float>(2*f+3*m+1) / static_cast<float>(7.0));
            EXPECT_FLOAT_EQ(new_c3d.data().frame(static_cast<int>(f)).points().point(markerNames[m]).y(), static_cast<float>(2*f+3*m+2) / static_cast<float>(7.0));
            EXPECT_FLOAT_EQ(new_c3d.data().frame(static_cast<int>(f)).points().point(markerNames[m]).z(), static_cast<float>(2*f+3*m+3) / static_cast<float>(7.0));
        }
    }

}

TEST(c3dModifier, addAnalogs) {
    // Create an empty c3d
    ezc3d::c3d new_c3d;

    // Setup some variables
    std::vector<std::string> analogNames = {"analog1", "analog2", "analog3"};
    size_t nAnalogs = analogNames.size();
    // Add markers to the new c3d
    for (size_t a = 0; a < nAnalogs; ++a)
        new_c3d.addAnalog(analogNames[a]);

    size_t nFrames = 10;
    size_t nSubframes = 5;
    float frameRate = 1000;
    ezc3d::ParametersNS::GroupNS::Parameter analogRate("RATE");
    analogRate.set(std::vector<float>()={frameRate}, {1});
    new_c3d.addParameter("ANALOG", analogRate);
    for (size_t f = 0; f < nFrames; ++f){
        ezc3d::DataNS::Frame frame;
        ezc3d::DataNS::AnalogsNS::Analogs analogs;
        for (size_t sf = 0; sf < nSubframes; ++sf){
            ezc3d::DataNS::AnalogsNS::SubFrame subframes;
            for (size_t c = 0; c < nAnalogs; ++c){
                ezc3d::DataNS::AnalogsNS::Channel channel;
                channel.name(analogNames[c]);
                channel.value(static_cast<float>(2*f+3*sf+4*c+1) / static_cast<float>(7.0)); // Generate random data
                subframes.addChannel(channel);
            }
            analogs.addSubframe(subframes);
        }
        frame.add(analogs);
        new_c3d.addFrame(frame);
    }


    // HEADER
    // Things that should remain as default
    defaultHeaderTest(new_c3d, HEADER_TYPE::POINT_AND_EVENT);

    // Things that should have change
    EXPECT_EQ(new_c3d.header().firstFrame(), 0);
    EXPECT_EQ(new_c3d.header().lastFrame(), nFrames - 1);
    EXPECT_EQ(new_c3d.header().nbFrames(), nFrames);
    EXPECT_EQ(new_c3d.header().nbAnalogsMeasurement(), nAnalogs * nSubframes);
    EXPECT_EQ(new_c3d.header().nbAnalogByFrame(), nSubframes);
    EXPECT_EQ(new_c3d.header().nbAnalogs(), nAnalogs);

    // PARAMETERS
    // Things that should remain as default
    defaultParametersTest(new_c3d, PARAMETER_TYPE::HEADER);
    defaultParametersTest(new_c3d, PARAMETER_TYPE::POINT);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("FRAMES").valuesAsInt()[0], nFrames);
    defaultParametersTest(new_c3d, PARAMETER_TYPE::FORCE_PLATFORM);

    // Things that should have change
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("USED").type(), ezc3d::INT);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("USED").valuesAsInt().size(), 1);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("USED").valuesAsInt()[0], nAnalogs);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("LABELS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("LABELS").valuesAsString().size(), nAnalogs);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("DESCRIPTIONS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("DESCRIPTIONS").valuesAsString().size(), nAnalogs);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("GEN_SCALE").type(), ezc3d::INT);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("GEN_SCALE").valuesAsInt().size(), 1);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("GEN_SCALE").valuesAsInt()[0], 1);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("SCALE").valuesAsFloat().size(), nAnalogs);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("OFFSET").type(), ezc3d::INT);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("OFFSET").valuesAsInt().size(), nAnalogs);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("UNITS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("UNITS").valuesAsString().size(), nAnalogs);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("RATE").type(), ezc3d::FLOAT);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("RATE").valuesAsFloat().size(), 1);
    EXPECT_FLOAT_EQ(new_c3d.parameters().group("ANALOG").parameter("RATE").valuesAsFloat()[0], frameRate);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("FORMAT").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("FORMAT").valuesAsString().size(), 0);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("BITS").type(), ezc3d::INT);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("BITS").valuesAsInt().size(), 0);

    for (size_t a = 0; a < nAnalogs; ++a){
        EXPECT_STREQ(new_c3d.parameters().group("ANALOG").parameter("LABELS").valuesAsString()[a].c_str(), analogNames[a].c_str());
        EXPECT_STREQ(new_c3d.parameters().group("ANALOG").parameter("DESCRIPTIONS").valuesAsString()[a].c_str(), "");
        EXPECT_FLOAT_EQ(new_c3d.parameters().group("ANALOG").parameter("SCALE").valuesAsFloat()[a], 1);
        EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("OFFSET").valuesAsInt()[a], 0);
        EXPECT_STREQ(new_c3d.parameters().group("ANALOG").parameter("UNITS").valuesAsString()[a].c_str(), "V");
    }

    // DATA
    for (size_t f = 0; f < nFrames; ++f)
        for (size_t sf = 0; sf < nSubframes; ++sf)
            for (size_t c = 0; c < nAnalogs; ++c)
                EXPECT_FLOAT_EQ(new_c3d.data().frame(static_cast<int>(f)).analogs().subframe(static_cast<int>(sf)).channel(static_cast<int>(c)).value(),
                                static_cast<float>(2*f+3*sf+4*c+1) / static_cast<float>(7.0));
}

TEST(c3dModifier, addPointsAndAnalogs){
    // Create an empty c3d
    ezc3d::c3d new_c3d;

    // Setup some variables
    std::vector<std::string> markerNames = {"marker1", "marker2", "marker3"};
    size_t nMarkers = markerNames.size();
    // Add markers to the new c3d
    for (size_t m = 0; m < nMarkers; ++m)
        new_c3d.addMarker(markerNames[m]);

    std::vector<std::string> analogNames = {"analog1", "analog2", "analog3"};
    size_t nAnalogs = analogNames.size();
    // Add markers to the new c3d
    for (size_t a = 0; a < nAnalogs; ++a)
        new_c3d.addAnalog(analogNames[a]);

    size_t nFrames = 10;
    float pointFrameRate = 100;
    float analogFrameRate = 1000;
    float nSubframes = analogFrameRate / pointFrameRate;
    ezc3d::ParametersNS::GroupNS::Parameter pointRate("RATE");
    pointRate.set(std::vector<float>()={pointFrameRate}, {1});
    new_c3d.addParameter("POINT", pointRate);
    ezc3d::ParametersNS::GroupNS::Parameter analogRate("RATE");
    analogRate.set(std::vector<float>()={analogFrameRate}, {1});
    new_c3d.addParameter("ANALOG", analogRate);
    for (size_t f = 0; f < nFrames; ++f){
        ezc3d::DataNS::Frame frame;

        ezc3d::DataNS::Points3dNS::Points pts;
        for (size_t m = 0; m < nMarkers; ++m){
            ezc3d::DataNS::Points3dNS::Point pt;
            pt.name(markerNames[m]);
            // Generate some random data
            pt.x(static_cast<float>(2*f+3*m+1) / static_cast<float>(7.0));
            pt.y(static_cast<float>(2*f+3*m+2) / static_cast<float>(7.0));
            pt.z(static_cast<float>(2*f+3*m+3) / static_cast<float>(7.0));
            pts.add(pt);
        }

        ezc3d::DataNS::AnalogsNS::Analogs analogs;
        for (size_t sf = 0; sf < nSubframes; ++sf){
            ezc3d::DataNS::AnalogsNS::SubFrame subframes;
            for (size_t c = 0; c < nAnalogs; ++c){
                ezc3d::DataNS::AnalogsNS::Channel channel;
                channel.name(analogNames[c]);
                channel.value(static_cast<float>(2*f+3*sf+4*c+1) / static_cast<float>(7.0)); // Generate random data
                subframes.addChannel(channel);
            }
            analogs.addSubframe(subframes);
        }

        frame.add(pts, analogs);
        new_c3d.addFrame(frame);
    }


    // HEADER
    // Things that should remain as default
    defaultHeaderTest(new_c3d, HEADER_TYPE::EVENT_ONLY);

    // Things that should have change
    EXPECT_EQ(new_c3d.header().nb3dPoints(), nMarkers);
    EXPECT_EQ(new_c3d.header().firstFrame(), 0);
    EXPECT_EQ(new_c3d.header().lastFrame(), nFrames - 1);
    EXPECT_EQ(new_c3d.header().nbMaxInterpGap(), 10);
    EXPECT_EQ(new_c3d.header().scaleFactor(), -1);
    EXPECT_FLOAT_EQ(new_c3d.header().frameRate(), pointFrameRate);
    EXPECT_EQ(new_c3d.header().nbFrames(), nFrames);
    EXPECT_EQ(new_c3d.header().nbAnalogsMeasurement(), nAnalogs * nSubframes);
    EXPECT_EQ(new_c3d.header().nbAnalogByFrame(), nSubframes);
    EXPECT_EQ(new_c3d.header().nbAnalogs(), nAnalogs);


    // PARAMETERS
    // Things that should remain as default
    defaultParametersTest(new_c3d, PARAMETER_TYPE::HEADER);
    defaultParametersTest(new_c3d, PARAMETER_TYPE::FORCE_PLATFORM);

    // Things that should have change
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("USED").type(), ezc3d::INT);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("USED").valuesAsInt().size(), 1);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("USED").valuesAsInt()[0], nMarkers);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("SCALE").valuesAsFloat().size(), 1);
    EXPECT_FLOAT_EQ(new_c3d.parameters().group("POINT").parameter("SCALE").valuesAsFloat()[0], -1);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("RATE").type(), ezc3d::FLOAT);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("RATE").valuesAsFloat().size(), 1);
    EXPECT_FLOAT_EQ(new_c3d.parameters().group("POINT").parameter("RATE").valuesAsFloat()[0], pointFrameRate);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("FRAMES").type(), ezc3d::INT);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("FRAMES").valuesAsInt().size(), 1);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("FRAMES").valuesAsInt()[0], nFrames);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("LABELS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("LABELS").valuesAsString().size(), nMarkers);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("DESCRIPTIONS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("DESCRIPTIONS").valuesAsString().size(), nMarkers);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("UNITS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("UNITS").valuesAsString().size(), nMarkers);
    for (size_t m = 0; m < nMarkers; ++m){
        EXPECT_STREQ(new_c3d.parameters().group("POINT").parameter("LABELS").valuesAsString()[m].c_str(), markerNames[m].c_str());
        EXPECT_STREQ(new_c3d.parameters().group("POINT").parameter("DESCRIPTIONS").valuesAsString()[m].c_str(), "");
        EXPECT_STREQ(new_c3d.parameters().group("POINT").parameter("UNITS").valuesAsString()[m].c_str(), "mm");
    }

    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("USED").type(), ezc3d::INT);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("USED").valuesAsInt().size(), 1);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("USED").valuesAsInt()[0], nAnalogs);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("LABELS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("LABELS").valuesAsString().size(), nAnalogs);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("DESCRIPTIONS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("DESCRIPTIONS").valuesAsString().size(), nAnalogs);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("GEN_SCALE").type(), ezc3d::INT);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("GEN_SCALE").valuesAsInt().size(), 1);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("GEN_SCALE").valuesAsInt()[0], 1);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("SCALE").valuesAsFloat().size(), nAnalogs);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("OFFSET").type(), ezc3d::INT);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("OFFSET").valuesAsInt().size(), nAnalogs);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("UNITS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("UNITS").valuesAsString().size(), nAnalogs);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("RATE").type(), ezc3d::FLOAT);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("RATE").valuesAsFloat().size(), 1);
    EXPECT_FLOAT_EQ(new_c3d.parameters().group("ANALOG").parameter("RATE").valuesAsFloat()[0], analogFrameRate);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("FORMAT").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("FORMAT").valuesAsString().size(), 0);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("BITS").type(), ezc3d::INT);
    EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("BITS").valuesAsInt().size(), 0);

    for (size_t a = 0; a < nAnalogs; ++a){
        EXPECT_STREQ(new_c3d.parameters().group("ANALOG").parameter("LABELS").valuesAsString()[a].c_str(), analogNames[a].c_str());
        EXPECT_STREQ(new_c3d.parameters().group("ANALOG").parameter("DESCRIPTIONS").valuesAsString()[a].c_str(), "");
        EXPECT_FLOAT_EQ(new_c3d.parameters().group("ANALOG").parameter("SCALE").valuesAsFloat()[a], 1);
        EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("OFFSET").valuesAsInt()[a], 0);
        EXPECT_STREQ(new_c3d.parameters().group("ANALOG").parameter("UNITS").valuesAsString()[a].c_str(), "V");
    }


    // DATA
    for (size_t f = 0; f < nFrames; ++f){
        for (size_t m = 0; m < nMarkers; ++m){
            EXPECT_FLOAT_EQ(new_c3d.data().frame(static_cast<int>(f)).points().point(markerNames[m]).x(), static_cast<float>(2*f+3*m+1) / static_cast<float>(7.0));
            EXPECT_FLOAT_EQ(new_c3d.data().frame(static_cast<int>(f)).points().point(markerNames[m]).y(), static_cast<float>(2*f+3*m+2) / static_cast<float>(7.0));
            EXPECT_FLOAT_EQ(new_c3d.data().frame(static_cast<int>(f)).points().point(markerNames[m]).z(), static_cast<float>(2*f+3*m+3) / static_cast<float>(7.0));
        }

        for (size_t sf = 0; sf < nSubframes; ++sf)
            for (size_t c = 0; c < nAnalogs; ++c)
                EXPECT_FLOAT_EQ(new_c3d.data().frame(static_cast<int>(f)).analogs().subframe(static_cast<int>(sf)).channel(static_cast<int>(c)).value(),
                                static_cast<float>(2*f+3*sf+4*c+1) / static_cast<float>(7.0));

    }

}

TEST(c3dFileIO, CreateWriteAndReadBack){
    // Create an empty c3d fill it with data and reopen
    ezc3d::c3d new_c3d;

    // Setup some variables
    std::vector<std::string> markerNames = {"marker1", "marker2", "marker3"};
    size_t nMarkers = markerNames.size();
    // Add markers to the new c3d
    for (size_t m = 0; m < nMarkers; ++m)
        new_c3d.addMarker(markerNames[m]);

    std::vector<std::string> analogNames = {"analog1", "analog2", "analog3"};
    size_t nAnalogs = analogNames.size();
    // Add markers to the new c3d
    for (size_t a = 0; a < nAnalogs; ++a)
        new_c3d.addAnalog(analogNames[a]);

    size_t nFrames = 10;
    float pointFrameRate = 100;
    float analogFrameRate = 1000;
    float nSubframes = analogFrameRate / pointFrameRate;
    ezc3d::ParametersNS::GroupNS::Parameter pointRate("RATE");
    pointRate.set(std::vector<float>()={pointFrameRate}, {1});
    new_c3d.addParameter("POINT", pointRate);
    ezc3d::ParametersNS::GroupNS::Parameter analogRate("RATE");
    analogRate.set(std::vector<float>()={analogFrameRate}, {1});
    new_c3d.addParameter("ANALOG", analogRate);
    for (size_t f = 0; f < nFrames; ++f){
        ezc3d::DataNS::Frame frame;

        ezc3d::DataNS::Points3dNS::Points pts;
        for (size_t m = 0; m < nMarkers; ++m){
            ezc3d::DataNS::Points3dNS::Point pt;
            pt.name(markerNames[m]);
            // Generate some random data
            pt.x(static_cast<float>(2*f+3*m+1) / static_cast<float>(7.0));
            pt.y(static_cast<float>(2*f+3*m+2) / static_cast<float>(7.0));
            pt.z(static_cast<float>(2*f+3*m+3) / static_cast<float>(7.0));
            pts.add(pt);
        }

        ezc3d::DataNS::AnalogsNS::Analogs analogs;
        for (size_t sf = 0; sf < nSubframes; ++sf){
            ezc3d::DataNS::AnalogsNS::SubFrame subframes;
            for (size_t c = 0; c < nAnalogs; ++c){
                ezc3d::DataNS::AnalogsNS::Channel channel;
                channel.name(analogNames[c]);
                channel.value(static_cast<float>(2*f+3*sf+4*c+1) / static_cast<float>(7.0)); // Generate random data
                subframes.addChannel(channel);
            }
            analogs.addSubframe(subframes);
        }

        frame.add(pts, analogs);
        new_c3d.addFrame(frame);
    }

    // Write the c3d on the disk
    std::string savePath("temporary.c3d");
    new_c3d.write(savePath.c_str());

    // Open it back and delete it
    ezc3d::c3d read_c3d(savePath.c_str());
    remove(savePath.c_str());

    // Test the read file
    // HEADER
    // Things that should remain as default
    defaultHeaderTest(read_c3d, HEADER_TYPE::EVENT_ONLY);

    // Things that should have change
    EXPECT_EQ(read_c3d.header().nb3dPoints(), nMarkers);
    EXPECT_EQ(read_c3d.header().firstFrame(), 0);
    EXPECT_EQ(read_c3d.header().lastFrame(), nFrames - 1);
    EXPECT_EQ(read_c3d.header().nbMaxInterpGap(), 10);
    EXPECT_EQ(read_c3d.header().scaleFactor(), -1);
    EXPECT_FLOAT_EQ(read_c3d.header().frameRate(), pointFrameRate);
    EXPECT_EQ(read_c3d.header().nbFrames(), nFrames);
    EXPECT_EQ(read_c3d.header().nbAnalogsMeasurement(), nAnalogs * nSubframes);
    EXPECT_EQ(read_c3d.header().nbAnalogByFrame(), nSubframes);
    EXPECT_EQ(read_c3d.header().nbAnalogs(), nAnalogs);


    // PARAMETERS
    // Things that should remain as default
    defaultParametersTest(read_c3d, PARAMETER_TYPE::HEADER);
    defaultParametersTest(read_c3d, PARAMETER_TYPE::FORCE_PLATFORM);

    // Things that should have change
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("USED").type(), ezc3d::INT);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("USED").valuesAsInt().size(), 1);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("USED").valuesAsInt()[0], nMarkers);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("SCALE").valuesAsFloat().size(), 1);
    EXPECT_FLOAT_EQ(read_c3d.parameters().group("POINT").parameter("SCALE").valuesAsFloat()[0], -1);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("RATE").type(), ezc3d::FLOAT);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("RATE").valuesAsFloat().size(), 1);
    EXPECT_FLOAT_EQ(read_c3d.parameters().group("POINT").parameter("RATE").valuesAsFloat()[0], pointFrameRate);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("FRAMES").type(), ezc3d::INT);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("FRAMES").valuesAsInt().size(), 1);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("FRAMES").valuesAsInt()[0], nFrames);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("LABELS").type(), ezc3d::CHAR);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("LABELS").valuesAsString().size(), nMarkers);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("DESCRIPTIONS").type(), ezc3d::CHAR);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("DESCRIPTIONS").valuesAsString().size(), nMarkers);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("UNITS").type(), ezc3d::CHAR);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("UNITS").valuesAsString().size(), nMarkers);
    for (size_t m = 0; m < nMarkers; ++m){
        EXPECT_STREQ(read_c3d.parameters().group("POINT").parameter("LABELS").valuesAsString()[m].c_str(), markerNames[m].c_str());
        EXPECT_STREQ(read_c3d.parameters().group("POINT").parameter("DESCRIPTIONS").valuesAsString()[m].c_str(), "");
        EXPECT_STREQ(read_c3d.parameters().group("POINT").parameter("UNITS").valuesAsString()[m].c_str(), "mm");
    }

    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("USED").type(), ezc3d::INT);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("USED").valuesAsInt().size(), 1);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("USED").valuesAsInt()[0], nAnalogs);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("LABELS").type(), ezc3d::CHAR);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("LABELS").valuesAsString().size(), nAnalogs);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("DESCRIPTIONS").type(), ezc3d::CHAR);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("DESCRIPTIONS").valuesAsString().size(), nAnalogs);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("GEN_SCALE").type(), ezc3d::INT);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("GEN_SCALE").valuesAsInt().size(), 1);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("GEN_SCALE").valuesAsInt()[0], 1);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("SCALE").valuesAsFloat().size(), nAnalogs);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("OFFSET").type(), ezc3d::INT);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("OFFSET").valuesAsInt().size(), nAnalogs);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("UNITS").type(), ezc3d::CHAR);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("UNITS").valuesAsString().size(), nAnalogs);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("RATE").type(), ezc3d::FLOAT);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("RATE").valuesAsFloat().size(), 1);
    EXPECT_FLOAT_EQ(read_c3d.parameters().group("ANALOG").parameter("RATE").valuesAsFloat()[0], analogFrameRate);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("FORMAT").type(), ezc3d::CHAR);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("FORMAT").valuesAsString().size(), 0);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("BITS").type(), ezc3d::INT);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("BITS").valuesAsInt().size(), 0);

    for (size_t a = 0; a < nAnalogs; ++a){
        EXPECT_STREQ(read_c3d.parameters().group("ANALOG").parameter("LABELS").valuesAsString()[a].c_str(), analogNames[a].c_str());
        EXPECT_STREQ(read_c3d.parameters().group("ANALOG").parameter("DESCRIPTIONS").valuesAsString()[a].c_str(), "");
        EXPECT_FLOAT_EQ(read_c3d.parameters().group("ANALOG").parameter("SCALE").valuesAsFloat()[a], 1);
        EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("OFFSET").valuesAsInt()[a], 0);
        EXPECT_STREQ(read_c3d.parameters().group("ANALOG").parameter("UNITS").valuesAsString()[a].c_str(), "V");
    }


    // DATA
    for (size_t f = 0; f < nFrames; ++f){
        for (size_t m = 0; m < nMarkers; ++m){
            EXPECT_FLOAT_EQ(read_c3d.data().frame(static_cast<int>(f)).points().point(markerNames[m]).x(), static_cast<float>(2*f+3*m+1) / static_cast<float>(7.0));
            EXPECT_FLOAT_EQ(read_c3d.data().frame(static_cast<int>(f)).points().point(markerNames[m]).y(), static_cast<float>(2*f+3*m+2) / static_cast<float>(7.0));
            EXPECT_FLOAT_EQ(read_c3d.data().frame(static_cast<int>(f)).points().point(markerNames[m]).z(), static_cast<float>(2*f+3*m+3) / static_cast<float>(7.0));
        }

        for (size_t sf = 0; sf < nSubframes; ++sf)
            for (size_t c = 0; c < nAnalogs; ++c)
                EXPECT_FLOAT_EQ(read_c3d.data().frame(static_cast<int>(f)).analogs().subframe(static_cast<int>(sf)).channel(static_cast<int>(c)).value(),
                                static_cast<float>(2*f+3*sf+4*c+1) / static_cast<float>(7.0));

    }
}



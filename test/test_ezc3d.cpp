#include <iostream>
#include <gtest/gtest.h>

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

struct c3dTestStruct{
    // Create an empty c3d
    ezc3d::c3d c3d;

    size_t nFrames = SIZE_MAX;
    float pointFrameRate = -1;
    size_t nMarkers = SIZE_MAX;
    std::vector<std::string> markerNames;

    float analogFrameRate = -1;
    float nSubframes = -1;
    size_t nAnalogs = SIZE_MAX;
    std::vector<std::string> analogNames;
};

void fillC3D(c3dTestStruct& c3dStruc, bool withMarkers, bool withAnalogs){
    // Setup some variables
    if (withMarkers){
        c3dStruc.markerNames = {"marker1", "marker2", "marker3"};
        c3dStruc.nMarkers = c3dStruc.markerNames.size();
        // Add markers to the new c3d
        for (size_t m = 0; m < c3dStruc.nMarkers; ++m)
            c3dStruc.c3d.addPoint(c3dStruc.markerNames[m]);
    }

    c3dStruc.analogNames.clear();
    c3dStruc.nAnalogs = SIZE_MAX;
    if (withAnalogs){
        c3dStruc.analogNames = {"analog1", "analog2", "analog3"};
        c3dStruc.nAnalogs = c3dStruc.analogNames.size();
        // Add markers to the new c3d
        for (size_t a = 0; a < c3dStruc.nAnalogs; ++a)
            c3dStruc.c3d.addAnalog(c3dStruc.analogNames[a]);
    }

    c3dStruc.nFrames = 10;
    c3dStruc.pointFrameRate = 100;
    if (withMarkers){
        ezc3d::ParametersNS::GroupNS::Parameter pointRate("RATE");
        pointRate.set(std::vector<float>()={c3dStruc.pointFrameRate}, {1});
        c3dStruc.c3d.addParameter("POINT", pointRate);
    }
    if (withAnalogs){
        c3dStruc.analogFrameRate = 1000;
        c3dStruc.nSubframes = c3dStruc.analogFrameRate / c3dStruc.pointFrameRate;

        ezc3d::ParametersNS::GroupNS::Parameter analogRate("RATE");
        analogRate.set(std::vector<float>()={c3dStruc.analogFrameRate}, {1});
        c3dStruc.c3d.addParameter("ANALOG", analogRate);
    }
    for (size_t f = 0; f < c3dStruc.nFrames; ++f){
        ezc3d::DataNS::Frame frame;

        ezc3d::DataNS::Points3dNS::Points pts;
        if (withMarkers){
            for (size_t m = 0; m < c3dStruc.nMarkers; ++m){
                ezc3d::DataNS::Points3dNS::Point pt;
                pt.name(c3dStruc.markerNames[m]);
                // Generate some random data
                pt.x(static_cast<float>(2*f+3*m+1) / static_cast<float>(7.0));
                pt.y(static_cast<float>(2*f+3*m+2) / static_cast<float>(7.0));
                pt.z(static_cast<float>(2*f+3*m+3) / static_cast<float>(7.0));
                pts.add(pt);
            }
        }

        ezc3d::DataNS::AnalogsNS::Analogs analogs;
        if (withAnalogs){
            for (size_t sf = 0; sf < c3dStruc.nSubframes; ++sf){
                ezc3d::DataNS::AnalogsNS::SubFrame subframes;
                for (size_t c = 0; c < c3dStruc.nAnalogs; ++c){
                    ezc3d::DataNS::AnalogsNS::Channel channel;
                    channel.name(c3dStruc.analogNames[c]);
                    channel.value(static_cast<float>(2*f+3*sf+4*c+1) / static_cast<float>(7.0)); // Generate random data
                    subframes.addChannel(channel);
                }
                analogs.addSubframe(subframes);
            }
        }

        if (withMarkers && withAnalogs)
            frame.add(pts, analogs);
        else if (withMarkers)
            frame.add(pts);
        else if (withAnalogs)
            frame.add(analogs);
        c3dStruc.c3d.addFrame(frame);
    }
}

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
        EXPECT_THROW(new_c3d.header().eventsTime(-1), std::invalid_argument);
        for (int e = 0; e < static_cast<int>(new_c3d.header().eventsTime().size()); ++e)
            EXPECT_FLOAT_EQ(new_c3d.header().eventsTime(e), 0);
        EXPECT_THROW(new_c3d.header().eventsTime(static_cast<int>(new_c3d.header().eventsTime().size())), std::invalid_argument);

        EXPECT_EQ(new_c3d.header().eventsLabel().size(), 18);
        EXPECT_THROW(new_c3d.header().eventsLabel(-1), std::invalid_argument);
        for (int e = 0; e < static_cast<int>(new_c3d.header().eventsLabel().size()); ++e)
            EXPECT_STREQ(new_c3d.header().eventsLabel(e).c_str(), "");
        EXPECT_THROW(new_c3d.header().eventsLabel(static_cast<int>(new_c3d.header().eventsLabel().size())), std::invalid_argument);

        EXPECT_EQ(new_c3d.header().eventsDisplay().size(), 9);
        EXPECT_THROW(new_c3d.header().eventsDisplay(-1), std::invalid_argument);
        for (int e = 0; e < static_cast<int>(new_c3d.header().eventsDisplay().size()); ++e)
            EXPECT_EQ(new_c3d.header().eventsDisplay(e), 0);
        EXPECT_THROW(new_c3d.header().eventsDisplay(static_cast<int>(new_c3d.header().eventsDisplay().size())), std::invalid_argument);

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

TEST(c3dShow, printIt){
    // Create an empty c3d and print it
    c3dTestStruct new_c3d;
    fillC3D(new_c3d, true, true);

    EXPECT_NO_THROW(new_c3d.c3d.print());
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


TEST(initialize, noC3D){
    EXPECT_THROW(ezc3d::c3d("ThereIsNoC3dThere.c3d"), std::ios_base::failure);
}


TEST(wrongC3D, wrongChecksumHeader){
    // Create an empty c3d
    ezc3d::c3d new_c3d;
    std::string savePath("temporary.c3d");
    new_c3d.write(savePath.c_str());

    // Modify the header checksum byte
    std::ofstream c3d_file(savePath.c_str(), std::ofstream::in);
    c3d_file.seekp(ezc3d::BYTE);
    int checksum(0x0);
    c3d_file.write(reinterpret_cast<const char*>(&checksum), ezc3d::BYTE);
    c3d_file.close();

    // Read the erroneous file
    EXPECT_THROW(ezc3d::c3d new_c3d("temporary.c3d"), std::ios_base::failure);
    remove(savePath.c_str());
}


TEST(wrongC3D, wrongChecksumParameter){
    // Create an empty c3d
    ezc3d::c3d new_c3d;
    std::string savePath("temporary.c3d");
    new_c3d.write(savePath.c_str());

    // Modify the header checksum byte
    std::ofstream c3d_file(savePath.c_str(), std::ofstream::in);
    c3d_file.seekp(256*ezc3d::DATA_TYPE::WORD*(new_c3d.header().parametersAddress()-1) + 1); // move to the parameter checksum
    int checksum(0x0);
    c3d_file.write(reinterpret_cast<const char*>(&checksum), ezc3d::BYTE);
    c3d_file.close();

    // Read the erroneous file
    EXPECT_THROW(ezc3d::c3d new_c3d("temporary.c3d"), std::ios_base::failure);

    // If a 0 is also on the byte before the checksum, this is a Qualisys C3D and should be read even if checksum si wrong
    std::ofstream c3d_file2(savePath.c_str(), std::ofstream::in);
    c3d_file2.seekp(256*ezc3d::DATA_TYPE::WORD*(new_c3d.header().parametersAddress()-1)); // move to the parameter checksum
    int parameterStart(0x0);
    c3d_file2.write(reinterpret_cast<const char*>(&parameterStart), ezc3d::BYTE);
    c3d_file2.close();

    // Read the Qualisys file
    EXPECT_NO_THROW(ezc3d::c3d new_c3d("temporary.c3d"));

    // Delete the file
    remove(savePath.c_str());
}

TEST(wrongC3D, wrongNextparamParameter){
    // Create an empty c3d
    ezc3d::c3d new_c3d;
    std::string savePath("temporary.c3d");
    new_c3d.write(savePath.c_str());

    // If a 0 is also on the byte before the checksum, this is a Qualisys C3D and should be read even if checksum si wrong
    std::ofstream c3d_file(savePath.c_str(), std::ofstream::in);
    c3d_file.seekp(256*ezc3d::DATA_TYPE::WORD*(new_c3d.header().parametersAddress()-1)); // move to the parameter checksum
    int parameterStart(0x0);
    c3d_file.write(reinterpret_cast<const char*>(&parameterStart), ezc3d::BYTE);
    c3d_file.close();

    // Read the erroneous file
    EXPECT_THROW(ezc3d::c3d new_c3d("temporary.c3d"), std::ios_base::failure);

    // Delete the file
    remove(savePath.c_str());
}


TEST(c3dModifier, specificParameters){
    // Create an empty c3d
    c3dTestStruct new_c3d;
    fillC3D(new_c3d, true, true);

    // Test update parameter
    EXPECT_THROW(new_c3d.c3d.updateParameters({"WrongPoint"}, {}), std::runtime_error);
    EXPECT_THROW(new_c3d.c3d.updateParameters({}, {"WrongAnalog"}), std::runtime_error);

    // Get an erroneous group
    EXPECT_THROW(new_c3d.c3d.parameters().group("ThisIsNotARealGroup"), std::invalid_argument);

    // Lock and unlock a group
    EXPECT_THROW(new_c3d.c3d.lockGroup("ThisIsNotARealGroup"), std::invalid_argument);
    EXPECT_THROW(new_c3d.c3d.unlockGroup("ThisIsNotARealGroup"), std::invalid_argument);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").isLocked(), false);
    EXPECT_NO_THROW(new_c3d.c3d.lockGroup("POINT"));
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").isLocked(), true);
    EXPECT_NO_THROW(new_c3d.c3d.unlockGroup("POINT"));
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").isLocked(), false);

    // Add an erroneous parameter to a group
    ezc3d::ParametersNS::GroupNS::Parameter p;
    EXPECT_THROW(new_c3d.c3d.addParameter("POINT", p), std::invalid_argument);
    p.name("EmptyParam");
    EXPECT_THROW(new_c3d.c3d.addParameter("POINT", p), std::runtime_error);

    // Create a new group
    ezc3d::ParametersNS::GroupNS::Parameter p2;
    p2.name("NewParam");
    p2.set(std::vector<int>(), {0});
    EXPECT_NO_THROW(new_c3d.c3d.addParameter("ThisIsANewRealGroup", p2));
    EXPECT_EQ(new_c3d.c3d.parameters().group("ThisIsANewRealGroup").parameter("NewParam").type(), ezc3d::INT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ThisIsANewRealGroup").parameter("NewParam").valuesAsInt().size(), 0);

    // Get an out of range parameter
    EXPECT_THROW(new_c3d.c3d.parameters().group("POINT").parameter(-1), std::out_of_range);
    size_t nPointParams(new_c3d.c3d.parameters().group("POINT").parameters().size());
    EXPECT_THROW(new_c3d.c3d.parameters().group("POINT").parameter(static_cast<int>(nPointParams)), std::out_of_range);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameterIdx("ThisIsNotARealParameter"), -1);

    // Try to read a parameter into the wrong format
    EXPECT_THROW(p.valuesAsByte(), std::invalid_argument);
    EXPECT_THROW(p.valuesAsInt(), std::invalid_argument);
    EXPECT_THROW(p.valuesAsFloat(), std::invalid_argument);
    EXPECT_THROW(p.valuesAsString(), std::invalid_argument);

    // Lock and unlock a parameter
    EXPECT_EQ(p.isLocked(), false);
    p.lock();
    EXPECT_EQ(p.isLocked(), true);
    p.unlock();
    EXPECT_EQ(p.isLocked(), false);
}


TEST(c3dModifier, addPoints) {
    // Create an empty c3d
    c3dTestStruct new_c3d;
    fillC3D(new_c3d, true, false);

    // HEADER
    // Things that should remain as default
    defaultHeaderTest(new_c3d.c3d, HEADER_TYPE::ANALOG_AND_EVENT);

    // Things that should have change
    EXPECT_EQ(new_c3d.c3d.header().nb3dPoints(), new_c3d.nMarkers);
    EXPECT_EQ(new_c3d.c3d.header().firstFrame(), 0);
    EXPECT_EQ(new_c3d.c3d.header().lastFrame(), new_c3d.nFrames - 1);
    EXPECT_EQ(new_c3d.c3d.header().nbMaxInterpGap(), 10);
    EXPECT_EQ(new_c3d.c3d.header().scaleFactor(), -1);
    EXPECT_FLOAT_EQ(new_c3d.c3d.header().frameRate(), new_c3d.pointFrameRate);
    EXPECT_EQ(new_c3d.c3d.header().nbFrames(), new_c3d.nFrames);

    // PARAMETERS
    // Things that should remain as default
    defaultParametersTest(new_c3d.c3d, PARAMETER_TYPE::HEADER);
    defaultParametersTest(new_c3d.c3d, PARAMETER_TYPE::ANALOG);
    defaultParametersTest(new_c3d.c3d, PARAMETER_TYPE::FORCE_PLATFORM);

    // Things that should have change
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("USED").type(), ezc3d::INT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("USED").valuesAsInt().size(), 1);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("USED").valuesAsInt()[0], new_c3d.nMarkers);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("SCALE").valuesAsFloat().size(), 1);
    EXPECT_FLOAT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("SCALE").valuesAsFloat()[0], -1);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("RATE").type(), ezc3d::FLOAT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("RATE").valuesAsFloat().size(), 1);
    EXPECT_FLOAT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("RATE").valuesAsFloat()[0], new_c3d.pointFrameRate);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("FRAMES").type(), ezc3d::INT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("FRAMES").valuesAsInt().size(), 1);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("FRAMES").valuesAsInt()[0], new_c3d.nFrames);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("LABELS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("LABELS").valuesAsString().size(), new_c3d.nMarkers);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("DESCRIPTIONS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("DESCRIPTIONS").valuesAsString().size(), new_c3d.nMarkers);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("UNITS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("UNITS").valuesAsString().size(), new_c3d.nMarkers);
    for (size_t m = 0; m < new_c3d.nMarkers; ++m){
        EXPECT_STREQ(new_c3d.c3d.parameters().group("POINT").parameter("LABELS").valuesAsString()[m].c_str(),new_c3d.markerNames[m].c_str());
        EXPECT_STREQ(new_c3d.c3d.parameters().group("POINT").parameter("DESCRIPTIONS").valuesAsString()[m].c_str(), "");
        EXPECT_STREQ(new_c3d.c3d.parameters().group("POINT").parameter("UNITS").valuesAsString()[m].c_str(), "mm");
    }

    // DATA
    for (size_t f = 0; f < new_c3d.nFrames; ++f){
        for (size_t m = 0; m < new_c3d.nMarkers; ++m){
            EXPECT_FLOAT_EQ(new_c3d.c3d.data().frame(static_cast<int>(f)).points().point(new_c3d.markerNames[m]).x(), static_cast<float>(2*f+3*m+1) / static_cast<float>(7.0));
            EXPECT_FLOAT_EQ(new_c3d.c3d.data().frame(static_cast<int>(f)).points().point(new_c3d.markerNames[m]).y(), static_cast<float>(2*f+3*m+2) / static_cast<float>(7.0));
            EXPECT_FLOAT_EQ(new_c3d.c3d.data().frame(static_cast<int>(f)).points().point(new_c3d.markerNames[m]).z(), static_cast<float>(2*f+3*m+3) / static_cast<float>(7.0));
            EXPECT_FLOAT_EQ(new_c3d.c3d.data().frame(static_cast<int>(f)).points().point(new_c3d.markerNames[m]).residual(), 0);

            std::vector<float> data(new_c3d.c3d.data().frame(static_cast<int>(f)).points().point(new_c3d.markerNames[m]).data());
            EXPECT_FLOAT_EQ(new_c3d.c3d.data().frame(static_cast<int>(f)).points().point(new_c3d.markerNames[m]).x(), data[0]);
            EXPECT_FLOAT_EQ(new_c3d.c3d.data().frame(static_cast<int>(f)).points().point(new_c3d.markerNames[m]).y(), data[1]);
            EXPECT_FLOAT_EQ(new_c3d.c3d.data().frame(static_cast<int>(f)).points().point(new_c3d.markerNames[m]).z(), data[2]);
            EXPECT_FLOAT_EQ(new_c3d.c3d.data().frame(static_cast<int>(f)).points().point(new_c3d.markerNames[m]).residual(), 0);
        }
    }
}


TEST(c3dModifier, specificPoint){
    // Create an empty c3d
    c3dTestStruct new_c3d;
    fillC3D(new_c3d, true, false);

    // test replacing points
    for (size_t f = 0; f < new_c3d.nFrames; ++f){
        ezc3d::DataNS::Frame frame;

        ezc3d::DataNS::Points3dNS::Points pts(new_c3d.c3d.data().frame(static_cast<int>(f)).points());
        for (size_t m = 0; m < new_c3d.nMarkers; ++m){
            ezc3d::DataNS::Points3dNS::Point pt;
            pt.name(new_c3d.markerNames[m]);
            // Generate some random data
            pt.x(static_cast<float>(4*f+7*m+5) / static_cast<float>(13.0));
            pt.y(static_cast<float>(4*f+7*m+6) / static_cast<float>(13.0));
            pt.z(static_cast<float>(4*f+7*m+7) / static_cast<float>(13.0));
            pts.replace(static_cast<int>(m), pt);
        }
        frame.add(pts);
        new_c3d.c3d.addFrame(frame, static_cast<int>(f));
    }

    // Things that should have change
    EXPECT_EQ(new_c3d.c3d.header().nb3dPoints(), new_c3d.nMarkers);
    EXPECT_EQ(new_c3d.c3d.header().firstFrame(), 0);
    EXPECT_EQ(new_c3d.c3d.header().lastFrame(), new_c3d.nFrames - 1);
    EXPECT_EQ(new_c3d.c3d.header().nbMaxInterpGap(), 10);
    EXPECT_EQ(new_c3d.c3d.header().scaleFactor(), -1);
    EXPECT_FLOAT_EQ(new_c3d.c3d.header().frameRate(), new_c3d.pointFrameRate);
    EXPECT_EQ(new_c3d.c3d.header().nbFrames(), new_c3d.nFrames);

    // PARAMETERS
    // Things that should remain as default
    defaultParametersTest(new_c3d.c3d, PARAMETER_TYPE::HEADER);
    defaultParametersTest(new_c3d.c3d, PARAMETER_TYPE::ANALOG);
    defaultParametersTest(new_c3d.c3d, PARAMETER_TYPE::FORCE_PLATFORM);

    // Things that should have change
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("USED").type(), ezc3d::INT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("USED").valuesAsInt().size(), 1);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("USED").valuesAsInt()[0], new_c3d.nMarkers);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("SCALE").valuesAsFloat().size(), 1);
    EXPECT_FLOAT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("SCALE").valuesAsFloat()[0], -1);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("RATE").type(), ezc3d::FLOAT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("RATE").valuesAsFloat().size(), 1);
    EXPECT_FLOAT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("RATE").valuesAsFloat()[0], new_c3d.pointFrameRate);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("FRAMES").type(), ezc3d::INT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("FRAMES").valuesAsInt().size(), 1);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("FRAMES").valuesAsInt()[0], new_c3d.nFrames);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("LABELS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("LABELS").valuesAsString().size(), new_c3d.nMarkers);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("DESCRIPTIONS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("DESCRIPTIONS").valuesAsString().size(), new_c3d.nMarkers);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("UNITS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("UNITS").valuesAsString().size(), new_c3d.nMarkers);
    for (size_t m = 0; m < new_c3d.nMarkers; ++m){
        EXPECT_STREQ(new_c3d.c3d.parameters().group("POINT").parameter("LABELS").valuesAsString()[m].c_str(),new_c3d.markerNames[m].c_str());
        EXPECT_STREQ(new_c3d.c3d.parameters().group("POINT").parameter("DESCRIPTIONS").valuesAsString()[m].c_str(), "");
        EXPECT_STREQ(new_c3d.c3d.parameters().group("POINT").parameter("UNITS").valuesAsString()[m].c_str(), "mm");
    }

    // DATA
    for (size_t f = 0; f < new_c3d.nFrames; ++f){
        for (size_t m = 0; m < new_c3d.nMarkers; ++m){
            EXPECT_FLOAT_EQ(new_c3d.c3d.data().frame(static_cast<int>(f)).points().point(new_c3d.markerNames[m]).x(), static_cast<float>(4*f+7*m+5) / static_cast<float>(13.0));
            EXPECT_FLOAT_EQ(new_c3d.c3d.data().frame(static_cast<int>(f)).points().point(new_c3d.markerNames[m]).y(), static_cast<float>(4*f+7*m+6) / static_cast<float>(13.0));
            EXPECT_FLOAT_EQ(new_c3d.c3d.data().frame(static_cast<int>(f)).points().point(new_c3d.markerNames[m]).z(), static_cast<float>(4*f+7*m+7) / static_cast<float>(13.0));
            EXPECT_FLOAT_EQ(new_c3d.c3d.data().frame(static_cast<int>(f)).points().point(new_c3d.markerNames[m]).residual(), 0);
        }
    }

    // Access a non-existant point
    EXPECT_THROW(new_c3d.c3d.data().frame(0).points().point(-1), std::out_of_range);
    EXPECT_THROW(new_c3d.c3d.data().frame(0).points().point(static_cast<int>(new_c3d.nMarkers)), std::out_of_range);

    // Test for removing space at the end of a label
    new_c3d.c3d.addPoint("PointNameWithSpaceAtTheEnd ");
    new_c3d.nMarkers += 1;
    new_c3d.markerNames.push_back("PointNameWithSpaceAtTheEnd");
    EXPECT_STREQ(new_c3d.c3d.parameters().group("POINT").parameter("LABELS").valuesAsString()[new_c3d.nMarkers - 1].c_str(), "PointNameWithSpaceAtTheEnd");

}


TEST(c3dModifier, addAnalogs) {
    // Create an empty c3d
    c3dTestStruct new_c3d;
    fillC3D(new_c3d, false, true);

    // HEADER
    // Things that should remain as default
    defaultHeaderTest(new_c3d.c3d, HEADER_TYPE::POINT_AND_EVENT);

    // Things that should have change
    EXPECT_EQ(new_c3d.c3d.header().firstFrame(), 0);
    EXPECT_EQ(new_c3d.c3d.header().lastFrame(), new_c3d.nFrames - 1);
    EXPECT_EQ(new_c3d.c3d.header().nbFrames(), new_c3d.nFrames);
    EXPECT_EQ(new_c3d.c3d.header().nbAnalogsMeasurement(), new_c3d.nAnalogs * new_c3d.nSubframes);
    EXPECT_EQ(new_c3d.c3d.header().nbAnalogByFrame(), new_c3d.nSubframes);
    EXPECT_EQ(new_c3d.c3d.header().nbAnalogs(), new_c3d.nAnalogs);

    // PARAMETERS
    // Things that should remain as default
    defaultParametersTest(new_c3d.c3d, PARAMETER_TYPE::HEADER);
    defaultParametersTest(new_c3d.c3d, PARAMETER_TYPE::POINT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("FRAMES").valuesAsInt()[0], new_c3d.nFrames);
    defaultParametersTest(new_c3d.c3d, PARAMETER_TYPE::FORCE_PLATFORM);

    // Things that should have change
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("USED").type(), ezc3d::INT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("USED").valuesAsInt().size(), 1);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("USED").valuesAsInt()[0], new_c3d.nAnalogs);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("LABELS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("LABELS").valuesAsString().size(), new_c3d.nAnalogs);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("DESCRIPTIONS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("DESCRIPTIONS").valuesAsString().size(), new_c3d.nAnalogs);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("GEN_SCALE").type(), ezc3d::INT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("GEN_SCALE").valuesAsInt().size(), 1);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("GEN_SCALE").valuesAsInt()[0], 1);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("SCALE").valuesAsFloat().size(), new_c3d.nAnalogs);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("OFFSET").type(), ezc3d::INT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("OFFSET").valuesAsInt().size(), new_c3d.nAnalogs);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("UNITS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("UNITS").valuesAsString().size(), new_c3d.nAnalogs);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("RATE").type(), ezc3d::FLOAT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("RATE").valuesAsFloat().size(), 1);
    EXPECT_FLOAT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("RATE").valuesAsFloat()[0], new_c3d.analogFrameRate);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("FORMAT").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("FORMAT").valuesAsString().size(), 0);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("BITS").type(), ezc3d::INT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("BITS").valuesAsInt().size(), 0);

    for (size_t a = 0; a < new_c3d.nAnalogs; ++a){
        EXPECT_STREQ(new_c3d.c3d.parameters().group("ANALOG").parameter("LABELS").valuesAsString()[a].c_str(), new_c3d.analogNames[a].c_str());
        EXPECT_STREQ(new_c3d.c3d.parameters().group("ANALOG").parameter("DESCRIPTIONS").valuesAsString()[a].c_str(), "");
        EXPECT_FLOAT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("SCALE").valuesAsFloat()[a], 1);
        EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("OFFSET").valuesAsInt()[a], 0);
        EXPECT_STREQ(new_c3d.c3d.parameters().group("ANALOG").parameter("UNITS").valuesAsString()[a].c_str(), "V");
    }

    // DATA
    for (size_t f = 0; f < new_c3d.nFrames; ++f)
        for (size_t sf = 0; sf < new_c3d.nSubframes; ++sf)
            for (size_t c = 0; c < new_c3d.nAnalogs; ++c)
                EXPECT_FLOAT_EQ(new_c3d.c3d.data().frame(static_cast<int>(f)).analogs().subframe(static_cast<int>(sf)).channel(static_cast<int>(c)).value(),
                                static_cast<float>(2*f+3*sf+4*c+1) / static_cast<float>(7.0));
}


TEST(c3dModifier, specificAnalog){
    // Create an empty c3d
    c3dTestStruct new_c3d;
    fillC3D(new_c3d, false, true);

    // Test for removing space at the end of a label
    new_c3d.c3d.addAnalog("AnalogNameWithSpaceAtTheEnd ");
    new_c3d.nAnalogs += 1;
    new_c3d.analogNames.push_back("AnalogNameWithSpaceAtTheEnd");
    EXPECT_STREQ(new_c3d.c3d.parameters().group("ANALOG").parameter("LABELS").valuesAsString()[new_c3d.nAnalogs - 1].c_str(), "AnalogNameWithSpaceAtTheEnd");

    // Add analog by frames
    std::vector<ezc3d::DataNS::Frame> frames;
    // Wrong number of frames
    EXPECT_THROW(new_c3d.c3d.addAnalog(frames), std::runtime_error);

    // Wrong number of subframes
    frames.resize(new_c3d.nFrames);
    EXPECT_THROW(new_c3d.c3d.addAnalog(frames), std::runtime_error);

    // Wrong number of channels
    for (size_t f = 0; f < new_c3d.nFrames; ++f){
        ezc3d::DataNS::Frame frame;
        ezc3d::DataNS::AnalogsNS::Analogs analogs;
        for (size_t sf = 0; sf < new_c3d.nSubframes; ++sf)
            analogs.addSubframe(ezc3d::DataNS::AnalogsNS::SubFrame());
        frame.add(analogs);
        frames[f] = frame;
    }
    EXPECT_THROW(new_c3d.c3d.addAnalog(frames), std::runtime_error);

    // Already existing channels
    for (size_t f = 0; f < new_c3d.nFrames; ++f){
        ezc3d::DataNS::Frame frame;
        ezc3d::DataNS::AnalogsNS::Analogs analogs;
        for (size_t sf = 0; sf < new_c3d.nSubframes; ++sf){
            ezc3d::DataNS::AnalogsNS::SubFrame subframes;
            for (size_t c = 0; c < new_c3d.nAnalogs; ++c){
                ezc3d::DataNS::AnalogsNS::Channel channel;
                channel.name(new_c3d.analogNames[c]);
                channel.value(static_cast<float>(2*f+3*sf+4*c+1) / static_cast<float>(7.0)); // Generate random data
                subframes.addChannel(channel);
            }
            analogs.addSubframe(subframes);
        }
        frame.add(analogs);
        frames[f] = frame;
    }
    EXPECT_THROW(new_c3d.c3d.addAnalog(frames), std::runtime_error);

    // No throw
    std::vector<std::string> analogNames = {"NewAnalog1", "NewAnalog2", "NewAnalog3", "NewAnalog4"};
    for (size_t f = 0; f < new_c3d.nFrames; ++f){
        ezc3d::DataNS::Frame frame;
        ezc3d::DataNS::AnalogsNS::Analogs analogs;
        for (size_t sf = 0; sf < new_c3d.nSubframes; ++sf){
            ezc3d::DataNS::AnalogsNS::SubFrame subframes;
            for (size_t c = 0; c < analogNames.size(); ++c){
                ezc3d::DataNS::AnalogsNS::Channel channel;
                channel.name(analogNames[c]);
                channel.value(static_cast<float>(2*f+3*sf+4*c+1) / static_cast<float>(7.0)); // Generate random data
                subframes.addChannel(channel);
            }
            analogs.addSubframe(subframes);
        }
        frame.add(analogs);
        frames[f] = frame;
    }
    EXPECT_NO_THROW(new_c3d.c3d.addAnalog(frames));
}


TEST(c3dModifier, addPointsAndAnalogs){
    // Create an empty c3d
    c3dTestStruct new_c3d;
    fillC3D(new_c3d, true, true);

    // HEADER
    // Things that should remain as default
    defaultHeaderTest(new_c3d.c3d, HEADER_TYPE::EVENT_ONLY);

    // Things that should have change
    EXPECT_EQ(new_c3d.c3d.header().nb3dPoints(), new_c3d.nMarkers);
    EXPECT_EQ(new_c3d.c3d.header().firstFrame(), 0);
    EXPECT_EQ(new_c3d.c3d.header().lastFrame(), new_c3d.nFrames - 1);
    EXPECT_EQ(new_c3d.c3d.header().nbMaxInterpGap(), 10);
    EXPECT_EQ(new_c3d.c3d.header().scaleFactor(), -1);
    EXPECT_FLOAT_EQ(new_c3d.c3d.header().frameRate(), new_c3d.pointFrameRate);
    EXPECT_EQ(new_c3d.c3d.header().nbFrames(), new_c3d.nFrames);
    EXPECT_EQ(new_c3d.c3d.header().nbAnalogsMeasurement(), new_c3d.nAnalogs * new_c3d.nSubframes);
    EXPECT_EQ(new_c3d.c3d.header().nbAnalogByFrame(), new_c3d.nSubframes);
    EXPECT_EQ(new_c3d.c3d.header().nbAnalogs(), new_c3d.nAnalogs);


    // PARAMETERS
    // Things that should remain as default
    defaultParametersTest(new_c3d.c3d, PARAMETER_TYPE::HEADER);
    defaultParametersTest(new_c3d.c3d, PARAMETER_TYPE::FORCE_PLATFORM);

    // Things that should have change
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("USED").type(), ezc3d::INT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("USED").valuesAsInt().size(), 1);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("USED").valuesAsInt()[0], new_c3d.nMarkers);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("SCALE").valuesAsFloat().size(), 1);
    EXPECT_FLOAT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("SCALE").valuesAsFloat()[0], -1);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("RATE").type(), ezc3d::FLOAT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("RATE").valuesAsFloat().size(), 1);
    EXPECT_FLOAT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("RATE").valuesAsFloat()[0], new_c3d.pointFrameRate);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("FRAMES").type(), ezc3d::INT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("FRAMES").valuesAsInt().size(), 1);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("FRAMES").valuesAsInt()[0], new_c3d.nFrames);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("LABELS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("LABELS").valuesAsString().size(), new_c3d.nMarkers);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("DESCRIPTIONS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("DESCRIPTIONS").valuesAsString().size(), new_c3d.nMarkers);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("UNITS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("UNITS").valuesAsString().size(), new_c3d.nMarkers);
    for (size_t m = 0; m < new_c3d.nMarkers; ++m){
        EXPECT_STREQ(new_c3d.c3d.parameters().group("POINT").parameter("LABELS").valuesAsString()[m].c_str(), new_c3d.markerNames[m].c_str());
        EXPECT_STREQ(new_c3d.c3d.parameters().group("POINT").parameter("DESCRIPTIONS").valuesAsString()[m].c_str(), "");
        EXPECT_STREQ(new_c3d.c3d.parameters().group("POINT").parameter("UNITS").valuesAsString()[m].c_str(), "mm");
    }

    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("USED").type(), ezc3d::INT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("USED").valuesAsInt().size(), 1);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("USED").valuesAsInt()[0], new_c3d.nAnalogs);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("LABELS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("LABELS").valuesAsString().size(), new_c3d.nAnalogs);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("DESCRIPTIONS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("DESCRIPTIONS").valuesAsString().size(), new_c3d.nAnalogs);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("GEN_SCALE").type(), ezc3d::INT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("GEN_SCALE").valuesAsInt().size(), 1);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("GEN_SCALE").valuesAsInt()[0], 1);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("SCALE").valuesAsFloat().size(), new_c3d.nAnalogs);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("OFFSET").type(), ezc3d::INT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("OFFSET").valuesAsInt().size(), new_c3d.nAnalogs);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("UNITS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("UNITS").valuesAsString().size(), new_c3d.nAnalogs);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("RATE").type(), ezc3d::FLOAT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("RATE").valuesAsFloat().size(), 1);
    EXPECT_FLOAT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("RATE").valuesAsFloat()[0], new_c3d.analogFrameRate);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("FORMAT").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("FORMAT").valuesAsString().size(), 0);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("BITS").type(), ezc3d::INT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("BITS").valuesAsInt().size(), 0);

    for (size_t a = 0; a < new_c3d.nAnalogs; ++a){
        EXPECT_STREQ(new_c3d.c3d.parameters().group("ANALOG").parameter("LABELS").valuesAsString()[a].c_str(), new_c3d.analogNames[a].c_str());
        EXPECT_STREQ(new_c3d.c3d.parameters().group("ANALOG").parameter("DESCRIPTIONS").valuesAsString()[a].c_str(), "");
        EXPECT_FLOAT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("SCALE").valuesAsFloat()[a], 1);
        EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("OFFSET").valuesAsInt()[a], 0);
        EXPECT_STREQ(new_c3d.c3d.parameters().group("ANALOG").parameter("UNITS").valuesAsString()[a].c_str(), "V");
    }


    // DATA
    for (size_t f = 0; f < new_c3d.nFrames; ++f){
        for (size_t m = 0; m < new_c3d.nMarkers; ++m){
            EXPECT_FLOAT_EQ(new_c3d.c3d.data().frame(static_cast<int>(f)).points().point(new_c3d.markerNames[m]).x(), static_cast<float>(2*f+3*m+1) / static_cast<float>(7.0));
            EXPECT_FLOAT_EQ(new_c3d.c3d.data().frame(static_cast<int>(f)).points().point(new_c3d.markerNames[m]).y(), static_cast<float>(2*f+3*m+2) / static_cast<float>(7.0));
            EXPECT_FLOAT_EQ(new_c3d.c3d.data().frame(static_cast<int>(f)).points().point(new_c3d.markerNames[m]).z(), static_cast<float>(2*f+3*m+3) / static_cast<float>(7.0));
        }

        for (size_t sf = 0; sf < new_c3d.nSubframes; ++sf)
            for (size_t c = 0; c < new_c3d.nAnalogs; ++c)
                EXPECT_FLOAT_EQ(new_c3d.c3d.data().frame(static_cast<int>(f)).analogs().subframe(static_cast<int>(sf)).channel(static_cast<int>(c)).value(),
                                static_cast<float>(2*f+3*sf+4*c+1) / static_cast<float>(7.0));

    }

}


TEST(c3dModifier, specificFrames){
    // Create an empty c3d
    c3dTestStruct new_c3d;
    fillC3D(new_c3d, true, false);

    // Create impossible points
    ezc3d::DataNS::Points3dNS::Points p(4);
    EXPECT_THROW( ezc3d::DataNS::Points3dNS::Points stupidPoint(-1), std::out_of_range);

    // Add an impossible frame
    ezc3d::DataNS::Frame stupidFrame;
    // Wrong number of points
    EXPECT_THROW(new_c3d.c3d.addFrame(stupidFrame), std::runtime_error);

    // Wrong number of frames
    ezc3d::DataNS::Points3dNS::Points stupidPoints(new_c3d.c3d.data().frame(0).points());
    stupidFrame.add(stupidPoints);
    new_c3d.c3d.addFrame(stupidFrame);

    // Wrong name of points
    stupidPoints.points_nonConst()[0].name("WrongName"); // Change the name of a marker
    stupidFrame.add(stupidPoints);
    EXPECT_THROW(new_c3d.c3d.addFrame(stupidFrame), std::runtime_error);


    // Replace existing frame
    for (size_t f = 0; f < new_c3d.nFrames; ++f){
        ezc3d::DataNS::Points3dNS::Points pts;
        for (size_t m = 0; m < new_c3d.nMarkers; ++m){
            ezc3d::DataNS::Points3dNS::Point pt;
            pt.name(new_c3d.markerNames[m]);
            // Generate some random data
            pt.x(static_cast<float>(4*f+2*m+5) / static_cast<float>(17.0));
            pt.y(static_cast<float>(4*f+2*m+6) / static_cast<float>(17.0));
            pt.z(static_cast<float>(4*f+2*m+7) / static_cast<float>(17.0));
            pts.add(pt);
        }
        ezc3d::DataNS::Frame frame;
        frame.add(pts);
        new_c3d.c3d.addFrame(frame, static_cast<int>(f));
    }

    // Add a new frame at 2 spaces after the last frame
    {
        size_t f(new_c3d.nFrames + 1);
        ezc3d::DataNS::Points3dNS::Points pts;
        for (size_t m = 0; m < new_c3d.nMarkers; ++m){
            ezc3d::DataNS::Points3dNS::Point pt;
            pt.name(new_c3d.markerNames[m]);
            // Generate some random data
            pt.x(static_cast<float>(4*f+2*m+5) / static_cast<float>(17.0));
            pt.y(static_cast<float>(4*f+2*m+6) / static_cast<float>(17.0));
            pt.z(static_cast<float>(4*f+2*m+7) / static_cast<float>(17.0));
            pts.add(pt);
        }
        ezc3d::DataNS::Frame frame;
        frame.add(pts);
        new_c3d.c3d.addFrame(frame, static_cast<int>(f));
    }
    new_c3d.nFrames += 2;

    // Get a frame and some inexistant one
    EXPECT_THROW(new_c3d.c3d.data().frame(-1), std::out_of_range);
    EXPECT_THROW(new_c3d.c3d.data().frame(static_cast<int>(new_c3d.nFrames)), std::out_of_range);


    // HEADER
    // Things that should remain as default
    defaultHeaderTest(new_c3d.c3d, HEADER_TYPE::ANALOG_AND_EVENT);

    // Things that should have change
    EXPECT_EQ(new_c3d.c3d.header().nb3dPoints(), new_c3d.nMarkers);
    EXPECT_EQ(new_c3d.c3d.header().firstFrame(), 0);
    EXPECT_EQ(new_c3d.c3d.header().lastFrame(), new_c3d.nFrames - 1);
    EXPECT_EQ(new_c3d.c3d.header().nbMaxInterpGap(), 10);
    EXPECT_EQ(new_c3d.c3d.header().scaleFactor(), -1);
    EXPECT_FLOAT_EQ(new_c3d.c3d.header().frameRate(), new_c3d.pointFrameRate);
    EXPECT_EQ(new_c3d.c3d.header().nbFrames(), new_c3d.nFrames);

    // PARAMETERS
    // Things that should remain as default
    defaultParametersTest(new_c3d.c3d, PARAMETER_TYPE::HEADER);
    defaultParametersTest(new_c3d.c3d, PARAMETER_TYPE::ANALOG);
    defaultParametersTest(new_c3d.c3d, PARAMETER_TYPE::FORCE_PLATFORM);

    // Things that should have change
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("USED").type(), ezc3d::INT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("USED").valuesAsInt().size(), 1);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("USED").valuesAsInt()[0], new_c3d.nMarkers);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("SCALE").valuesAsFloat().size(), 1);
    EXPECT_FLOAT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("SCALE").valuesAsFloat()[0], -1);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("RATE").type(), ezc3d::FLOAT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("RATE").valuesAsFloat().size(), 1);
    EXPECT_FLOAT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("RATE").valuesAsFloat()[0], new_c3d.pointFrameRate);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("FRAMES").type(), ezc3d::INT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("FRAMES").valuesAsInt().size(), 1);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("FRAMES").valuesAsInt()[0], new_c3d.nFrames);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("LABELS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("LABELS").valuesAsString().size(), new_c3d.nMarkers);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("DESCRIPTIONS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("DESCRIPTIONS").valuesAsString().size(), new_c3d.nMarkers);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("UNITS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("UNITS").valuesAsString().size(), new_c3d.nMarkers);
    for (size_t m = 0; m < new_c3d.nMarkers; ++m){
        EXPECT_STREQ(new_c3d.c3d.parameters().group("POINT").parameter("LABELS").valuesAsString()[m].c_str(),new_c3d.markerNames[m].c_str());
        EXPECT_STREQ(new_c3d.c3d.parameters().group("POINT").parameter("DESCRIPTIONS").valuesAsString()[m].c_str(), "");
        EXPECT_STREQ(new_c3d.c3d.parameters().group("POINT").parameter("UNITS").valuesAsString()[m].c_str(), "mm");
    }

    // DATA
    for (size_t f = 0; f < new_c3d.nFrames; ++f){
        if (f != new_c3d.nFrames - 2) { // Where no actual frames where added
            for (size_t m = 0; m < new_c3d.nMarkers; ++m){
                EXPECT_FLOAT_EQ(new_c3d.c3d.data().frame(static_cast<int>(f)).points().point(new_c3d.markerNames[m]).x(), static_cast<float>(4*f+2*m+5) / static_cast<float>(17.0));
                EXPECT_FLOAT_EQ(new_c3d.c3d.data().frame(static_cast<int>(f)).points().point(new_c3d.markerNames[m]).y(), static_cast<float>(4*f+2*m+6) / static_cast<float>(17.0));
                EXPECT_FLOAT_EQ(new_c3d.c3d.data().frame(static_cast<int>(f)).points().point(new_c3d.markerNames[m]).z(), static_cast<float>(4*f+2*m+7) / static_cast<float>(17.0));
            }
        } else {
            for (size_t m = 0; m < new_c3d.nMarkers; ++m){
                EXPECT_THROW(new_c3d.c3d.data().frame(static_cast<int>(f)).points().point(new_c3d.markerNames[m]).x(), std::invalid_argument);
                EXPECT_THROW(new_c3d.c3d.data().frame(static_cast<int>(f)).points().point(new_c3d.markerNames[m]).y(), std::invalid_argument);
                EXPECT_THROW(new_c3d.c3d.data().frame(static_cast<int>(f)).points().point(new_c3d.markerNames[m]).z(), std::invalid_argument);
            }
        }
    }
}




TEST(c3dFileIO, CreateWriteAndReadBack){
    // Create an empty c3d fill it with data and reopen
    c3dTestStruct ref_c3d;
    fillC3D(ref_c3d, true, true);

    // Lock Point parameter
    ref_c3d.c3d.lockGroup("POINT");

    // Write the c3d on the disk
    std::string savePath("temporary.c3d");
    ref_c3d.c3d.write(savePath.c_str());

    // Open it back and delete it
    ezc3d::c3d read_c3d(savePath.c_str());
    remove(savePath.c_str());

    // Test the read file
    // HEADER
    // Things that should remain as default
    defaultHeaderTest(read_c3d, HEADER_TYPE::EVENT_ONLY);

    // Things that should have change
    EXPECT_EQ(read_c3d.header().nb3dPoints(), ref_c3d.nMarkers);
    EXPECT_EQ(read_c3d.header().firstFrame(), 0);
    EXPECT_EQ(read_c3d.header().lastFrame(), ref_c3d.nFrames - 1);
    EXPECT_EQ(read_c3d.header().nbMaxInterpGap(), 10);
    EXPECT_EQ(read_c3d.header().scaleFactor(), -1);
    EXPECT_FLOAT_EQ(read_c3d.header().frameRate(), ref_c3d.pointFrameRate);
    EXPECT_EQ(read_c3d.header().nbFrames(), ref_c3d.nFrames);
    EXPECT_EQ(read_c3d.header().nbAnalogsMeasurement(), ref_c3d.nAnalogs * ref_c3d.nSubframes);
    EXPECT_EQ(read_c3d.header().nbAnalogByFrame(), ref_c3d.nSubframes);
    EXPECT_EQ(read_c3d.header().nbAnalogs(), ref_c3d.nAnalogs);


    // PARAMETERS
    // Things that should remain as default
    defaultParametersTest(read_c3d, PARAMETER_TYPE::HEADER);
    defaultParametersTest(read_c3d, PARAMETER_TYPE::FORCE_PLATFORM);

    // Things that should have change
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("USED").type(), ezc3d::INT);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("USED").valuesAsInt().size(), 1);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("USED").valuesAsInt()[0], ref_c3d.nMarkers);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("SCALE").valuesAsFloat().size(), 1);
    EXPECT_FLOAT_EQ(read_c3d.parameters().group("POINT").parameter("SCALE").valuesAsFloat()[0], -1);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("RATE").type(), ezc3d::FLOAT);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("RATE").valuesAsFloat().size(), 1);
    EXPECT_FLOAT_EQ(read_c3d.parameters().group("POINT").parameter("RATE").valuesAsFloat()[0], ref_c3d.pointFrameRate);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("FRAMES").type(), ezc3d::INT);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("FRAMES").valuesAsInt().size(), 1);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("FRAMES").valuesAsInt()[0], ref_c3d.nFrames);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("LABELS").type(), ezc3d::CHAR);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("LABELS").valuesAsString().size(), ref_c3d.nMarkers);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("DESCRIPTIONS").type(), ezc3d::CHAR);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("DESCRIPTIONS").valuesAsString().size(), ref_c3d.nMarkers);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("UNITS").type(), ezc3d::CHAR);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("UNITS").valuesAsString().size(), ref_c3d.nMarkers);
    for (size_t m = 0; m < ref_c3d.nMarkers; ++m){
        EXPECT_STREQ(read_c3d.parameters().group("POINT").parameter("LABELS").valuesAsString()[m].c_str(), ref_c3d.markerNames[m].c_str());
        EXPECT_STREQ(read_c3d.parameters().group("POINT").parameter("DESCRIPTIONS").valuesAsString()[m].c_str(), "");
        EXPECT_STREQ(read_c3d.parameters().group("POINT").parameter("UNITS").valuesAsString()[m].c_str(), "mm");
    }

    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("USED").type(), ezc3d::INT);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("USED").valuesAsInt().size(), 1);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("USED").valuesAsInt()[0], ref_c3d.nAnalogs);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("LABELS").type(), ezc3d::CHAR);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("LABELS").valuesAsString().size(), ref_c3d.nAnalogs);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("DESCRIPTIONS").type(), ezc3d::CHAR);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("DESCRIPTIONS").valuesAsString().size(), ref_c3d.nAnalogs);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("GEN_SCALE").type(), ezc3d::INT);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("GEN_SCALE").valuesAsInt().size(), 1);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("GEN_SCALE").valuesAsInt()[0], 1);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("SCALE").valuesAsFloat().size(), ref_c3d.nAnalogs);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("OFFSET").type(), ezc3d::INT);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("OFFSET").valuesAsInt().size(), ref_c3d.nAnalogs);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("UNITS").type(), ezc3d::CHAR);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("UNITS").valuesAsString().size(), ref_c3d.nAnalogs);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("RATE").type(), ezc3d::FLOAT);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("RATE").valuesAsFloat().size(), 1);
    EXPECT_FLOAT_EQ(read_c3d.parameters().group("ANALOG").parameter("RATE").valuesAsFloat()[0], ref_c3d.analogFrameRate);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("FORMAT").type(), ezc3d::CHAR);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("FORMAT").valuesAsString().size(), 0);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("BITS").type(), ezc3d::INT);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("BITS").valuesAsInt().size(), 0);

    for (size_t a = 0; a < ref_c3d.nAnalogs; ++a){
        EXPECT_STREQ(read_c3d.parameters().group("ANALOG").parameter("LABELS").valuesAsString()[a].c_str(), ref_c3d.analogNames[a].c_str());
        EXPECT_STREQ(read_c3d.parameters().group("ANALOG").parameter("DESCRIPTIONS").valuesAsString()[a].c_str(), "");
        EXPECT_FLOAT_EQ(read_c3d.parameters().group("ANALOG").parameter("SCALE").valuesAsFloat()[a], 1);
        EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("OFFSET").valuesAsInt()[a], 0);
        EXPECT_STREQ(read_c3d.parameters().group("ANALOG").parameter("UNITS").valuesAsString()[a].c_str(), "V");
    }


    // DATA
    for (size_t f = 0; f < ref_c3d.nFrames; ++f){
        for (size_t m = 0; m < ref_c3d.nMarkers; ++m){
            EXPECT_FLOAT_EQ(read_c3d.data().frame(static_cast<int>(f)).points().point(ref_c3d.markerNames[m]).x(), static_cast<float>(2*f+3*m+1) / static_cast<float>(7.0));
            EXPECT_FLOAT_EQ(read_c3d.data().frame(static_cast<int>(f)).points().point(ref_c3d.markerNames[m]).y(), static_cast<float>(2*f+3*m+2) / static_cast<float>(7.0));
            EXPECT_FLOAT_EQ(read_c3d.data().frame(static_cast<int>(f)).points().point(ref_c3d.markerNames[m]).z(), static_cast<float>(2*f+3*m+3) / static_cast<float>(7.0));
        }

        for (size_t sf = 0; sf < ref_c3d.nSubframes; ++sf)
            for (size_t c = 0; c < ref_c3d.nAnalogs; ++c)
                EXPECT_FLOAT_EQ(read_c3d.data().frame(static_cast<int>(f)).analogs().subframe(static_cast<int>(sf)).channel(static_cast<int>(c)).value(),
                                static_cast<float>(2*f+3*sf+4*c+1) / static_cast<float>(7.0));

    }
}



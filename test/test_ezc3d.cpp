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
                pts.point(pt);
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
                    subframes.channel(channel);
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
    EXPECT_EQ(new_c3d.data().nbFrames(), 0);
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

    // Get an erroneous parameter
    EXPECT_THROW(new_c3d.c3d.parameters().group("POINT").parameter("ThisIsNotARealParamter"), std::invalid_argument);

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

    // Fill the parameter improperly
    EXPECT_THROW(p.set(std::vector<int>()={}, {1}), std::range_error);
    EXPECT_THROW(p.set(std::vector<int>()={1}, {0}), std::range_error);
    EXPECT_THROW(p.set(std::vector<float>()={}, {1}), std::range_error);
    EXPECT_THROW(p.set(std::vector<float>()={1}, {0}), std::range_error);
    EXPECT_THROW(p.set(std::vector<std::string>()={}, {1}), std::range_error);
    EXPECT_THROW(p.set(std::vector<std::string>()={""}, {0}), std::range_error);
    p.dimension();

    // Add twice the same group (That should not be needed for a user API since he should use new_c3d.c3d.addParameter() )
    {
        ezc3d::ParametersNS::Parameters params;
        ezc3d::ParametersNS::GroupNS::Group groupToBeAddedTwice;
        ezc3d::ParametersNS::GroupNS::Parameter p("UselessParameter");
        p.set(std::vector<int>()={}, {0});
        groupToBeAddedTwice.addParameter(p);
        EXPECT_NO_THROW(params.addGroup(groupToBeAddedTwice));
        EXPECT_NO_THROW(params.addGroup(groupToBeAddedTwice));
    }
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

    // Add frame with a new marker with not enough frames
    std::vector<ezc3d::DataNS::Frame> new_frames;
    EXPECT_THROW(new_c3d.c3d.addPoint(new_frames), std::runtime_error);
    for (size_t f = 0; f < new_c3d.c3d.data().nbFrames() - 1; ++f)
        new_frames.push_back(ezc3d::DataNS::Frame());
    EXPECT_THROW(new_c3d.c3d.addPoint(new_frames), std::runtime_error);

    // Not enough points
    new_frames.push_back(ezc3d::DataNS::Frame());
    EXPECT_THROW(new_c3d.c3d.addPoint(new_frames), std::runtime_error);

    // Try adding an already existing point
    for (size_t f = 0; f < new_c3d.c3d.data().nbFrames(); ++f){
        ezc3d::DataNS::Points3dNS::Points pts;
        ezc3d::DataNS::Points3dNS::Point pt(new_c3d.c3d.data().frame(static_cast<int>(f)).points().point(0));
        pts.point(pt);
        new_frames[f].add(pts);
    }
    EXPECT_THROW(new_c3d.c3d.addPoint(new_frames), std::runtime_error);

    // Adding it properly
    for (size_t f = 0; f < new_c3d.c3d.data().nbFrames(); ++f){
        ezc3d::DataNS::Points3dNS::Points pts;
        ezc3d::DataNS::Points3dNS::Point pt;
        pts.point(pt);
        new_frames[f].add(pts);
    }
    EXPECT_NO_THROW(new_c3d.c3d.addPoint(new_frames));

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
            pts.point(pt, m);
        }

        frame.add(pts);
        new_c3d.c3d.addFrame(frame, static_cast<int>(f));
    }

    // failed test replacing a point
    {
        ezc3d::DataNS::Points3dNS::Point ptToBeReplaced;
        ptToBeReplaced.name("ToBeReplaced");
        ezc3d::DataNS::Points3dNS::Point ptToReplace;
        ptToReplace.name("ToReplace");

        ezc3d::DataNS::Points3dNS::Points pts;
        pts.point(ptToBeReplaced);
        size_t previousNbPoints(pts.nbPoints());
        EXPECT_NO_THROW(pts.point(ptToReplace, pts.nbPoints()+2));
        EXPECT_EQ(pts.nbPoints(), previousNbPoints + 3);
        EXPECT_NO_THROW(pts.point(ptToReplace, 0));
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
    EXPECT_NO_THROW(ezc3d::DataNS::AnalogsNS::SubFrame(0));
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
                subframes.channel(channel);
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
                subframes.channel(channel);
            }
            analogs.addSubframe(subframes);
        }
        frame.add(analogs);
        frames[f] = frame;
    }
    EXPECT_NO_THROW(new_c3d.c3d.addAnalog(frames));

    // Get channel names
    for (size_t c = 0; c < new_c3d.analogNames.size(); ++c)
        EXPECT_NO_THROW(new_c3d.c3d.data().frame(0).analogs().subframe(0).channel(new_c3d.analogNames[c]));
    EXPECT_THROW(new_c3d.c3d.data().frame(0).analogs().subframe(0).channel("ThisIsNotARealChannel"), std::invalid_argument);

    // Get a subframe
    {
        EXPECT_THROW(new_c3d.c3d.data().frame(-1), std::out_of_range);
        EXPECT_THROW(new_c3d.c3d.data().frame(static_cast<int>(new_c3d.c3d.data().nbFrames())), std::out_of_range);
        EXPECT_NO_THROW(new_c3d.c3d.data().frame(0));
    }

    // Create an analog and replace the subframes
    {
        EXPECT_THROW(ezc3d::DataNS::AnalogsNS::Analogs(-1), std::out_of_range);
        ezc3d::DataNS::AnalogsNS::Analogs analogs;
        ezc3d::DataNS::AnalogsNS::SubFrame sfToBeReplaced;
        ezc3d::DataNS::AnalogsNS::SubFrame sfToReplace;
        analogs.addSubframe(sfToBeReplaced);

        EXPECT_THROW(analogs.replaceSubframe(-1, sfToReplace), std::out_of_range);
        EXPECT_THROW(analogs.replaceSubframe(static_cast<int>(analogs.subframes().size()), sfToReplace), std::out_of_range);
        EXPECT_NO_THROW(analogs.replaceSubframe(0, sfToReplace));

        EXPECT_THROW(analogs.subframe(-1), std::out_of_range);
        EXPECT_THROW(analogs.subframe(static_cast<int>(analogs.subframes().size())), std::out_of_range);
        EXPECT_NO_THROW(analogs.subframe(0));
    }

    // adding/replacing channels and getting them
    {
        ezc3d::DataNS::AnalogsNS::Channel channelToBeReplaced;
        channelToBeReplaced.name("ToBeReplaced");
        ezc3d::DataNS::AnalogsNS::Channel channelToReplace;
        channelToReplace.name("ToReplace");

        ezc3d::DataNS::AnalogsNS::SubFrame subframe;
        subframe.channel(channelToBeReplaced);
        size_t nbChannelOld(subframe.nbChannels());
        EXPECT_NO_THROW(subframe.channel(channelToReplace, subframe.nbChannels()+2));
        EXPECT_EQ(subframe.nbChannels(), nbChannelOld + 3);
        EXPECT_NO_THROW(subframe.channel(channelToReplace, 0));

        EXPECT_THROW(subframe.channel(-1), std::out_of_range);
        EXPECT_THROW(subframe.channel(static_cast<int>(subframe.nbChannels())), std::out_of_range);
        EXPECT_NO_THROW(subframe.channel(0));
    }
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


TEST(c3dModifier, addFrames){
    // Create an empty c3d
    c3dTestStruct new_c3d;
    fillC3D(new_c3d, true, false);

    // Create impossible points
    ezc3d::DataNS::Points3dNS::Points p(4);

    // Add an impossible frame
    ezc3d::DataNS::Frame stupidFramePoint;
    // Wrong number of points
    EXPECT_THROW(new_c3d.c3d.addFrame(stupidFramePoint), std::runtime_error);

    // Wrong number of frames
    ezc3d::DataNS::Points3dNS::Points stupidPoints(new_c3d.c3d.data().frame(0).points());
    stupidFramePoint.add(stupidPoints);
    new_c3d.c3d.addFrame(stupidFramePoint);

    // Wrong name of points
    ezc3d::DataNS::Points3dNS::Point pt(new_c3d.c3d.data().frame(0).points().point(0));
    std::string realPointName(pt.name());
    pt.name("WrongName");
    stupidPoints.point(pt, 0);
    stupidFramePoint.add(stupidPoints);
    EXPECT_THROW(new_c3d.c3d.addFrame(stupidFramePoint), std::invalid_argument);

    // Put back the good name
    pt.name(realPointName);
    stupidPoints.point(pt, 0);

    // Add Analogs
    new_c3d.nAnalogs = 3;
    ezc3d::ParametersNS::GroupNS::Parameter analogUsed(new_c3d.c3d.parameters().group("ANALOG").parameter("USED"));
    analogUsed.set(std::vector<int>()={static_cast<int>(new_c3d.nAnalogs)}, {1});
    new_c3d.c3d.addParameter("ANALOG", analogUsed);

    ezc3d::DataNS::Frame stupidFrameAnalog;
    ezc3d::DataNS::AnalogsNS::Analogs stupidAnalogs(new_c3d.c3d.data().frame(0).analogs());
    ezc3d::DataNS::AnalogsNS::SubFrame stupidSubframe(static_cast<int>(new_c3d.nAnalogs)-1);
    stupidAnalogs.addSubframe(stupidSubframe);
    stupidFrameAnalog.add(stupidPoints, stupidAnalogs);
    EXPECT_THROW(new_c3d.c3d.addFrame(stupidFrameAnalog), std::runtime_error); // Wrong frame rate for analogs

    ezc3d::ParametersNS::GroupNS::Parameter analogRate(new_c3d.c3d.parameters().group("ANALOG").parameter("RATE"));
    analogRate.set(std::vector<float>()={100}, {1});
    new_c3d.c3d.addParameter("ANALOG", analogRate);
    EXPECT_THROW(new_c3d.c3d.addFrame(stupidFrameAnalog), std::runtime_error);

    ezc3d::DataNS::AnalogsNS::SubFrame notSoStupidSubframe(static_cast<int>(new_c3d.nAnalogs));
    stupidAnalogs.replaceSubframe(0, notSoStupidSubframe);
    stupidFrameAnalog.add(stupidPoints, stupidAnalogs);
    EXPECT_NO_THROW(new_c3d.c3d.addFrame(stupidFrameAnalog));

    // Remove point frame rate and then
    ezc3d::ParametersNS::GroupNS::Parameter pointRate(new_c3d.c3d.parameters().group("POINT").parameter("RATE"));
    pointRate.set(std::vector<float>()={0}, {1});
    new_c3d.c3d.addParameter("POINT", pointRate);
    EXPECT_THROW(new_c3d.c3d.addFrame(stupidFrameAnalog), std::runtime_error);
}


TEST(c3dModifier, specificFrames){
    // Create an empty c3d
    c3dTestStruct new_c3d;
    fillC3D(new_c3d, true, false);


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
            pts.point(pt);
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
            pts.point(pt);
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


TEST(c3dFileIO, readViconC3D){
    ezc3d::c3d Vicon("c3dTestFiles/Vicon.c3d");

    // Header test
    // Generic stuff
    EXPECT_EQ(Vicon.header().checksum(), 80);
    EXPECT_EQ(Vicon.header().keyLabelPresent(), 0);
    EXPECT_EQ(Vicon.header().firstBlockKeyLabel(), 0);
    EXPECT_EQ(Vicon.header().fourCharPresent(), 12345);
    EXPECT_EQ(Vicon.header().emptyBlock1(), 0);
    EXPECT_EQ(Vicon.header().emptyBlock2(), 0);
    EXPECT_EQ(Vicon.header().emptyBlock3(), 0);
    EXPECT_EQ(Vicon.header().emptyBlock4(), 0);

    // Point stuff
    EXPECT_EQ(Vicon.header().nb3dPoints(), 51);
    EXPECT_EQ(Vicon.header().nbMaxInterpGap(), 0);
    EXPECT_EQ(Vicon.header().scaleFactor(), -1138501878);
    EXPECT_FLOAT_EQ(Vicon.header().frameRate(), 100);

    // Analog stuff
    EXPECT_EQ(Vicon.header().nbAnalogsMeasurement(), 760);
    EXPECT_EQ(Vicon.header().nbAnalogByFrame(), 20);
    EXPECT_EQ(Vicon.header().nbAnalogs(), 38);

    // Event stuff
    EXPECT_EQ(Vicon.header().nbEvents(), 0);

    EXPECT_EQ(Vicon.header().eventsTime().size(), 18);
    EXPECT_THROW(Vicon.header().eventsTime(-1), std::invalid_argument);
    for (int e = 0; e < static_cast<int>(Vicon.header().eventsTime().size()); ++e)
        EXPECT_FLOAT_EQ(Vicon.header().eventsTime(e), 0);
    EXPECT_THROW(Vicon.header().eventsTime(static_cast<int>(Vicon.header().eventsTime().size())), std::invalid_argument);

    EXPECT_EQ(Vicon.header().eventsLabel().size(), 18);
    EXPECT_THROW(Vicon.header().eventsLabel(-1), std::invalid_argument);
    for (int e = 0; e < static_cast<int>(Vicon.header().eventsLabel().size()); ++e)
        EXPECT_STREQ(Vicon.header().eventsLabel(e).c_str(), "");
    EXPECT_THROW(Vicon.header().eventsLabel(static_cast<int>(Vicon.header().eventsLabel().size())), std::invalid_argument);

    EXPECT_EQ(Vicon.header().eventsDisplay().size(), 9);
    EXPECT_THROW(Vicon.header().eventsDisplay(-1), std::invalid_argument);
    for (int e = 0; e < static_cast<int>(Vicon.header().eventsDisplay().size()); ++e)
        EXPECT_EQ(Vicon.header().eventsDisplay(e), 0);
    EXPECT_THROW(Vicon.header().eventsDisplay(static_cast<int>(Vicon.header().eventsDisplay().size())), std::invalid_argument);


    EXPECT_EQ(Vicon.header().firstFrame(), 0);
    EXPECT_EQ(Vicon.header().lastFrame(), 579);
    EXPECT_EQ(Vicon.header().nbFrames(), 580);


    // Parameter tests
    EXPECT_EQ(Vicon.parameters().checksum(), 80);
    EXPECT_EQ(Vicon.parameters().groups().size(), 9);

    EXPECT_EQ(Vicon.parameters().group("POINT").parameter("USED").type(), ezc3d::INT);
    EXPECT_EQ(Vicon.parameters().group("POINT").parameter("USED").valuesAsInt().size(), 1);
    EXPECT_EQ(Vicon.parameters().group("POINT").parameter("USED").valuesAsInt()[0], 51);
    EXPECT_EQ(Vicon.parameters().group("POINT").parameter("SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(Vicon.parameters().group("POINT").parameter("SCALE").valuesAsFloat().size(), 1);
    EXPECT_FLOAT_EQ(Vicon.parameters().group("POINT").parameter("SCALE").valuesAsFloat()[0], static_cast<float>(-0.0099999998));
    EXPECT_EQ(Vicon.parameters().group("POINT").parameter("RATE").type(), ezc3d::FLOAT);
    EXPECT_EQ(Vicon.parameters().group("POINT").parameter("RATE").valuesAsFloat().size(), 1);
    EXPECT_FLOAT_EQ(Vicon.parameters().group("POINT").parameter("RATE").valuesAsFloat()[0], 100);
    EXPECT_EQ(Vicon.parameters().group("POINT").parameter("FRAMES").valuesAsInt()[0], 580); // ignore because it changes if analog is present
    EXPECT_EQ(Vicon.parameters().group("POINT").parameter("FRAMES").type(), ezc3d::INT);
    EXPECT_EQ(Vicon.parameters().group("POINT").parameter("LABELS").type(), ezc3d::CHAR);
    EXPECT_EQ(Vicon.parameters().group("POINT").parameter("LABELS").valuesAsString().size(), 51);
    EXPECT_EQ(Vicon.parameters().group("POINT").parameter("DESCRIPTIONS").type(), ezc3d::CHAR);
    EXPECT_EQ(Vicon.parameters().group("POINT").parameter("DESCRIPTIONS").valuesAsString().size(), 51);
    EXPECT_EQ(Vicon.parameters().group("POINT").parameter("UNITS").type(), ezc3d::CHAR);
    EXPECT_EQ(Vicon.parameters().group("POINT").parameter("UNITS").valuesAsString().size(), 1);

    EXPECT_EQ(Vicon.parameters().group("ANALOG").parameter("USED").type(), ezc3d::INT);
    EXPECT_EQ(Vicon.parameters().group("ANALOG").parameter("USED").valuesAsInt().size(), 1);
    EXPECT_EQ(Vicon.parameters().group("ANALOG").parameter("USED").valuesAsInt()[0], 38);
    EXPECT_EQ(Vicon.parameters().group("ANALOG").parameter("LABELS").type(), ezc3d::CHAR);
    EXPECT_EQ(Vicon.parameters().group("ANALOG").parameter("LABELS").valuesAsString().size(), 38);
    EXPECT_EQ(Vicon.parameters().group("ANALOG").parameter("DESCRIPTIONS").type(), ezc3d::CHAR);
    EXPECT_EQ(Vicon.parameters().group("ANALOG").parameter("DESCRIPTIONS").valuesAsString().size(), 38);
    EXPECT_EQ(Vicon.parameters().group("ANALOG").parameter("GEN_SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(Vicon.parameters().group("ANALOG").parameter("GEN_SCALE").valuesAsFloat().size(), 1);
    EXPECT_EQ(Vicon.parameters().group("ANALOG").parameter("GEN_SCALE").valuesAsFloat()[0], 1);
    EXPECT_EQ(Vicon.parameters().group("ANALOG").parameter("SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(Vicon.parameters().group("ANALOG").parameter("SCALE").valuesAsFloat().size(), 38);
    EXPECT_EQ(Vicon.parameters().group("ANALOG").parameter("OFFSET").type(), ezc3d::INT);
    EXPECT_EQ(Vicon.parameters().group("ANALOG").parameter("OFFSET").valuesAsInt().size(), 38);
    EXPECT_EQ(Vicon.parameters().group("ANALOG").parameter("UNITS").type(), ezc3d::CHAR);
    EXPECT_EQ(Vicon.parameters().group("ANALOG").parameter("UNITS").valuesAsString().size(), 38);
    EXPECT_EQ(Vicon.parameters().group("ANALOG").parameter("RATE").type(), ezc3d::FLOAT);
    EXPECT_EQ(Vicon.parameters().group("ANALOG").parameter("RATE").valuesAsFloat().size(), 1);
    EXPECT_FLOAT_EQ(Vicon.parameters().group("ANALOG").parameter("RATE").valuesAsFloat()[0], 2000);

    EXPECT_EQ(Vicon.parameters().group("FORCE_PLATFORM").parameter("USED").type(), ezc3d::INT);
    EXPECT_EQ(Vicon.parameters().group("FORCE_PLATFORM").parameter("USED").valuesAsInt().size(), 1);
    EXPECT_EQ(Vicon.parameters().group("FORCE_PLATFORM").parameter("USED").valuesAsInt()[0], 0);
    EXPECT_EQ(Vicon.parameters().group("FORCE_PLATFORM").parameter("TYPE").type(), ezc3d::INT);
    EXPECT_EQ(Vicon.parameters().group("FORCE_PLATFORM").parameter("TYPE").valuesAsInt().size(), 0);
    EXPECT_EQ(Vicon.parameters().group("FORCE_PLATFORM").parameter("ZERO").type(), ezc3d::INT);
    EXPECT_EQ(Vicon.parameters().group("FORCE_PLATFORM").parameter("ZERO").valuesAsInt().size(), 2);
    EXPECT_EQ(Vicon.parameters().group("FORCE_PLATFORM").parameter("ZERO").valuesAsInt()[0], 1);
    EXPECT_EQ(Vicon.parameters().group("FORCE_PLATFORM").parameter("ZERO").valuesAsInt()[1], 0);
    EXPECT_EQ(Vicon.parameters().group("FORCE_PLATFORM").parameter("CORNERS").type(), ezc3d::FLOAT);
    EXPECT_EQ(Vicon.parameters().group("FORCE_PLATFORM").parameter("CORNERS").valuesAsFloat().size(), 0);
    EXPECT_EQ(Vicon.parameters().group("FORCE_PLATFORM").parameter("ORIGIN").type(), ezc3d::FLOAT);
    EXPECT_EQ(Vicon.parameters().group("FORCE_PLATFORM").parameter("ORIGIN").valuesAsFloat().size(), 0);
    EXPECT_EQ(Vicon.parameters().group("FORCE_PLATFORM").parameter("CHANNEL").type(), ezc3d::INT);
    EXPECT_EQ(Vicon.parameters().group("FORCE_PLATFORM").parameter("CHANNEL").valuesAsInt().size(), 0);

    EXPECT_STREQ(Vicon.parameters().group("MANUFACTURER").parameter("COMPANY").valuesAsString()[0].c_str(), "Vicon");
    EXPECT_STREQ(Vicon.parameters().group("MANUFACTURER").parameter("SOFTWARE").valuesAsString()[0].c_str(), "Vicon Nexus");
    EXPECT_STREQ(Vicon.parameters().group("MANUFACTURER").parameter("VERSION_LABEL").valuesAsString()[0].c_str(), "2.4.0.91647h");

    // DATA
    for (size_t f = 0; f < 580; ++f){
        EXPECT_EQ(Vicon.data().frame(static_cast<int>(f)).points().nbPoints(), 51);
        for (size_t sf = 0; sf < 10; ++sf)
            EXPECT_EQ(Vicon.data().frame(static_cast<int>(f)).analogs().subframe(static_cast<int>(sf)).nbChannels(), 38);
    }
}


TEST(c3dFileIO, readQualisysC3D){
    ezc3d::c3d Qualisys("c3dTestFiles/Qualisys.c3d");

    // Header test
    // Generic stuff
    EXPECT_EQ(Qualisys.header().checksum(), 80);
    EXPECT_EQ(Qualisys.header().keyLabelPresent(), 0);
    EXPECT_EQ(Qualisys.header().firstBlockKeyLabel(), 0);
    EXPECT_EQ(Qualisys.header().fourCharPresent(), 0);
    EXPECT_EQ(Qualisys.header().emptyBlock1(), 0);
    EXPECT_EQ(Qualisys.header().emptyBlock2(), 0);
    EXPECT_EQ(Qualisys.header().emptyBlock3(), 0);
    EXPECT_EQ(Qualisys.header().emptyBlock4(), 0);

    // Point stuff
    EXPECT_EQ(Qualisys.header().nb3dPoints(), 55);
    EXPECT_EQ(Qualisys.header().nbMaxInterpGap(), 10);
    EXPECT_EQ(Qualisys.header().scaleFactor(), -1113841752);
    EXPECT_FLOAT_EQ(Qualisys.header().frameRate(), 200);

    // Analog stuff
    EXPECT_EQ(Qualisys.header().nbAnalogsMeasurement(), 690);
    EXPECT_EQ(Qualisys.header().nbAnalogByFrame(), 10);
    EXPECT_EQ(Qualisys.header().nbAnalogs(), 69);

    // Event stuff
    EXPECT_EQ(Qualisys.header().nbEvents(), 0);

    EXPECT_EQ(Qualisys.header().eventsTime().size(), 18);
    EXPECT_THROW(Qualisys.header().eventsTime(-1), std::invalid_argument);
    for (int e = 0; e < static_cast<int>(Qualisys.header().eventsTime().size()); ++e)
        EXPECT_FLOAT_EQ(Qualisys.header().eventsTime(e), 0);
    EXPECT_THROW(Qualisys.header().eventsTime(static_cast<int>(Qualisys.header().eventsTime().size())), std::invalid_argument);

    EXPECT_EQ(Qualisys.header().eventsLabel().size(), 18);
    EXPECT_THROW(Qualisys.header().eventsLabel(-1), std::invalid_argument);
    for (int e = 0; e < static_cast<int>(Qualisys.header().eventsLabel().size()); ++e)
        EXPECT_STREQ(Qualisys.header().eventsLabel(e).c_str(), "");
    EXPECT_THROW(Qualisys.header().eventsLabel(static_cast<int>(Qualisys.header().eventsLabel().size())), std::invalid_argument);

    EXPECT_EQ(Qualisys.header().eventsDisplay().size(), 9);
    EXPECT_THROW(Qualisys.header().eventsDisplay(-1), std::invalid_argument);
    for (int e = 0; e < static_cast<int>(Qualisys.header().eventsDisplay().size()); ++e)
        EXPECT_EQ(Qualisys.header().eventsDisplay(e), 257);
    EXPECT_THROW(Qualisys.header().eventsDisplay(static_cast<int>(Qualisys.header().eventsDisplay().size())), std::invalid_argument);


    EXPECT_EQ(Qualisys.header().firstFrame(), 704);
    EXPECT_EQ(Qualisys.header().lastFrame(), 1043);
    EXPECT_EQ(Qualisys.header().nbFrames(), 340);


    // Parameter tests
    EXPECT_EQ(Qualisys.parameters().checksum(), 80);
    EXPECT_EQ(Qualisys.parameters().groups().size(), 7);

    EXPECT_EQ(Qualisys.parameters().group("POINT").parameter("USED").type(), ezc3d::INT);
    EXPECT_EQ(Qualisys.parameters().group("POINT").parameter("USED").valuesAsInt().size(), 1);
    EXPECT_EQ(Qualisys.parameters().group("POINT").parameter("USED").valuesAsInt()[0], 55);
    EXPECT_EQ(Qualisys.parameters().group("POINT").parameter("SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(Qualisys.parameters().group("POINT").parameter("SCALE").valuesAsFloat().size(), 1);
    EXPECT_FLOAT_EQ(Qualisys.parameters().group("POINT").parameter("SCALE").valuesAsFloat()[0], static_cast<float>(-0.076232255));
    EXPECT_EQ(Qualisys.parameters().group("POINT").parameter("RATE").type(), ezc3d::FLOAT);
    EXPECT_EQ(Qualisys.parameters().group("POINT").parameter("RATE").valuesAsFloat().size(), 1);
    EXPECT_FLOAT_EQ(Qualisys.parameters().group("POINT").parameter("RATE").valuesAsFloat()[0], 200);
    EXPECT_EQ(Qualisys.parameters().group("POINT").parameter("FRAMES").valuesAsInt()[0], 340); // ignore because it changes if analog is present
    EXPECT_EQ(Qualisys.parameters().group("POINT").parameter("FRAMES").type(), ezc3d::INT);
    EXPECT_EQ(Qualisys.parameters().group("POINT").parameter("LABELS").type(), ezc3d::CHAR);
    EXPECT_EQ(Qualisys.parameters().group("POINT").parameter("LABELS").valuesAsString().size(), 55);
    EXPECT_EQ(Qualisys.parameters().group("POINT").parameter("DESCRIPTIONS").type(), ezc3d::CHAR);
    EXPECT_EQ(Qualisys.parameters().group("POINT").parameter("DESCRIPTIONS").valuesAsString().size(), 55);
    EXPECT_EQ(Qualisys.parameters().group("POINT").parameter("UNITS").type(), ezc3d::CHAR);
    EXPECT_EQ(Qualisys.parameters().group("POINT").parameter("UNITS").valuesAsString().size(), 1);

    EXPECT_EQ(Qualisys.parameters().group("ANALOG").parameter("USED").type(), ezc3d::INT);
    EXPECT_EQ(Qualisys.parameters().group("ANALOG").parameter("USED").valuesAsInt().size(), 1);
    EXPECT_EQ(Qualisys.parameters().group("ANALOG").parameter("USED").valuesAsInt()[0], 69);
    EXPECT_EQ(Qualisys.parameters().group("ANALOG").parameter("LABELS").type(), ezc3d::CHAR);
    EXPECT_EQ(Qualisys.parameters().group("ANALOG").parameter("LABELS").valuesAsString().size(), 69);
    EXPECT_EQ(Qualisys.parameters().group("ANALOG").parameter("DESCRIPTIONS").type(), ezc3d::CHAR);
    EXPECT_EQ(Qualisys.parameters().group("ANALOG").parameter("DESCRIPTIONS").valuesAsString().size(), 69);
    EXPECT_EQ(Qualisys.parameters().group("ANALOG").parameter("GEN_SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(Qualisys.parameters().group("ANALOG").parameter("GEN_SCALE").valuesAsFloat().size(), 1);
    EXPECT_EQ(Qualisys.parameters().group("ANALOG").parameter("GEN_SCALE").valuesAsFloat()[0], 1);
    EXPECT_EQ(Qualisys.parameters().group("ANALOG").parameter("SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(Qualisys.parameters().group("ANALOG").parameter("SCALE").valuesAsFloat().size(), 69);
    EXPECT_EQ(Qualisys.parameters().group("ANALOG").parameter("OFFSET").type(), ezc3d::INT);
    EXPECT_EQ(Qualisys.parameters().group("ANALOG").parameter("OFFSET").valuesAsInt().size(), 69);
    EXPECT_EQ(Qualisys.parameters().group("ANALOG").parameter("UNITS").type(), ezc3d::CHAR);
    EXPECT_EQ(Qualisys.parameters().group("ANALOG").parameter("UNITS").valuesAsString().size(), 69);
    EXPECT_EQ(Qualisys.parameters().group("ANALOG").parameter("RATE").type(), ezc3d::FLOAT);
    EXPECT_EQ(Qualisys.parameters().group("ANALOG").parameter("RATE").valuesAsFloat().size(), 1);
    EXPECT_FLOAT_EQ(Qualisys.parameters().group("ANALOG").parameter("RATE").valuesAsFloat()[0], 2000);

    EXPECT_EQ(Qualisys.parameters().group("FORCE_PLATFORM").parameter("USED").type(), ezc3d::INT);
    EXPECT_EQ(Qualisys.parameters().group("FORCE_PLATFORM").parameter("USED").valuesAsInt().size(), 1);
    EXPECT_EQ(Qualisys.parameters().group("FORCE_PLATFORM").parameter("USED").valuesAsInt()[0], 2);
    EXPECT_EQ(Qualisys.parameters().group("FORCE_PLATFORM").parameter("TYPE").type(), ezc3d::INT);
    EXPECT_EQ(Qualisys.parameters().group("FORCE_PLATFORM").parameter("TYPE").valuesAsInt().size(), 2);
    EXPECT_EQ(Qualisys.parameters().group("FORCE_PLATFORM").parameter("ZERO").type(), ezc3d::INT);
    EXPECT_EQ(Qualisys.parameters().group("FORCE_PLATFORM").parameter("ZERO").valuesAsInt().size(), 2);
    EXPECT_EQ(Qualisys.parameters().group("FORCE_PLATFORM").parameter("ZERO").valuesAsInt()[0], 0);
    EXPECT_EQ(Qualisys.parameters().group("FORCE_PLATFORM").parameter("ZERO").valuesAsInt()[1], 0);
    EXPECT_EQ(Qualisys.parameters().group("FORCE_PLATFORM").parameter("CORNERS").type(), ezc3d::FLOAT);
    EXPECT_EQ(Qualisys.parameters().group("FORCE_PLATFORM").parameter("CORNERS").valuesAsFloat().size(), 24);
    EXPECT_EQ(Qualisys.parameters().group("FORCE_PLATFORM").parameter("ORIGIN").type(), ezc3d::FLOAT);
    EXPECT_EQ(Qualisys.parameters().group("FORCE_PLATFORM").parameter("ORIGIN").valuesAsFloat().size(), 6);
    EXPECT_EQ(Qualisys.parameters().group("FORCE_PLATFORM").parameter("CHANNEL").type(), ezc3d::INT);
    EXPECT_EQ(Qualisys.parameters().group("FORCE_PLATFORM").parameter("CHANNEL").valuesAsInt().size(), 12);

    EXPECT_STREQ(Qualisys.parameters().group("MANUFACTURER").parameter("COMPANY").valuesAsString()[0].c_str(), "Qualisys");
    EXPECT_STREQ(Qualisys.parameters().group("MANUFACTURER").parameter("SOFTWARE").valuesAsString()[0].c_str(), "Qualisys Track Manager");
    EXPECT_EQ(Qualisys.parameters().group("MANUFACTURER").parameter("VERSION").valuesAsInt().size(), 3);
    EXPECT_EQ(Qualisys.parameters().group("MANUFACTURER").parameter("VERSION").valuesAsInt()[0], 2);
    EXPECT_EQ(Qualisys.parameters().group("MANUFACTURER").parameter("VERSION").valuesAsInt()[1], 17);
    EXPECT_EQ(Qualisys.parameters().group("MANUFACTURER").parameter("VERSION").valuesAsInt()[2], 3720);

    // DATA
    for (size_t f = 0; f < 340; ++f){
        EXPECT_EQ(Qualisys.data().frame(static_cast<int>(f)).points().nbPoints(), 55);
        for (size_t sf = 0; sf < 10; ++sf)
            EXPECT_EQ(Qualisys.data().frame(static_cast<int>(f)).analogs().subframe(static_cast<int>(sf)).nbChannels(), 69);
    }
}


TEST(c3dFileIO, readOptotrakC3D){
    ezc3d::c3d Optotrak("c3dTestFiles/Optotrak.c3d");

    // Header test
    // Generic stuff
    EXPECT_EQ(Optotrak.header().checksum(), 80);
    EXPECT_EQ(Optotrak.header().keyLabelPresent(), 0);
    EXPECT_EQ(Optotrak.header().firstBlockKeyLabel(), 0);
    EXPECT_EQ(Optotrak.header().fourCharPresent(), 12345);
    EXPECT_EQ(Optotrak.header().emptyBlock1(), 0);
    EXPECT_EQ(Optotrak.header().emptyBlock2(), 0);
    EXPECT_EQ(Optotrak.header().emptyBlock3(), 0);
    EXPECT_EQ(Optotrak.header().emptyBlock4(), 0);

    // Point stuff
    EXPECT_EQ(Optotrak.header().nb3dPoints(), 54);
    EXPECT_EQ(Optotrak.header().nbMaxInterpGap(), 0);
    EXPECT_EQ(Optotrak.header().scaleFactor(), -1057245329);
    EXPECT_FLOAT_EQ(Optotrak.header().frameRate(), 30);

    // Analog stuff
    EXPECT_EQ(Optotrak.header().nbAnalogsMeasurement(), 0);
    EXPECT_EQ(Optotrak.header().nbAnalogByFrame(), 0);
    EXPECT_EQ(Optotrak.header().nbAnalogs(), 0);

    // Event stuff
    EXPECT_EQ(Optotrak.header().nbEvents(), 0);

    EXPECT_EQ(Optotrak.header().eventsTime().size(), 18);
    EXPECT_THROW(Optotrak.header().eventsTime(-1), std::invalid_argument);
    for (int e = 0; e < static_cast<int>(Optotrak.header().eventsTime().size()); ++e)
        EXPECT_FLOAT_EQ(Optotrak.header().eventsTime(e), 0);
    EXPECT_THROW(Optotrak.header().eventsTime(static_cast<int>(Optotrak.header().eventsTime().size())), std::invalid_argument);

    EXPECT_EQ(Optotrak.header().eventsLabel().size(), 18);
    EXPECT_THROW(Optotrak.header().eventsLabel(-1), std::invalid_argument);
    for (int e = 0; e < static_cast<int>(Optotrak.header().eventsLabel().size()); ++e)
        EXPECT_STREQ(Optotrak.header().eventsLabel(e).c_str(), "");
    EXPECT_THROW(Optotrak.header().eventsLabel(static_cast<int>(Optotrak.header().eventsLabel().size())), std::invalid_argument);

    EXPECT_EQ(Optotrak.header().eventsDisplay().size(), 9);
    EXPECT_THROW(Optotrak.header().eventsDisplay(-1), std::invalid_argument);
    for (int e = 0; e < static_cast<int>(Optotrak.header().eventsDisplay().size()); ++e)
        EXPECT_EQ(Optotrak.header().eventsDisplay(e), 0);
    EXPECT_THROW(Optotrak.header().eventsDisplay(static_cast<int>(Optotrak.header().eventsDisplay().size())), std::invalid_argument);


    EXPECT_EQ(Optotrak.header().firstFrame(), 0);
    EXPECT_EQ(Optotrak.header().lastFrame(), 1148);
    EXPECT_EQ(Optotrak.header().nbFrames(), 1149);


    // Parameter tests
    EXPECT_EQ(Optotrak.parameters().checksum(), 80);
    EXPECT_EQ(Optotrak.parameters().groups().size(), 3);

    EXPECT_EQ(Optotrak.parameters().group("POINT").parameter("USED").type(), ezc3d::INT);
    EXPECT_EQ(Optotrak.parameters().group("POINT").parameter("USED").valuesAsInt().size(), 1);
    EXPECT_EQ(Optotrak.parameters().group("POINT").parameter("USED").valuesAsInt()[0], 54);
    EXPECT_EQ(Optotrak.parameters().group("POINT").parameter("SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(Optotrak.parameters().group("POINT").parameter("SCALE").valuesAsFloat().size(), 1);
    EXPECT_FLOAT_EQ(Optotrak.parameters().group("POINT").parameter("SCALE").valuesAsFloat()[0], static_cast<float>(-7.8661418));
    EXPECT_EQ(Optotrak.parameters().group("POINT").parameter("RATE").type(), ezc3d::FLOAT);
    EXPECT_EQ(Optotrak.parameters().group("POINT").parameter("RATE").valuesAsFloat().size(), 1);
    EXPECT_FLOAT_EQ(Optotrak.parameters().group("POINT").parameter("RATE").valuesAsFloat()[0], 30);
    EXPECT_EQ(Optotrak.parameters().group("POINT").parameter("FRAMES").valuesAsInt()[0], 1149); // ignore because it changes if analog is present
    EXPECT_EQ(Optotrak.parameters().group("POINT").parameter("FRAMES").type(), ezc3d::INT);
    EXPECT_EQ(Optotrak.parameters().group("POINT").parameter("LABELS").type(), ezc3d::CHAR);
    EXPECT_EQ(Optotrak.parameters().group("POINT").parameter("LABELS").valuesAsString().size(), 54);
    EXPECT_EQ(Optotrak.parameters().group("POINT").parameter("DESCRIPTIONS").type(), ezc3d::CHAR);
    EXPECT_EQ(Optotrak.parameters().group("POINT").parameter("DESCRIPTIONS").valuesAsString().size(), 54);
    EXPECT_EQ(Optotrak.parameters().group("POINT").parameter("UNITS").type(), ezc3d::CHAR);
    EXPECT_EQ(Optotrak.parameters().group("POINT").parameter("UNITS").valuesAsString().size(), 54);

    EXPECT_EQ(Optotrak.parameters().group("FORCE_PLATFORM").parameter("USED").type(), ezc3d::INT);
    EXPECT_EQ(Optotrak.parameters().group("FORCE_PLATFORM").parameter("USED").valuesAsInt().size(), 1);
    EXPECT_EQ(Optotrak.parameters().group("FORCE_PLATFORM").parameter("USED").valuesAsInt()[0], 0);

    // DATA
    for (size_t f = 0; f < 1149; ++f)
        EXPECT_EQ(Optotrak.data().frame(static_cast<int>(f)).points().nbPoints(), 54);
}

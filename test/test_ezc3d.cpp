#include <iostream>
#include <gtest/gtest.h>

#include "ezc3d_all.h"

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

    float pointFrameRate = -1;
    float analogFrameRate = -1;
    size_t nFrames = SIZE_MAX;
    size_t nPoints = SIZE_MAX;
    float nSubframes = -1;
    float dummyForByteAlignment = -1; // Because of float precision, 4 bytes must be padded here due to the odd numer of float variables
    std::vector<std::string> pointNames;

    size_t nAnalogs = SIZE_MAX;
    std::vector<std::string> analogNames;
};

void fillC3D(c3dTestStruct& c3dStruc, bool withPoints, bool withAnalogs){
    // Setup some variables
    if (withPoints){
        c3dStruc.pointNames = {"point1", "point2", "point3"};
        c3dStruc.nPoints = c3dStruc.pointNames.size();
        // Add points to the new c3d
        for (size_t m = 0; m < c3dStruc.nPoints; ++m)
            c3dStruc.c3d.point(c3dStruc.pointNames[m]);
    }

    c3dStruc.analogNames.clear();
    c3dStruc.nAnalogs = SIZE_MAX;
    if (withAnalogs){
        c3dStruc.analogNames = {"analog1", "analog2", "analog3"};
        c3dStruc.nAnalogs = c3dStruc.analogNames.size();
        // Add analog to the new c3d
        for (size_t a = 0; a < c3dStruc.nAnalogs; ++a)
            c3dStruc.c3d.analog(c3dStruc.analogNames[a]);
    }

    c3dStruc.nFrames = 10;
    c3dStruc.pointFrameRate = 100;
    if (withPoints){
        ezc3d::ParametersNS::GroupNS::Parameter pointRate("RATE");
        pointRate.set(std::vector<double>()={
                static_cast<double>(c3dStruc.pointFrameRate)});
        c3dStruc.c3d.parameter("POINT", pointRate);
    }
    if (withAnalogs){
        c3dStruc.analogFrameRate = 1000;
        c3dStruc.nSubframes = c3dStruc.analogFrameRate / c3dStruc.pointFrameRate;

        ezc3d::ParametersNS::GroupNS::Parameter analogRate("RATE");
        analogRate.set(std::vector<double>()={
                static_cast<double>(c3dStruc.analogFrameRate)});
        c3dStruc.c3d.parameter("ANALOG", analogRate);
    }
    for (size_t f = 0; f < c3dStruc.nFrames; ++f){
        ezc3d::DataNS::Frame frame;

        ezc3d::DataNS::Points3dNS::Points pts;
        if (withPoints){
            for (size_t m = 0; m < c3dStruc.nPoints; ++m){
                ezc3d::DataNS::Points3dNS::Point pt;
                // Generate some random data
                pt.x(static_cast<double>(2*f+3*m+1) / static_cast<double>(7.0));
                pt.y(static_cast<double>(2*f+3*m+2) / static_cast<double>(7.0));
                pt.z(static_cast<double>(2*f+3*m+3) / static_cast<double>(7.0));
                pts.point(pt);
            }
        }

        ezc3d::DataNS::AnalogsNS::Analogs analogs;
        if (withAnalogs){
            for (size_t sf = 0; sf < c3dStruc.nSubframes; ++sf){
                ezc3d::DataNS::AnalogsNS::SubFrame subframes;
                for (size_t c = 0; c < c3dStruc.nAnalogs; ++c){
                    ezc3d::DataNS::AnalogsNS::Channel channel;
                    channel.data(static_cast<float>(2*f+3*sf+4*c+1) / static_cast<float>(7.0)); // Generate random data
                    subframes.channel(channel);
                }
                analogs.subframe(subframes);
            }
        }

        if (withPoints && withAnalogs)
            frame.add(pts, analogs);
        else if (withPoints)
            frame.add(pts);
        else if (withAnalogs)
            frame.add(analogs);
        c3dStruc.c3d.frame(frame);
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
        for (size_t e = 0; e < new_c3d.header().eventsTime().size(); ++e)
            EXPECT_FLOAT_EQ(new_c3d.header().eventsTime(e), 0);
        EXPECT_THROW(new_c3d.header().eventsTime(new_c3d.header().eventsTime().size()), std::out_of_range);

        EXPECT_EQ(new_c3d.header().eventsLabel().size(), 18);
        for (size_t e = 0; e < new_c3d.header().eventsTime().size(); ++e)
            EXPECT_STREQ(new_c3d.header().eventsLabel(e).c_str(), "");
        EXPECT_THROW(new_c3d.header().eventsLabel(new_c3d.header().eventsLabel().size()), std::out_of_range);

        EXPECT_EQ(new_c3d.header().eventsDisplay().size(), 9);
        for (size_t e = 0; e < new_c3d.header().eventsDisplay().size(); ++e)
            EXPECT_EQ(new_c3d.header().eventsDisplay(e), 0);
        EXPECT_THROW(new_c3d.header().eventsDisplay(new_c3d.header().eventsDisplay().size()), std::out_of_range);

    }

    if (type == HEADER_TYPE::ALL){
        EXPECT_EQ(new_c3d.header().firstFrame(), 0);
        EXPECT_EQ(new_c3d.header().lastFrame(), 0);
        EXPECT_EQ(new_c3d.header().nbFrames(), 0);
    }
}

void compareHeader(const ezc3d::c3d& c3d1, const ezc3d::c3d& c3d2){
    EXPECT_EQ(c3d1.header().nbFrames(), c3d2.header().nbFrames());
    EXPECT_FLOAT_EQ(c3d1.header().frameRate(), c3d2.header().frameRate());
    EXPECT_EQ(c3d1.header().firstFrame(), c3d2.header().firstFrame());
    EXPECT_EQ(c3d1.header().lastFrame(), c3d2.header().lastFrame());
    EXPECT_EQ(c3d1.header().nb3dPoints(), c3d2.header().nb3dPoints());
    EXPECT_EQ(c3d1.header().nbAnalogs(), c3d2.header().nbAnalogs());
    EXPECT_EQ(c3d1.header().nbAnalogByFrame(), c3d2.header().nbAnalogByFrame());
    EXPECT_EQ(c3d1.header().nbAnalogsMeasurement(), c3d2.header().nbAnalogsMeasurement());
    EXPECT_EQ(c3d1.header().nbEvents(), c3d2.header().nbEvents());
    for (size_t e = 0; e<c3d1.header().nbEvents(); ++e){
        EXPECT_FLOAT_EQ(c3d1.header().eventsTime(e), c3d2.header().eventsTime(e));
        EXPECT_EQ(c3d1.header().eventsDisplay(e), c3d2.header().eventsDisplay(e));
        EXPECT_STREQ(c3d1.header().eventsLabel(e).c_str(), c3d2.header().eventsLabel(e).c_str());
    }
}

void compareData(const ezc3d::c3d& c3d1, const ezc3d::c3d& c3d2
                 , bool skipResidual = false){
    // All the data should be the same
    for (size_t f=0; f<c3d1.header().nbFrames(); ++f){
        for (size_t p=0; p<c3d1.header().nb3dPoints(); ++p){
            if (c3d1.data().frame(f).points().point(p).residual() < 0) {
                ASSERT_TRUE(std::isnan(c3d1.data().frame(f).points().point(p).x()));
                ASSERT_TRUE(std::isnan(c3d1.data().frame(f).points().point(p).y()));
                ASSERT_TRUE(std::isnan(c3d1.data().frame(f).points().point(p).z()));

                ASSERT_TRUE(std::isnan(c3d2.data().frame(f).points().point(p).x()));
                ASSERT_TRUE(std::isnan(c3d2.data().frame(f).points().point(p).y()));
                ASSERT_TRUE(std::isnan(c3d2.data().frame(f).points().point(p).z()));
            }
            else {
                EXPECT_FLOAT_EQ(c3d1.data().frame(f).points().point(p).x(), c3d2.data().frame(f).points().point(p).x());
                EXPECT_FLOAT_EQ(c3d1.data().frame(f).points().point(p).y(), c3d2.data().frame(f).points().point(p).y());
                EXPECT_FLOAT_EQ(c3d1.data().frame(f).points().point(p).z(), c3d2.data().frame(f).points().point(p).z());
            }
            if (!skipResidual) {
                EXPECT_FLOAT_EQ(c3d1.data().frame(f).points().point(p).residual(), c3d2.data().frame(f).points().point(p).residual());
                std::vector<bool> cameraMasks1(c3d1.data().frame(f).points().point(p).cameraMask());
                std::vector<bool> cameraMasks2(c3d1.data().frame(f).points().point(p).cameraMask());
                EXPECT_EQ(cameraMasks1.size(), cameraMasks2.size());
                for (size_t cam = 0; cam < cameraMasks1.size(); ++cam){
                    EXPECT_EQ(cameraMasks1[cam], cameraMasks2[cam]);
                }
            }
        }
        for (size_t sf=0; sf<c3d1.data().frame(f).analogs().nbSubframes(); ++sf){
            for (size_t c=0; c<c3d1.header().nbAnalogByFrame(); ++c){
                EXPECT_FLOAT_EQ(c3d1.data().frame(f).analogs().subframe(sf).channel(c).data(), c3d2.data().frame(f).analogs().subframe(sf).channel(c).data());
            }
        }
    }
}

void defaultParametersTest(const ezc3d::c3d& new_c3d, PARAMETER_TYPE type){
    if (type == PARAMETER_TYPE::HEADER){
        EXPECT_EQ(new_c3d.parameters().checksum(), 80);
        EXPECT_EQ(new_c3d.parameters().nbGroups(), 3);
    } else if (type == PARAMETER_TYPE::POINT){
        EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("USED").type(), ezc3d::INT);
        EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("USED").valuesAsInt().size(), 1);
        EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("USED").valuesAsInt()[0], 0);
        EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("SCALE").type(), ezc3d::FLOAT);
        EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("SCALE").valuesAsDouble().size(), 1);
        EXPECT_FLOAT_EQ(new_c3d.parameters().group("POINT").parameter("SCALE").valuesAsDouble()[0], -1);
        EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("RATE").type(), ezc3d::FLOAT);
        EXPECT_EQ(new_c3d.parameters().group("POINT").parameter("RATE").valuesAsDouble().size(), 1);
        EXPECT_FLOAT_EQ(new_c3d.parameters().group("POINT").parameter("RATE").valuesAsDouble()[0], 0);
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
        EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("GEN_SCALE").type(), ezc3d::FLOAT);
        EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("GEN_SCALE").valuesAsDouble().size(), 1);
        EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("GEN_SCALE").valuesAsDouble()[0], 1);
        EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("SCALE").type(), ezc3d::FLOAT);
        EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("SCALE").valuesAsDouble().size(), 0);
        EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("OFFSET").type(), ezc3d::INT);
        EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("OFFSET").valuesAsInt().size(), 0);
        EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("UNITS").type(), ezc3d::CHAR);
        EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("UNITS").valuesAsString().size(), 0);
        EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("RATE").type(), ezc3d::FLOAT);
        EXPECT_EQ(new_c3d.parameters().group("ANALOG").parameter("RATE").valuesAsDouble().size(), 1);
        EXPECT_FLOAT_EQ(new_c3d.parameters().group("ANALOG").parameter("RATE").valuesAsDouble()[0], 0);
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
        EXPECT_EQ(new_c3d.parameters().group("FORCE_PLATFORM").parameter("CORNERS").valuesAsDouble().size(), 0);
        EXPECT_EQ(new_c3d.parameters().group("FORCE_PLATFORM").parameter("ORIGIN").type(), ezc3d::FLOAT);
        EXPECT_EQ(new_c3d.parameters().group("FORCE_PLATFORM").parameter("ORIGIN").valuesAsDouble().size(), 0);
        EXPECT_EQ(new_c3d.parameters().group("FORCE_PLATFORM").parameter("CHANNEL").type(), ezc3d::INT);
        EXPECT_EQ(new_c3d.parameters().group("FORCE_PLATFORM").parameter("CHANNEL").valuesAsInt().size(), 0);
        EXPECT_EQ(new_c3d.parameters().group("FORCE_PLATFORM").parameter("CAL_MATRIX").type(), ezc3d::FLOAT);
        EXPECT_EQ(new_c3d.parameters().group("FORCE_PLATFORM").parameter("CAL_MATRIX").valuesAsDouble().size(), 0);
    }
}

TEST(String, unittest){
    EXPECT_STREQ(ezc3d::toUpper("toUpper").c_str(), "TOUPPER");
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
    c3d_file.seekp(static_cast<int>(256*ezc3d::DATA_TYPE::WORD*(new_c3d.header().parametersAddress()-1) + 1)); // move to the parameter checksum
    int checksum(0x0);
    c3d_file.write(reinterpret_cast<const char*>(&checksum), ezc3d::BYTE);
    c3d_file.close();

    // Read the erroneous file
    EXPECT_THROW(ezc3d::c3d new_c3d("temporary.c3d"), std::ios_base::failure);

    // If a 0 is also on the byte before the checksum, this is a Qualisys C3D and should be read even if checksum si wrong
    std::ofstream c3d_file2(savePath.c_str(), std::ofstream::in);
    c3d_file2.seekp(static_cast<int>(256*ezc3d::DATA_TYPE::WORD*(new_c3d.header().parametersAddress()-1))); // move to the parameter checksum
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
    c3d_file.seekp(static_cast<int>(256*ezc3d::DATA_TYPE::WORD*(new_c3d.header().parametersAddress()-1))); // move to the parameter checksum
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

    // Get an erroneous group
    EXPECT_THROW(new_c3d.c3d.parameters().group("ThisIsNotARealGroup"), std::invalid_argument);
    EXPECT_THROW(new_c3d.c3d.parameters().group(3), std::out_of_range);
    const auto& allGroups(new_c3d.c3d.parameters().groups());
    EXPECT_EQ(allGroups.size(), 3);

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
    EXPECT_THROW(new_c3d.c3d.parameter("POINT", p), std::invalid_argument);
    p.name("EmptyParam");
    EXPECT_THROW(new_c3d.c3d.parameter("POINT", p), std::runtime_error);

    // Get an erroneous parameter
    EXPECT_THROW(new_c3d.c3d.parameters().group("POINT").parameter("ThisIsNotARealParamter"), std::invalid_argument);

    // Create a new group
    ezc3d::ParametersNS::GroupNS::Parameter p2;
    p2.name("NewParam");
    p2.set(std::vector<int>());
    EXPECT_NO_THROW(new_c3d.c3d.parameter("ThisIsANewRealGroup", p2));
    EXPECT_EQ(new_c3d.c3d.parameters().group("ThisIsANewRealGroup").parameter("NewParam").type(), ezc3d::INT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ThisIsANewRealGroup").parameter("NewParam").valuesAsInt().size(), 0);

    // Get an out of range parameter
    size_t nPointParams(new_c3d.c3d.parameters().group("POINT").nbParameters());
    EXPECT_THROW(new_c3d.c3d.parameters().group("POINT").parameter(nPointParams), std::out_of_range);
    EXPECT_THROW(new_c3d.c3d.parameters().group("POINT").parameterIdx("ThisIsNotARealParameter"), std::invalid_argument);

    // Get all the parameters
    const auto& params(new_c3d.c3d.parameters().group("ThisIsANewRealGroup").parameters());
    EXPECT_EQ(params.size(), 1);

    // Reading an empty parameter is actually type irrelevant
    EXPECT_NO_THROW(p.valuesAsByte());
    EXPECT_NO_THROW(p.valuesAsInt());
    EXPECT_NO_THROW(p.valuesAsDouble());
    EXPECT_NO_THROW(p.valuesAsString());

    {
        // There is no pNonEmptyByte, since the only way to declare a byte
        // is from the .c3d file itself. Otherwise it is always an int

        ezc3d::ParametersNS::GroupNS::Parameter pNonEmptyInt;
        pNonEmptyInt.name("NewIntParam");
        pNonEmptyInt.set(std::vector<int>(1));
        EXPECT_THROW(pNonEmptyInt.valuesAsByte(), std::invalid_argument);
        EXPECT_NO_THROW(pNonEmptyInt.valuesAsInt());
        EXPECT_THROW(pNonEmptyInt.valuesAsDouble(), std::invalid_argument);
        EXPECT_THROW(pNonEmptyInt.valuesAsString(), std::invalid_argument);

        ezc3d::ParametersNS::GroupNS::Parameter pNonEmptyFloat;
        pNonEmptyFloat.name("NewFloatParam");
        pNonEmptyFloat.set(std::vector<double>(1.));
        EXPECT_THROW(pNonEmptyFloat.valuesAsByte(), std::invalid_argument);
        EXPECT_THROW(pNonEmptyFloat.valuesAsInt(), std::invalid_argument);
        EXPECT_NO_THROW(pNonEmptyFloat.valuesAsDouble());
        EXPECT_THROW(pNonEmptyFloat.valuesAsString(), std::invalid_argument);

        ezc3d::ParametersNS::GroupNS::Parameter pNonEmptyChar;
        pNonEmptyChar.name("NewCharParam");
        pNonEmptyChar.set(std::vector<std::string>(1));
        EXPECT_THROW(pNonEmptyChar.valuesAsByte(), std::invalid_argument);
        EXPECT_THROW(pNonEmptyChar.valuesAsInt(), std::invalid_argument);
        EXPECT_THROW(pNonEmptyChar.valuesAsDouble(), std::invalid_argument);
        EXPECT_NO_THROW(pNonEmptyChar.valuesAsString());
    }

    // Lock and unlock a parameter
    EXPECT_EQ(p.isLocked(), false);
    p.lock();
    EXPECT_EQ(p.isLocked(), true);
    p.unlock();
    EXPECT_EQ(p.isLocked(), false);

    // Fill the parameter improperly
    EXPECT_THROW(p.set(std::vector<int>()={}, {1}), std::range_error);
    EXPECT_THROW(p.set(std::vector<int>()={1}, {0}), std::range_error);
    EXPECT_THROW(p.set(std::vector<double>()={}, {1}), std::range_error);
    EXPECT_THROW(p.set(std::vector<double>()={1.}, {0}), std::range_error);
    EXPECT_THROW(p.set(std::vector<std::string>()={}, {1}), std::range_error);
    EXPECT_THROW(p.set(std::vector<std::string>()={""}, {0}), std::range_error);
    p.dimension();

    // Add twice the same group (That should not be needed for a user API
    // since he should use new_c3d.c3d.addParameter() )
    {
        ezc3d::ParametersNS::Parameters params;
        ezc3d::ParametersNS::GroupNS::Group groupToBeAddedTwice;
        ezc3d::ParametersNS::GroupNS::Parameter p("UselessParameter");
        p.set(std::vector<int>()={});
        groupToBeAddedTwice.parameter(p);
        EXPECT_NO_THROW(params.group(groupToBeAddedTwice));
        EXPECT_NO_THROW(params.group(groupToBeAddedTwice));
    }
}

TEST(c3dModifier, groupMetaData){
    // Create an empty c3d
    c3dTestStruct new_c3d;
    fillC3D(new_c3d, true, true);

    EXPECT_STREQ(new_c3d.c3d.parameters().group("POINT").description().c_str(),
                 "");
    EXPECT_FALSE(new_c3d.c3d.parameters().group("POINT").isLocked());
    new_c3d.c3d.setGroupMetadata("POINT", "My new description", true);
    EXPECT_STREQ(new_c3d.c3d.parameters().group("POINT").description().c_str(),
                 "My new description");
    EXPECT_TRUE(new_c3d.c3d.parameters().group("POINT").isLocked());
}

TEST(c3dModifier, addPoints) {
    // Create an empty c3d
    c3dTestStruct new_c3d;
    fillC3D(new_c3d, true, false);

    // HEADER
    // Things that should remain as default
    defaultHeaderTest(new_c3d.c3d, HEADER_TYPE::ANALOG_AND_EVENT);

    // Things that should have change
    EXPECT_EQ(new_c3d.c3d.header().nb3dPoints(), new_c3d.nPoints);
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
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("USED").valuesAsInt()[0], new_c3d.nPoints);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("SCALE").valuesAsDouble().size(), 1);
    EXPECT_FLOAT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("SCALE").valuesAsDouble()[0], -1);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("RATE").type(), ezc3d::FLOAT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("RATE").valuesAsDouble().size(), 1);
    EXPECT_FLOAT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("RATE").valuesAsDouble()[0], new_c3d.pointFrameRate);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("FRAMES").type(), ezc3d::INT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("FRAMES").valuesAsInt().size(), 1);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("FRAMES").valuesAsInt()[0], new_c3d.nFrames);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("LABELS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("LABELS").valuesAsString().size(), new_c3d.nPoints);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("DESCRIPTIONS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("DESCRIPTIONS").valuesAsString().size(), new_c3d.nPoints);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("UNITS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("UNITS").valuesAsString().size(), new_c3d.nPoints);
    for (size_t m = 0; m < new_c3d.nPoints; ++m){
        EXPECT_STREQ(new_c3d.c3d.parameters().group("POINT").parameter("LABELS").valuesAsString()[m].c_str(),new_c3d.pointNames[m].c_str());
        EXPECT_STREQ(new_c3d.c3d.parameters().group("POINT").parameter("DESCRIPTIONS").valuesAsString()[m].c_str(), "");
        EXPECT_STREQ(new_c3d.c3d.parameters().group("POINT").parameter("UNITS").valuesAsString()[m].c_str(), "mm");
    }

    // DATA
    for (size_t f = 0; f < new_c3d.nFrames; ++f){
        ASSERT_EQ(new_c3d.c3d.data().frame(f).points().isEmpty(), false);
        for (size_t m = 0; m < new_c3d.nPoints; ++m){
            size_t pointIdx(new_c3d.c3d.pointIdx(new_c3d.pointNames[m]));
            EXPECT_DOUBLE_EQ(new_c3d.c3d.data().frame(f).points().point(pointIdx).x(), static_cast<double>(2*f+3*m+1) / static_cast<double>(7.0));
            EXPECT_DOUBLE_EQ(new_c3d.c3d.data().frame(f).points().point(pointIdx).y(), static_cast<double>(2*f+3*m+2) / static_cast<double>(7.0));
            EXPECT_DOUBLE_EQ(new_c3d.c3d.data().frame(f).points().point(pointIdx).z(), static_cast<double>(2*f+3*m+3) / static_cast<double>(7.0));
            EXPECT_DOUBLE_EQ(new_c3d.c3d.data().frame(f).points().point(pointIdx).residual(), 0);
        }
        ASSERT_EQ(new_c3d.c3d.data().frame(f).analogs().isEmpty(), true);
    }

    // Add frame with a new point with not enough frames
    std::vector<ezc3d::DataNS::Frame> new_frames;
    EXPECT_THROW(new_c3d.c3d.point("uselessPoint", new_frames), std::invalid_argument);
    for (size_t f = 0; f < new_c3d.c3d.data().nbFrames() - 1; ++f)
        new_frames.push_back(ezc3d::DataNS::Frame());
    EXPECT_THROW(new_c3d.c3d.point("uselessPoint", new_frames), std::invalid_argument);

    // Not enough points
    new_frames.push_back(ezc3d::DataNS::Frame());
    EXPECT_THROW(new_c3d.c3d.point("uselessPoint", new_frames), std::invalid_argument);

    // Try adding an already existing point
    for (size_t f = 0; f < new_c3d.c3d.data().nbFrames(); ++f){
        ezc3d::DataNS::Points3dNS::Points pts;
        ezc3d::DataNS::Points3dNS::Point pt(new_c3d.c3d.data().frame(f).points().point(0));
        pts.point(pt);
        new_frames[f].add(pts);
    }
    EXPECT_THROW(new_c3d.c3d.point(new_c3d.pointNames, new_frames), std::invalid_argument);

    // Adding it properly
    for (size_t f = 0; f < new_c3d.c3d.data().nbFrames(); ++f){
        ezc3d::DataNS::Points3dNS::Points pts;
        ezc3d::DataNS::Points3dNS::Point pt;
        pts.point(pt);
        new_frames[f].add(pts);
    }
    EXPECT_NO_THROW(new_c3d.c3d.point("goodPoint", new_frames));

}


TEST(c3dModifier, specificPoint){
    // Create an empty c3d
    c3dTestStruct new_c3d;
    fillC3D(new_c3d, true, false);

    // Create unused points
    ezc3d::DataNS::Points3dNS::Point unusedPt;
    unusedPt.set(1.1, 2.2, 3.3, 4.4);
    EXPECT_DOUBLE_EQ(unusedPt.x(), 1.1);
    EXPECT_DOUBLE_EQ(unusedPt.y(), 2.2);
    EXPECT_DOUBLE_EQ(unusedPt.z(), 3.3);
    EXPECT_DOUBLE_EQ(unusedPt.residual(), 4.4);
    unusedPt.set(5.5, 6.6, 7.7);
    EXPECT_DOUBLE_EQ(unusedPt.x(), 5.5);
    EXPECT_DOUBLE_EQ(unusedPt.y(), 6.6);
    EXPECT_DOUBLE_EQ(unusedPt.z(), 7.7);
    EXPECT_DOUBLE_EQ(unusedPt.residual(), 0.0);
    unusedPt.cameraMask({true, false});
    EXPECT_EQ(unusedPt.cameraMask().size(), 2);
    EXPECT_TRUE(unusedPt.cameraMask()[0]);
    EXPECT_FALSE(unusedPt.cameraMask()[1]);
    EXPECT_FALSE(unusedPt.isEmpty());
    unusedPt.set(NAN, NAN, NAN);
    EXPECT_TRUE(unusedPt.isEmpty());
    unusedPt.set(0.0, 0.0, 0.0);
    EXPECT_TRUE(unusedPt.isEmpty());

    // test replacing points
    for (size_t f = 0; f < new_c3d.nFrames; ++f){
        ezc3d::DataNS::Frame frame;

        ezc3d::DataNS::Points3dNS::Points pts(new_c3d.c3d.data().frame(f).points());
        for (size_t m = 0; m < new_c3d.nPoints; ++m){
            ezc3d::DataNS::Points3dNS::Point pt;
            // Generate some random data
            pt.x(static_cast<double>(4*f+7*m+5) / static_cast<double>(13.0));
            pt.y(static_cast<double>(4*f+7*m+6) / static_cast<double>(13.0));
            pt.z(static_cast<double>(4*f+7*m+7) / static_cast<double>(13.0));
            pts.point(pt, m);
        }

        frame.add(pts);
        new_c3d.c3d.frame(frame, f);
    }

    // failed test replacing a point
    {
        ezc3d::DataNS::Points3dNS::Point ptToBeReplaced;
        ezc3d::DataNS::Points3dNS::Point ptToReplace;

        ezc3d::DataNS::Points3dNS::Points pts;
        pts.point(ptToBeReplaced);
        size_t previousNbPoints(pts.nbPoints());
        EXPECT_NO_THROW(pts.point(ptToReplace, pts.nbPoints()+2));
        EXPECT_EQ(pts.nbPoints(), previousNbPoints + 3);
        EXPECT_NO_THROW(pts.point(ptToReplace, 0));
    }

    // Things that should have change
    EXPECT_EQ(new_c3d.c3d.header().nb3dPoints(), new_c3d.nPoints);
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
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("USED").valuesAsInt()[0], new_c3d.nPoints);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("SCALE").valuesAsDouble().size(), 1);
    EXPECT_FLOAT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("SCALE").valuesAsDouble()[0], -1);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("RATE").type(), ezc3d::FLOAT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("RATE").valuesAsDouble().size(), 1);
    EXPECT_FLOAT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("RATE").valuesAsDouble()[0], new_c3d.pointFrameRate);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("FRAMES").type(), ezc3d::INT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("FRAMES").valuesAsInt().size(), 1);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("FRAMES").valuesAsInt()[0], new_c3d.nFrames);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("LABELS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("LABELS").valuesAsString().size(), new_c3d.nPoints);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("DESCRIPTIONS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("DESCRIPTIONS").valuesAsString().size(), new_c3d.nPoints);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("UNITS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("UNITS").valuesAsString().size(), new_c3d.nPoints);
    for (size_t m = 0; m < new_c3d.nPoints; ++m){
        EXPECT_STREQ(new_c3d.c3d.parameters().group("POINT").parameter("LABELS").valuesAsString()[m].c_str(),new_c3d.pointNames[m].c_str());
        EXPECT_STREQ(new_c3d.c3d.parameters().group("POINT").parameter("DESCRIPTIONS").valuesAsString()[m].c_str(), "");
        EXPECT_STREQ(new_c3d.c3d.parameters().group("POINT").parameter("UNITS").valuesAsString()[m].c_str(), "mm");
    }

    // DATA
    for (size_t f = 0; f < new_c3d.nFrames; ++f){
        for (size_t m = 0; m < new_c3d.nPoints; ++m){
            size_t pointIdx(new_c3d.c3d.pointIdx(new_c3d.pointNames[m]));
            EXPECT_DOUBLE_EQ(new_c3d.c3d.data().frame(f).points().point(pointIdx).x(), static_cast<double>(4*f+7*m+5) / static_cast<double>(13.0));
            EXPECT_DOUBLE_EQ(new_c3d.c3d.data().frame(f).points().point(pointIdx).y(), static_cast<double>(4*f+7*m+6) / static_cast<double>(13.0));
            EXPECT_DOUBLE_EQ(new_c3d.c3d.data().frame(f).points().point(pointIdx).z(), static_cast<double>(4*f+7*m+7) / static_cast<double>(13.0));
            EXPECT_DOUBLE_EQ(new_c3d.c3d.data().frame(f).points().point(pointIdx).residual(), 0);
        }
    }

    // Access a non-existant point
    EXPECT_THROW(new_c3d.c3d.data().frame(0).points().point(new_c3d.nPoints), std::out_of_range);

    // Test for removing space at the end of a label
    new_c3d.c3d.point("PointNameWithSpaceAtTheEnd ");
    new_c3d.nPoints += 1;
    new_c3d.pointNames.push_back("PointNameWithSpaceAtTheEnd");
    EXPECT_STREQ(new_c3d.c3d.parameters().group("POINT").parameter("LABELS").valuesAsString()[new_c3d.nPoints - 1].c_str(), "PointNameWithSpaceAtTheEnd");

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
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("GEN_SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("GEN_SCALE").valuesAsDouble().size(), 1);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("GEN_SCALE").valuesAsDouble()[0], 1);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("SCALE").valuesAsDouble().size(), new_c3d.nAnalogs);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("OFFSET").type(), ezc3d::INT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("OFFSET").valuesAsInt().size(), new_c3d.nAnalogs);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("UNITS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("UNITS").valuesAsString().size(), new_c3d.nAnalogs);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("RATE").type(), ezc3d::FLOAT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("RATE").valuesAsDouble().size(), 1);
    EXPECT_FLOAT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("RATE").valuesAsDouble()[0], new_c3d.analogFrameRate);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("FORMAT").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("FORMAT").valuesAsString().size(), 0);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("BITS").type(), ezc3d::INT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("BITS").valuesAsInt().size(), 0);

    for (size_t a = 0; a < new_c3d.nAnalogs; ++a){
        EXPECT_STREQ(new_c3d.c3d.parameters().group("ANALOG").parameter("LABELS").valuesAsString()[a].c_str(), new_c3d.analogNames[a].c_str());
        EXPECT_STREQ(new_c3d.c3d.parameters().group("ANALOG").parameter("DESCRIPTIONS").valuesAsString()[a].c_str(), "");
        EXPECT_FLOAT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("SCALE").valuesAsDouble()[a], 1);
        EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("OFFSET").valuesAsInt()[a], 0);
        EXPECT_STREQ(new_c3d.c3d.parameters().group("ANALOG").parameter("UNITS").valuesAsString()[a].c_str(), "");
    }

    // DATA
    for (size_t f = 0; f < new_c3d.nFrames; ++f){
        ASSERT_EQ(new_c3d.c3d.data().frame(f).points().isEmpty(), true);
        ASSERT_EQ(new_c3d.c3d.data().frame(f).analogs().isEmpty(), false);
        for (size_t sf = 0; sf < new_c3d.nSubframes; ++sf)
            for (size_t c = 0; c < new_c3d.nAnalogs; ++c)
                EXPECT_FLOAT_EQ(new_c3d.c3d.data().frame(f).analogs().subframe(sf).channel(c).data(),
                                static_cast<double>(2*f+3*sf+4*c+1) / static_cast<double>(7.0));
    }
}

TEST(c3dModifier, removeAnalog){
    // Remove some analog parameter, while making sure the C3D is still valid
    // TODO when removing analog is ready
}

TEST(c3dModifier, specificAnalog){
    // Create an empty c3d
    c3dTestStruct new_c3d;
    fillC3D(new_c3d, false, true);

    // Test for removing space at the end of a label
    new_c3d.c3d.analog("AnalogNameWithSpaceAtTheEnd ");
    new_c3d.nAnalogs += 1;
    new_c3d.analogNames.push_back("AnalogNameWithSpaceAtTheEnd");
    EXPECT_STREQ(new_c3d.c3d.parameters().group("ANALOG").parameter("LABELS").valuesAsString()[new_c3d.nAnalogs - 1].c_str(), "AnalogNameWithSpaceAtTheEnd");

    // Add analog by frames
    std::vector<ezc3d::DataNS::Frame> frames;
    // Wrong number of frames
    EXPECT_THROW(new_c3d.c3d.analog("uselessChannel", frames), std::invalid_argument);

    // Test if analog is empty
    ezc3d::DataNS::AnalogsNS::Channel channelToFill;
    channelToFill.data(0.0);
    EXPECT_TRUE(channelToFill.isEmpty());
    channelToFill.data(1.0);
    EXPECT_FALSE(channelToFill.isEmpty());

    // Wrong number of subframes
    frames.resize(new_c3d.nFrames);
    EXPECT_THROW(new_c3d.c3d.analog("uselessChannel", frames), std::invalid_argument);

    // Wrong number of channels
    for (size_t f = 0; f < new_c3d.nFrames; ++f){
        ezc3d::DataNS::Frame frame;
        ezc3d::DataNS::AnalogsNS::Analogs analogs;
        for (size_t sf = 0; sf < new_c3d.nSubframes; ++sf)
            analogs.subframe(ezc3d::DataNS::AnalogsNS::SubFrame());
        frame.add(analogs);
        frames[f] = frame;
    }
    EXPECT_THROW(new_c3d.c3d.analog("wrongChannel", frames), std::invalid_argument);

    // Already existing channels
    for (size_t f = 0; f < new_c3d.nFrames; ++f){
        ezc3d::DataNS::Frame frame;
        ezc3d::DataNS::AnalogsNS::Analogs analogs;
        for (size_t sf = 0; sf < new_c3d.nSubframes; ++sf){
            ezc3d::DataNS::AnalogsNS::SubFrame subframes;
            for (size_t c = 0; c < new_c3d.nAnalogs; ++c){
                ezc3d::DataNS::AnalogsNS::Channel channel;
                channel.data(static_cast<double>(2*f+3*sf+4*c+1) / static_cast<double>(7.0)); // Generate random data
                subframes.channel(channel);
            }
            analogs.subframe(subframes);
        }
        frame.add(analogs);
        frames[f] = frame;
    }
    EXPECT_THROW(new_c3d.c3d.analog(new_c3d.analogNames, frames), std::invalid_argument);

    // No throw
    std::vector<std::string> analogNames = {"NewAnalog1", "NewAnalog2", "NewAnalog3", "NewAnalog4"};
    for (size_t f = 0; f < new_c3d.nFrames; ++f){
        ezc3d::DataNS::Frame frame;
        ezc3d::DataNS::AnalogsNS::Analogs analogs;
        for (size_t sf = 0; sf < new_c3d.nSubframes; ++sf){
            ezc3d::DataNS::AnalogsNS::SubFrame subframes;
            for (size_t c = 0; c < analogNames.size(); ++c){
                ezc3d::DataNS::AnalogsNS::Channel channel;
                channel.data(static_cast<double>(2*f+3*sf+4*c+1) / static_cast<double>(7.0)); // Generate random data
                subframes.channel(channel);
            }
            analogs.subframe(subframes);
        }
        frame.add(analogs);
        frames[f] = frame;
    }
    EXPECT_NO_THROW(new_c3d.c3d.analog(analogNames, frames));

    // Get outside channel
    EXPECT_THROW(new_c3d.c3d.data().frame(0).analogs().subframe(0).channel(8), std::out_of_range);

    // Get channel names
    for (size_t c = 0; c < new_c3d.analogNames.size(); ++c){
        size_t channelName(new_c3d.c3d.channelIdx(new_c3d.analogNames[c]));
        EXPECT_NO_THROW(new_c3d.c3d.data().frame(0).analogs().subframe(0).channel(channelName));
    }
    EXPECT_THROW(new_c3d.c3d.channelIdx("ThisIsNotARealChannel"), std::invalid_argument);

    // Get a subframe
    {
        EXPECT_THROW(new_c3d.c3d.data().frame(new_c3d.c3d.data().nbFrames()), std::out_of_range);
        EXPECT_NO_THROW(new_c3d.c3d.data().frame(0));
    }

    // Create an analog and replace the subframes
    {
        ezc3d::DataNS::AnalogsNS::Analogs analogs;
        ezc3d::DataNS::AnalogsNS::SubFrame sfToBeReplaced;
        ezc3d::DataNS::AnalogsNS::SubFrame sfToReplace;
        analogs.subframe(sfToBeReplaced);

        size_t nbSubframesOld(analogs.nbSubframes());
        EXPECT_NO_THROW(analogs.subframe(sfToReplace, analogs.nbSubframes()+2));
        EXPECT_EQ(analogs.nbSubframes(), nbSubframesOld + 3);
        EXPECT_NO_THROW(analogs.subframe(sfToReplace, 0));

        EXPECT_THROW(analogs.subframe(analogs.nbSubframes()), std::out_of_range);
        EXPECT_NO_THROW(analogs.subframe(0));
    }

    // adding/replacing channels and getting them
    {
        ezc3d::DataNS::AnalogsNS::Channel channelToBeReplaced;
        ezc3d::DataNS::AnalogsNS::Channel channelToReplace;

        ezc3d::DataNS::AnalogsNS::SubFrame subframe;
        subframe.channel(channelToBeReplaced);
        size_t nbChannelOld(subframe.nbChannels());
        EXPECT_NO_THROW(subframe.channel(channelToReplace, subframe.nbChannels()+2));
        EXPECT_EQ(subframe.nbChannels(), nbChannelOld + 3);
        EXPECT_NO_THROW(subframe.channel(channelToReplace, 0));

        EXPECT_THROW(subframe.channel(subframe.nbChannels()), std::out_of_range);
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
    EXPECT_EQ(new_c3d.c3d.header().nb3dPoints(), new_c3d.nPoints);
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
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("USED").valuesAsInt()[0], new_c3d.nPoints);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("SCALE").valuesAsDouble().size(), 1);
    EXPECT_FLOAT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("SCALE").valuesAsDouble()[0], -1);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("RATE").type(), ezc3d::FLOAT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("RATE").valuesAsDouble().size(), 1);
    EXPECT_FLOAT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("RATE").valuesAsDouble()[0], new_c3d.pointFrameRate);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("FRAMES").type(), ezc3d::INT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("FRAMES").valuesAsInt().size(), 1);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("FRAMES").valuesAsInt()[0], new_c3d.nFrames);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("LABELS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("LABELS").valuesAsString().size(), new_c3d.nPoints);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("DESCRIPTIONS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("DESCRIPTIONS").valuesAsString().size(), new_c3d.nPoints);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("UNITS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("UNITS").valuesAsString().size(), new_c3d.nPoints);
    for (size_t m = 0; m < new_c3d.nPoints; ++m){
        EXPECT_STREQ(new_c3d.c3d.parameters().group("POINT").parameter("LABELS").valuesAsString()[m].c_str(), new_c3d.pointNames[m].c_str());
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
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("GEN_SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("GEN_SCALE").valuesAsDouble().size(), 1);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("GEN_SCALE").valuesAsDouble()[0], 1);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("SCALE").valuesAsDouble().size(), new_c3d.nAnalogs);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("OFFSET").type(), ezc3d::INT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("OFFSET").valuesAsInt().size(), new_c3d.nAnalogs);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("UNITS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("UNITS").valuesAsString().size(), new_c3d.nAnalogs);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("RATE").type(), ezc3d::FLOAT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("RATE").valuesAsDouble().size(), 1);
    EXPECT_FLOAT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("RATE").valuesAsDouble()[0], new_c3d.analogFrameRate);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("FORMAT").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("FORMAT").valuesAsString().size(), 0);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("BITS").type(), ezc3d::INT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("BITS").valuesAsInt().size(), 0);

    for (size_t a = 0; a < new_c3d.nAnalogs; ++a){
        EXPECT_STREQ(new_c3d.c3d.parameters().group("ANALOG").parameter("LABELS").valuesAsString()[a].c_str(), new_c3d.analogNames[a].c_str());
        EXPECT_STREQ(new_c3d.c3d.parameters().group("ANALOG").parameter("DESCRIPTIONS").valuesAsString()[a].c_str(), "");
        EXPECT_FLOAT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("SCALE").valuesAsDouble()[a], 1);
        EXPECT_EQ(new_c3d.c3d.parameters().group("ANALOG").parameter("OFFSET").valuesAsInt()[a], 0);
        EXPECT_STREQ(new_c3d.c3d.parameters().group("ANALOG").parameter("UNITS").valuesAsString()[a].c_str(), "");
    }


    // DATA
    for (size_t f = 0; f < new_c3d.nFrames; ++f){
        for (size_t m = 0; m < new_c3d.nPoints; ++m){
            size_t pointIdx(new_c3d.c3d.pointIdx(new_c3d.pointNames[m]));
            EXPECT_DOUBLE_EQ(new_c3d.c3d.data().frame(f).points().point(pointIdx).x(), static_cast<double>(2*f+3*m+1) / static_cast<double>(7.0));
            EXPECT_DOUBLE_EQ(new_c3d.c3d.data().frame(f).points().point(pointIdx).y(), static_cast<double>(2*f+3*m+2) / static_cast<double>(7.0));
            EXPECT_DOUBLE_EQ(new_c3d.c3d.data().frame(f).points().point(pointIdx).z(), static_cast<double>(2*f+3*m+3) / static_cast<double>(7.0));
        }

        for (size_t sf = 0; sf < new_c3d.nSubframes; ++sf)
            for (size_t c = 0; c < new_c3d.nAnalogs; ++c)
                EXPECT_FLOAT_EQ(new_c3d.c3d.data().frame(f).analogs().subframe(sf).channel(c).data(),
                                static_cast<double>(2*f+3*sf+4*c+1) / static_cast<double>(7.0));

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
    EXPECT_THROW(new_c3d.c3d.frame(stupidFramePoint), std::runtime_error);

    // Wrong number of frames
    ezc3d::DataNS::Points3dNS::Points stupidPoints(new_c3d.c3d.data().frame(0).points());
    stupidFramePoint.add(stupidPoints);
    new_c3d.c3d.frame(stupidFramePoint);

    // Add Analogs
    new_c3d.nAnalogs = 3;
    ezc3d::ParametersNS::GroupNS::Parameter analogUsed(new_c3d.c3d.parameters().group("ANALOG").parameter("USED"));
    analogUsed.set(std::vector<int>()={static_cast<int>(new_c3d.nAnalogs)});
    new_c3d.c3d.parameter("ANALOG", analogUsed);

    ezc3d::DataNS::Frame stupidFrameAnalog;
    ezc3d::DataNS::AnalogsNS::Analogs stupidAnalogs(new_c3d.c3d.data().frame(0).analogs());
    ezc3d::DataNS::AnalogsNS::SubFrame stupidSubframe;
    stupidSubframe.nbChannels(new_c3d.nAnalogs-1);
    stupidAnalogs.subframe(stupidSubframe);
    stupidFrameAnalog.add(stupidPoints, stupidAnalogs);
    EXPECT_THROW(new_c3d.c3d.frame(stupidFrameAnalog), std::runtime_error); // Wrong frame rate for analogs

    ezc3d::ParametersNS::GroupNS::Parameter analogRate(new_c3d.c3d.parameters().group("ANALOG").parameter("RATE"));
    analogRate.set(std::vector<double>()={100.});
    new_c3d.c3d.parameter("ANALOG", analogRate);
    EXPECT_THROW(new_c3d.c3d.frame(stupidFrameAnalog), std::runtime_error);

    ezc3d::DataNS::AnalogsNS::SubFrame notSoStupidSubframe;
    notSoStupidSubframe.nbChannels(new_c3d.nAnalogs);
    stupidAnalogs.subframe(notSoStupidSubframe, 0);
    stupidFrameAnalog.add(stupidPoints, stupidAnalogs);
    EXPECT_NO_THROW(new_c3d.c3d.frame(stupidFrameAnalog));

    // Remove point frame rate and then
    ezc3d::ParametersNS::GroupNS::Parameter pointRate(new_c3d.c3d.parameters().group("POINT").parameter("RATE"));
    pointRate.set(std::vector<double>()={0.});
    new_c3d.c3d.parameter("POINT", pointRate);
    EXPECT_THROW(new_c3d.c3d.frame(stupidFrameAnalog), std::runtime_error);
}


TEST(c3dModifier, specificFrames){
    // Create an empty c3d
    c3dTestStruct new_c3d;
    fillC3D(new_c3d, true, false);


    // Replace existing frame
    for (size_t f = 0; f < new_c3d.nFrames; ++f){
        ezc3d::DataNS::Points3dNS::Points pts;
        for (size_t m = 0; m < new_c3d.nPoints; ++m){
            ezc3d::DataNS::Points3dNS::Point pt;
            // Generate some random data
            pt.x(static_cast<double>(4*f+2*m+5) / static_cast<double>(17.0));
            pt.y(static_cast<double>(4*f+2*m+6) / static_cast<double>(17.0));
            pt.z(static_cast<double>(4*f+2*m+7) / static_cast<double>(17.0));
            pts.point(pt);
        }
        ezc3d::DataNS::Frame frame;
        frame.add(pts);
        new_c3d.c3d.frame(frame, f);
    }

    // Add a new frame at 2 spaces after the last frame
    {
        size_t f(new_c3d.nFrames + 1);
        ezc3d::DataNS::Points3dNS::Points pts;
        for (size_t m = 0; m < new_c3d.nPoints; ++m){
            ezc3d::DataNS::Points3dNS::Point pt;
            // Generate some random data
            pt.x(static_cast<double>(4*f+2*m+5) / static_cast<double>(17.0));
            pt.y(static_cast<double>(4*f+2*m+6) / static_cast<double>(17.0));
            pt.z(static_cast<double>(4*f+2*m+7) / static_cast<double>(17.0));
            pts.point(pt);
        }
        ezc3d::DataNS::Frame frame;
        frame.add(pts);
        new_c3d.c3d.frame(frame, f);
    }
    new_c3d.nFrames += 2;

    // Get a frame and some inexistant one
    EXPECT_THROW(new_c3d.c3d.data().frame(new_c3d.nFrames), std::out_of_range);

    // Try to get a all subframes and an out-of-range subframe
    EXPECT_NO_THROW(new_c3d.c3d.data().frame(0).analogs().subframes());
    EXPECT_THROW(new_c3d.c3d.data().frame(0).analogs().subframe(0), std::out_of_range);

    // HEADER
    // Things that should remain as default
    defaultHeaderTest(new_c3d.c3d, HEADER_TYPE::ANALOG_AND_EVENT);

    // Things that should have change
    EXPECT_EQ(new_c3d.c3d.header().nb3dPoints(), new_c3d.nPoints);
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
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("USED").valuesAsInt()[0], new_c3d.nPoints);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("SCALE").valuesAsDouble().size(), 1);
    EXPECT_FLOAT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("SCALE").valuesAsDouble()[0], -1);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("RATE").type(), ezc3d::FLOAT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("RATE").valuesAsDouble().size(), 1);
    EXPECT_FLOAT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("RATE").valuesAsDouble()[0], new_c3d.pointFrameRate);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("FRAMES").type(), ezc3d::INT);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("FRAMES").valuesAsInt().size(), 1);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("FRAMES").valuesAsInt()[0], new_c3d.nFrames);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("LABELS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("LABELS").valuesAsString().size(), new_c3d.nPoints);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("DESCRIPTIONS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("DESCRIPTIONS").valuesAsString().size(), new_c3d.nPoints);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("UNITS").type(), ezc3d::CHAR);
    EXPECT_EQ(new_c3d.c3d.parameters().group("POINT").parameter("UNITS").valuesAsString().size(), new_c3d.nPoints);
    for (size_t m = 0; m < new_c3d.nPoints; ++m){
        EXPECT_STREQ(new_c3d.c3d.parameters().group("POINT").parameter("LABELS").valuesAsString()[m].c_str(),new_c3d.pointNames[m].c_str());
        EXPECT_STREQ(new_c3d.c3d.parameters().group("POINT").parameter("DESCRIPTIONS").valuesAsString()[m].c_str(), "");
        EXPECT_STREQ(new_c3d.c3d.parameters().group("POINT").parameter("UNITS").valuesAsString()[m].c_str(), "mm");
    }

    // DATA
    for (size_t f = 0; f < new_c3d.nFrames; ++f){
        if (f != new_c3d.nFrames - 2) { // Where no frames where added
            for (size_t m = 0; m < new_c3d.nPoints; ++m){
                size_t pointIdx(new_c3d.c3d.pointIdx(new_c3d.pointNames[m]));
                EXPECT_DOUBLE_EQ(new_c3d.c3d.data().frame(f).points().point(pointIdx).x(), static_cast<double>(4*f+2*m+5) / static_cast<double>(17.0));
                EXPECT_DOUBLE_EQ(new_c3d.c3d.data().frame(f).points().point(pointIdx).y(), static_cast<double>(4*f+2*m+6) / static_cast<double>(17.0));
                EXPECT_DOUBLE_EQ(new_c3d.c3d.data().frame(f).points().point(pointIdx).z(), static_cast<double>(4*f+2*m+7) / static_cast<double>(17.0));
            }
        } else {
            for (size_t m = 0; m < new_c3d.nPoints; ++m){
                size_t pointIdx(new_c3d.c3d.pointIdx(new_c3d.pointNames[m]));
                EXPECT_THROW(new_c3d.c3d.data().frame(f).points().point(pointIdx).x(), std::out_of_range);
                EXPECT_THROW(new_c3d.c3d.data().frame(f).points().point(pointIdx).y(), std::out_of_range);
                EXPECT_THROW(new_c3d.c3d.data().frame(f).points().point(pointIdx).z(), std::out_of_range);
            }
        }
    }
}


TEST(c3dFileIO, CreateWriteAndReadBack){
    // Create an empty c3d fill it with data and reopen
    c3dTestStruct ref_c3d;
    fillC3D(ref_c3d, true, true);

    // Change the first frame
    ref_c3d.c3d.setFirstFrame(10);

    // Lock Point parameter
    ref_c3d.c3d.lockGroup("POINT");
    ezc3d::ParametersNS::GroupNS::Parameter p;
    p.name("MyNewParameter");
    p.set("ThisisEmpty");
    ref_c3d.c3d.parameter("MyNewGroup", p);

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
    EXPECT_EQ(read_c3d.header().nb3dPoints(), ref_c3d.nPoints);
    EXPECT_EQ(read_c3d.header().firstFrame(), 10);
    EXPECT_EQ(read_c3d.header().lastFrame(), 10 + ref_c3d.nFrames - 1);
    EXPECT_EQ(read_c3d.header().nbMaxInterpGap(), 10);
    EXPECT_EQ(read_c3d.header().scaleFactor(), -1);
    EXPECT_FLOAT_EQ(read_c3d.header().frameRate(), ref_c3d.pointFrameRate);
    EXPECT_EQ(read_c3d.header().nbFrames(), ref_c3d.nFrames);
    EXPECT_EQ(read_c3d.header().nbAnalogsMeasurement(), ref_c3d.nAnalogs * ref_c3d.nSubframes);
    EXPECT_EQ(read_c3d.header().nbAnalogByFrame(), ref_c3d.nSubframes);
    EXPECT_EQ(read_c3d.header().nbAnalogs(), ref_c3d.nAnalogs);


    // PARAMETERS
    // Things that should remain as default
    EXPECT_EQ(read_c3d.parameters().checksum(), 80);
    EXPECT_EQ(read_c3d.parameters().nbGroups(), 5);
    defaultParametersTest(read_c3d, PARAMETER_TYPE::FORCE_PLATFORM);

    // Things that should have change
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("USED").type(), ezc3d::INT);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("USED").valuesAsInt().size(), 1);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("USED").valuesAsInt()[0], ref_c3d.nPoints);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("SCALE").valuesAsDouble().size(), 1);
    EXPECT_FLOAT_EQ(read_c3d.parameters().group("POINT").parameter("SCALE").valuesAsDouble()[0], -1);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("RATE").type(), ezc3d::FLOAT);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("RATE").valuesAsDouble().size(), 1);
    EXPECT_FLOAT_EQ(read_c3d.parameters().group("POINT").parameter("RATE").valuesAsDouble()[0], ref_c3d.pointFrameRate);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("FRAMES").type(), ezc3d::INT);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("FRAMES").valuesAsInt().size(), 1);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("FRAMES").valuesAsInt()[0], ref_c3d.nFrames);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("LABELS").type(), ezc3d::CHAR);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("LABELS").valuesAsString().size(), ref_c3d.nPoints);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("DESCRIPTIONS").type(), ezc3d::CHAR);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("DESCRIPTIONS").valuesAsString().size(), ref_c3d.nPoints);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("UNITS").type(), ezc3d::CHAR);
    EXPECT_EQ(read_c3d.parameters().group("POINT").parameter("UNITS").valuesAsString().size(), ref_c3d.nPoints);
    for (size_t m = 0; m < ref_c3d.nPoints; ++m){
        EXPECT_STREQ(read_c3d.parameters().group("POINT").parameter("LABELS").valuesAsString()[m].c_str(), ref_c3d.pointNames[m].c_str());
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
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("GEN_SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("GEN_SCALE").valuesAsDouble().size(), 1);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("GEN_SCALE").valuesAsDouble()[0], 1);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("SCALE").valuesAsDouble().size(), ref_c3d.nAnalogs);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("OFFSET").type(), ezc3d::INT);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("OFFSET").valuesAsInt().size(), ref_c3d.nAnalogs);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("UNITS").type(), ezc3d::CHAR);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("UNITS").valuesAsString().size(), ref_c3d.nAnalogs);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("RATE").type(), ezc3d::FLOAT);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("RATE").valuesAsDouble().size(), 1);
    EXPECT_FLOAT_EQ(read_c3d.parameters().group("ANALOG").parameter("RATE").valuesAsDouble()[0], ref_c3d.analogFrameRate);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("FORMAT").type(), ezc3d::CHAR);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("FORMAT").valuesAsString().size(), 0);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("BITS").type(), ezc3d::INT);
    EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("BITS").valuesAsInt().size(), 0);

    for (size_t a = 0; a < ref_c3d.nAnalogs; ++a){
        EXPECT_STREQ(read_c3d.parameters().group("ANALOG").parameter("LABELS").valuesAsString()[a].c_str(), ref_c3d.analogNames[a].c_str());
        EXPECT_STREQ(read_c3d.parameters().group("ANALOG").parameter("DESCRIPTIONS").valuesAsString()[a].c_str(), "");
        EXPECT_FLOAT_EQ(read_c3d.parameters().group("ANALOG").parameter("SCALE").valuesAsDouble()[a], 1);
        EXPECT_EQ(read_c3d.parameters().group("ANALOG").parameter("OFFSET").valuesAsInt()[a], 0);
        EXPECT_STREQ(read_c3d.parameters().group("ANALOG").parameter("UNITS").valuesAsString()[a].c_str(), "");
    }

    EXPECT_STREQ(read_c3d.parameters().group("MyNewGroup").parameter("MyNewParameter").valuesAsString()[0].c_str(), "ThisisEmpty");
    EXPECT_STREQ(read_c3d.parameters().group("EZC3D").parameter("VERSION").valuesAsString()[0].c_str(), EZC3D_VERSION);
    EXPECT_STREQ(read_c3d.parameters().group("EZC3D").parameter("CONTACT").valuesAsString()[0].c_str(), EZC3D_CONTACT);

    // DATA
    for (size_t f = 0; f < ref_c3d.nFrames; ++f){
        for (size_t m = 0; m < ref_c3d.nPoints; ++m){
            size_t pointIdx(read_c3d.pointIdx(ref_c3d.pointNames[m]));
            EXPECT_FLOAT_EQ(read_c3d.data().frame(f).points().point(pointIdx).x(), static_cast<double>(2*f+3*m+1) / static_cast<double>(7.0));
            EXPECT_FLOAT_EQ(read_c3d.data().frame(f).points().point(pointIdx).y(), static_cast<double>(2*f+3*m+2) / static_cast<double>(7.0));
            EXPECT_FLOAT_EQ(read_c3d.data().frame(f).points().point(pointIdx).z(), static_cast<double>(2*f+3*m+3) / static_cast<double>(7.0));
        }

        for (size_t sf = 0; sf < ref_c3d.nSubframes; ++sf)
            for (size_t c = 0; c < ref_c3d.nAnalogs; ++c)
                EXPECT_FLOAT_EQ(read_c3d.data().frame(f).analogs().subframe(sf).channel(c).data(),
                                static_cast<double>(2*f+3*sf+4*c+1) / static_cast<double>(7.0));

    }
}


TEST(c3dFileIO, CreateWriteAndReadBackWithNan){
    // Create an empty c3d fill it with data and reopen
    c3dTestStruct ref_c3d;
    fillC3D(ref_c3d, true, true);

    // Change some values for Nan
    size_t idxFrame(1);
    size_t idxSubframe(2);
    size_t idxPoint(1);
    size_t idxChannel(2);
    // For some reason, the compiler doesn't notice that
    // data is supposed to be const...
    ezc3d::DataNS::Frame frame(ref_c3d.c3d.data().frame(idxFrame));
    frame.points().point(idxPoint).x(NAN);
    frame.points().point(idxPoint).y(NAN);
    frame.points().point(idxPoint).z(NAN);
    frame.analogs().subframe(idxSubframe).channel(idxChannel).data(NAN);

    // Write the c3d on the disk
    std::string savePath("temporary.c3d");
    ref_c3d.c3d.write(savePath.c_str());

    // Open it back and delete it
    ezc3d::c3d read_c3d(savePath.c_str());
    remove(savePath.c_str());

    ezc3d::DataNS::Points3dNS::Point point(
                read_c3d.data().frame(idxFrame).points().point(idxPoint));
    EXPECT_TRUE(std::isnan(point.x()));
    EXPECT_TRUE(std::isnan(point.y()));
    EXPECT_TRUE(std::isnan(point.z()));
    EXPECT_EQ(point.residual(), -1);

    ezc3d::DataNS::AnalogsNS::Channel channel(
                read_c3d.data().frame(idxFrame).analogs().subframe(idxSubframe)
                .channel(idxChannel));
    EXPECT_TRUE(std::isnan(channel.data()));
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
    EXPECT_FLOAT_EQ(Vicon.header().scaleFactor(), static_cast<float>(-0.01));
    EXPECT_FLOAT_EQ(Vicon.header().frameRate(), 100);

    // Analog stuff
    EXPECT_EQ(Vicon.header().nbAnalogsMeasurement(), 760);
    EXPECT_EQ(Vicon.header().nbAnalogByFrame(), 20);
    EXPECT_EQ(Vicon.header().nbAnalogs(), 38);

    // Event stuff
    EXPECT_EQ(Vicon.header().nbEvents(), 0);

    EXPECT_EQ(Vicon.header().eventsTime().size(), 18);
    for (size_t e = 0; e < Vicon.header().eventsTime().size(); ++e)
        EXPECT_FLOAT_EQ(Vicon.header().eventsTime(e), 0);
    EXPECT_THROW(Vicon.header().eventsTime(Vicon.header().eventsTime().size()), std::out_of_range);

    EXPECT_EQ(Vicon.header().eventsLabel().size(), 18);
    for (size_t e = 0; e < Vicon.header().eventsLabel().size(); ++e)
        EXPECT_STREQ(Vicon.header().eventsLabel(e).c_str(), "");
    EXPECT_THROW(Vicon.header().eventsLabel(Vicon.header().eventsLabel().size()), std::out_of_range);

    EXPECT_EQ(Vicon.header().eventsDisplay().size(), 9);
    for (size_t e = 0; e < Vicon.header().eventsDisplay().size(); ++e)
        EXPECT_EQ(Vicon.header().eventsDisplay(e), 0);
    EXPECT_THROW(Vicon.header().eventsDisplay(Vicon.header().eventsDisplay().size()), std::out_of_range);


    EXPECT_EQ(Vicon.header().firstFrame(), 0);
    EXPECT_EQ(Vicon.header().lastFrame(), 579);
    EXPECT_EQ(Vicon.header().nbFrames(), 580);


    // Parameter tests
    EXPECT_EQ(Vicon.parameters().checksum(), 80);
    EXPECT_EQ(Vicon.parameters().nbGroups(), 9);

    EXPECT_EQ(Vicon.parameters().group("POINT").parameter("USED").type(), ezc3d::INT);
    EXPECT_EQ(Vicon.parameters().group("POINT").parameter("USED").valuesAsInt().size(), 1);
    EXPECT_EQ(Vicon.parameters().group("POINT").parameter("USED").valuesAsInt()[0], 51);
    EXPECT_EQ(Vicon.parameters().group("POINT").parameter("SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(Vicon.parameters().group("POINT").parameter("SCALE").valuesAsDouble().size(), 1);
    EXPECT_FLOAT_EQ(Vicon.parameters().group("POINT").parameter("SCALE").valuesAsDouble()[0], static_cast<float>(-0.0099999998));
    EXPECT_EQ(Vicon.parameters().group("POINT").parameter("RATE").type(), ezc3d::FLOAT);
    EXPECT_EQ(Vicon.parameters().group("POINT").parameter("RATE").valuesAsDouble().size(), 1);
    EXPECT_FLOAT_EQ(Vicon.parameters().group("POINT").parameter("RATE").valuesAsDouble()[0], 100);
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
    EXPECT_EQ(Vicon.parameters().group("ANALOG").parameter("GEN_SCALE").valuesAsDouble().size(), 1);
    EXPECT_EQ(Vicon.parameters().group("ANALOG").parameter("GEN_SCALE").valuesAsDouble()[0], 1);
    EXPECT_EQ(Vicon.parameters().group("ANALOG").parameter("SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(Vicon.parameters().group("ANALOG").parameter("SCALE").valuesAsDouble().size(), 38);
    EXPECT_EQ(Vicon.parameters().group("ANALOG").parameter("OFFSET").type(), ezc3d::INT);
    EXPECT_EQ(Vicon.parameters().group("ANALOG").parameter("OFFSET").valuesAsInt().size(), 38);
    EXPECT_EQ(Vicon.parameters().group("ANALOG").parameter("UNITS").type(), ezc3d::CHAR);
    EXPECT_EQ(Vicon.parameters().group("ANALOG").parameter("UNITS").valuesAsString().size(), 38);
    EXPECT_EQ(Vicon.parameters().group("ANALOG").parameter("RATE").type(), ezc3d::FLOAT);
    EXPECT_EQ(Vicon.parameters().group("ANALOG").parameter("RATE").valuesAsDouble().size(), 1);
    EXPECT_FLOAT_EQ(Vicon.parameters().group("ANALOG").parameter("RATE").valuesAsDouble()[0], 2000);

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
    EXPECT_EQ(Vicon.parameters().group("FORCE_PLATFORM").parameter("CORNERS").valuesAsDouble().size(), 0);
    EXPECT_EQ(Vicon.parameters().group("FORCE_PLATFORM").parameter("ORIGIN").type(), ezc3d::FLOAT);
    EXPECT_EQ(Vicon.parameters().group("FORCE_PLATFORM").parameter("ORIGIN").valuesAsDouble().size(), 0);
    EXPECT_EQ(Vicon.parameters().group("FORCE_PLATFORM").parameter("CHANNEL").type(), ezc3d::INT);
    EXPECT_EQ(Vicon.parameters().group("FORCE_PLATFORM").parameter("CHANNEL").valuesAsInt().size(), 0);

    EXPECT_STREQ(Vicon.parameters().group("MANUFACTURER").parameter("COMPANY").valuesAsString()[0].c_str(), "Vicon");
    EXPECT_STREQ(Vicon.parameters().group("MANUFACTURER").parameter("SOFTWARE").valuesAsString()[0].c_str(), "Vicon Nexus");
    EXPECT_STREQ(Vicon.parameters().group("MANUFACTURER").parameter("VERSION_LABEL").valuesAsString()[0].c_str(), "2.4.0.91647h");

    // DATA
    for (size_t f = 0; f < 580; ++f){
        EXPECT_EQ(Vicon.data().frame(f).points().nbPoints(), 51);
        for (size_t sf = 0; sf < 10; ++sf)
            EXPECT_EQ(Vicon.data().frame(f).analogs().subframe(sf).nbChannels(), 38);
    }
    // Test some values randomly
    EXPECT_FLOAT_EQ(Vicon.data().frame(0).points().point(0).x(), 44.16278839111328);
    EXPECT_FLOAT_EQ(Vicon.data().frame(Vicon.data().frames().size()-1).points().point(50).z(), 99.682586669921875);

    // Test sum of all values
    double sumValues(0);
    for (size_t f = 0; f < 580; ++f){
        EXPECT_EQ(Vicon.data().frame(f).points().nbPoints(), 51);
        for (auto pt : Vicon.data().frame(f).points().points()){
            if (pt.isValid()){
                sumValues += pt.x();
                sumValues += pt.y();
                sumValues += pt.z();
            }
        }
    }
    EXPECT_FLOAT_EQ(sumValues, 42506014.918278672);
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
    EXPECT_FLOAT_EQ(Qualisys.header().scaleFactor(), static_cast<float>(-0.0762322545));
    EXPECT_FLOAT_EQ(Qualisys.header().frameRate(), 200);

    // Analog stuff
    EXPECT_EQ(Qualisys.header().nbAnalogsMeasurement(), 690);
    EXPECT_EQ(Qualisys.header().nbAnalogByFrame(), 10);
    EXPECT_EQ(Qualisys.header().nbAnalogs(), 69);

    // Event stuff
    EXPECT_EQ(Qualisys.header().nbEvents(), 0);

    EXPECT_EQ(Qualisys.header().eventsTime().size(), 18);
    for (size_t e = 0; e < Qualisys.header().eventsTime().size(); ++e)
        EXPECT_FLOAT_EQ(Qualisys.header().eventsTime(e), 0);
    EXPECT_THROW(Qualisys.header().eventsTime(Qualisys.header().eventsTime().size()), std::out_of_range);

    EXPECT_EQ(Qualisys.header().eventsLabel().size(), 18);
    for (size_t e = 0; e < Qualisys.header().eventsLabel().size(); ++e)
        EXPECT_STREQ(Qualisys.header().eventsLabel(e).c_str(), "");
    EXPECT_THROW(Qualisys.header().eventsLabel(Qualisys.header().eventsLabel().size()), std::out_of_range);

    EXPECT_EQ(Qualisys.header().eventsDisplay().size(), 9);
    for (size_t e = 0; e < Qualisys.header().eventsDisplay().size(); ++e)
        EXPECT_EQ(Qualisys.header().eventsDisplay(e), 257);
    EXPECT_THROW(Qualisys.header().eventsDisplay(Qualisys.header().eventsDisplay().size()), std::out_of_range);


    EXPECT_EQ(Qualisys.header().firstFrame(), 704);
    EXPECT_EQ(Qualisys.header().lastFrame(), 1043);
    EXPECT_EQ(Qualisys.header().nbFrames(), 340);


    // Parameter tests
    EXPECT_EQ(Qualisys.parameters().checksum(), 80);
    EXPECT_EQ(Qualisys.parameters().nbGroups(), 7);

    EXPECT_EQ(Qualisys.parameters().group("POINT").parameter("USED").type(), ezc3d::INT);
    EXPECT_EQ(Qualisys.parameters().group("POINT").parameter("USED").valuesAsInt().size(), 1);
    EXPECT_EQ(Qualisys.parameters().group("POINT").parameter("USED").valuesAsInt()[0], 55);
    EXPECT_EQ(Qualisys.parameters().group("POINT").parameter("SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(Qualisys.parameters().group("POINT").parameter("SCALE").valuesAsDouble().size(), 1);
    EXPECT_FLOAT_EQ(Qualisys.parameters().group("POINT").parameter("SCALE").valuesAsDouble()[0], static_cast<float>(-0.076232255));
    EXPECT_EQ(Qualisys.parameters().group("POINT").parameter("RATE").type(), ezc3d::FLOAT);
    EXPECT_EQ(Qualisys.parameters().group("POINT").parameter("RATE").valuesAsDouble().size(), 1);
    EXPECT_FLOAT_EQ(Qualisys.parameters().group("POINT").parameter("RATE").valuesAsDouble()[0], 200);
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
    EXPECT_EQ(Qualisys.parameters().group("ANALOG").parameter("GEN_SCALE").valuesAsDouble().size(), 1);
    EXPECT_EQ(Qualisys.parameters().group("ANALOG").parameter("GEN_SCALE").valuesAsDouble()[0], 1);
    EXPECT_EQ(Qualisys.parameters().group("ANALOG").parameter("SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(Qualisys.parameters().group("ANALOG").parameter("SCALE").valuesAsDouble().size(), 69);
    EXPECT_EQ(Qualisys.parameters().group("ANALOG").parameter("OFFSET").type(), ezc3d::INT);
    EXPECT_EQ(Qualisys.parameters().group("ANALOG").parameter("OFFSET").valuesAsInt().size(), 69);
    EXPECT_EQ(Qualisys.parameters().group("ANALOG").parameter("UNITS").type(), ezc3d::CHAR);
    EXPECT_EQ(Qualisys.parameters().group("ANALOG").parameter("UNITS").valuesAsString().size(), 69);
    EXPECT_EQ(Qualisys.parameters().group("ANALOG").parameter("RATE").type(), ezc3d::FLOAT);
    EXPECT_EQ(Qualisys.parameters().group("ANALOG").parameter("RATE").valuesAsDouble().size(), 1);
    EXPECT_FLOAT_EQ(Qualisys.parameters().group("ANALOG").parameter("RATE").valuesAsDouble()[0], 2000);

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
    EXPECT_EQ(Qualisys.parameters().group("FORCE_PLATFORM").parameter("CORNERS").valuesAsDouble().size(), 24);
    EXPECT_EQ(Qualisys.parameters().group("FORCE_PLATFORM").parameter("ORIGIN").type(), ezc3d::FLOAT);
    EXPECT_EQ(Qualisys.parameters().group("FORCE_PLATFORM").parameter("ORIGIN").valuesAsDouble().size(), 6);
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
        EXPECT_EQ(Qualisys.data().frame(f).points().nbPoints(), 55);
        for (size_t sf = 0; sf < 10; ++sf)
            EXPECT_EQ(Qualisys.data().frame(f).analogs().subframe(sf).nbChannels(), 69);
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
    EXPECT_FLOAT_EQ(Optotrak.header().scaleFactor(), static_cast<float>(-7.8661418));
    EXPECT_FLOAT_EQ(Optotrak.header().frameRate(), 30);

    // Analog stuff
    EXPECT_EQ(Optotrak.header().nbAnalogsMeasurement(), 0);
    EXPECT_EQ(Optotrak.header().nbAnalogByFrame(), 0);
    EXPECT_EQ(Optotrak.header().nbAnalogs(), 0);

    // Event stuff
    EXPECT_EQ(Optotrak.header().nbEvents(), 0);

    EXPECT_EQ(Optotrak.header().eventsTime().size(), 18);
    for (size_t e = 0; e < Optotrak.header().eventsTime().size(); ++e)
        EXPECT_FLOAT_EQ(Optotrak.header().eventsTime(e), 0);
    EXPECT_THROW(Optotrak.header().eventsTime(Optotrak.header().eventsTime().size()), std::out_of_range);

    EXPECT_EQ(Optotrak.header().eventsLabel().size(), 18);
    for (size_t e = 0; e < Optotrak.header().eventsLabel().size(); ++e)
        EXPECT_STREQ(Optotrak.header().eventsLabel(e).c_str(), "");
    EXPECT_THROW(Optotrak.header().eventsLabel(Optotrak.header().eventsLabel().size()), std::out_of_range);

    EXPECT_EQ(Optotrak.header().eventsDisplay().size(), 9);
    for (size_t e = 0; e < Optotrak.header().eventsDisplay().size(); ++e)
        EXPECT_EQ(Optotrak.header().eventsDisplay(e), 0);
    EXPECT_THROW(Optotrak.header().eventsDisplay(Optotrak.header().eventsDisplay().size()), std::out_of_range);


    EXPECT_EQ(Optotrak.header().firstFrame(), 0);
    EXPECT_EQ(Optotrak.header().lastFrame(), 29);
    EXPECT_EQ(Optotrak.header().nbFrames(), 30);


    // Parameter tests
    EXPECT_EQ(Optotrak.parameters().checksum(), 80);
    EXPECT_EQ(Optotrak.parameters().nbGroups(), 3);

    EXPECT_EQ(Optotrak.parameters().group("POINT").parameter("USED").type(), ezc3d::INT);
    EXPECT_EQ(Optotrak.parameters().group("POINT").parameter("USED").valuesAsInt().size(), 1);
    EXPECT_EQ(Optotrak.parameters().group("POINT").parameter("USED").valuesAsInt()[0], 54);
    EXPECT_EQ(Optotrak.parameters().group("POINT").parameter("SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(Optotrak.parameters().group("POINT").parameter("SCALE").valuesAsDouble().size(), 1);
    EXPECT_FLOAT_EQ(Optotrak.parameters().group("POINT").parameter("SCALE").valuesAsDouble()[0], static_cast<float>(-7.8661418));
    EXPECT_EQ(Optotrak.parameters().group("POINT").parameter("RATE").type(), ezc3d::FLOAT);
    EXPECT_EQ(Optotrak.parameters().group("POINT").parameter("RATE").valuesAsDouble().size(), 1);
    EXPECT_FLOAT_EQ(Optotrak.parameters().group("POINT").parameter("RATE").valuesAsDouble()[0], 30);
    EXPECT_EQ(Optotrak.parameters().group("POINT").parameter("FRAMES").valuesAsInt()[0], 30); // ignore because it changes if analog is present
    EXPECT_EQ(Optotrak.parameters().group("POINT").parameter("FRAMES").type(), ezc3d::INT);
    EXPECT_EQ(Optotrak.parameters().group("POINT").parameter("LABELS").type(), ezc3d::CHAR);
    EXPECT_EQ(Optotrak.parameters().group("POINT").parameter("LABELS").valuesAsString().size(), 54);
    EXPECT_EQ(Optotrak.parameters().group("POINT").parameter("DESCRIPTIONS").type(), ezc3d::CHAR);
    EXPECT_EQ(Optotrak.parameters().group("POINT").parameter("DESCRIPTIONS").valuesAsString().size(), 54);
    EXPECT_EQ(Optotrak.parameters().group("POINT").parameter("UNITS").type(), ezc3d::CHAR);
    EXPECT_EQ(Optotrak.parameters().group("POINT").parameter("UNITS").valuesAsString().size(), 54);

    defaultParametersTest(Optotrak, PARAMETER_TYPE::ANALOG);
    defaultParametersTest(Optotrak, PARAMETER_TYPE::FORCE_PLATFORM);

    // DATA
    for (size_t f = 0; f < 30; ++f)
        EXPECT_EQ(Optotrak.data().frame(f).points().nbPoints(), 54);
}

TEST(c3dFileio,readBtsC3D){
    ezc3d::c3d BTS("c3dTestFiles/BTS.c3d");
    // Header test
    // Generic stuff
    EXPECT_EQ(BTS.header().checksum(), 80);
    EXPECT_EQ(BTS.header().keyLabelPresent(), 0);
    EXPECT_EQ(BTS.header().firstBlockKeyLabel(), 0);
    EXPECT_EQ(BTS.header().fourCharPresent(), 12345);
    EXPECT_EQ(BTS.header().emptyBlock1(), 0);
    EXPECT_EQ(BTS.header().emptyBlock2(), 0);
    EXPECT_EQ(BTS.header().emptyBlock3(), 0);
    EXPECT_EQ(BTS.header().emptyBlock4(), 0);

    // Point stuff
    EXPECT_EQ(BTS.header().nb3dPoints(), 22);
    EXPECT_EQ(BTS.header().nbMaxInterpGap(), 10);
    EXPECT_FLOAT_EQ(BTS.header().scaleFactor(), static_cast<float>(-0.1));
    EXPECT_FLOAT_EQ(BTS.header().frameRate(), static_cast<float>(100.0));

    // Analog stuff
    EXPECT_EQ(BTS.header().nbAnalogsMeasurement(), 440);
    EXPECT_EQ(BTS.header().nbAnalogByFrame(), 10);
    EXPECT_EQ(BTS.header().nbAnalogs(), 44);

    // Event stuff
    EXPECT_EQ(BTS.header().nbEvents(), 0);

    EXPECT_EQ(BTS.header().eventsTime().size(), 18);
    for (size_t e = 0; e < BTS.header().eventsTime().size(); ++e)
        EXPECT_FLOAT_EQ(BTS.header().eventsTime(e), 0);
    EXPECT_THROW(BTS.header().eventsTime(BTS.header().eventsTime().size()), std::out_of_range);

    EXPECT_EQ(BTS.header().eventsLabel().size(), 18);
    for (size_t e = 0; e < BTS.header().eventsLabel().size(); ++e)
        EXPECT_STREQ(BTS.header().eventsLabel(e).c_str(), "");
    EXPECT_THROW(BTS.header().eventsLabel(BTS.header().eventsLabel().size()), std::out_of_range);

    EXPECT_EQ(BTS.header().eventsDisplay().size(), 9);
    for (size_t e = 0; e < BTS.header().eventsDisplay().size(); ++e)
        EXPECT_EQ(BTS.header().eventsDisplay(e), 0);
    EXPECT_THROW(BTS.header().eventsDisplay(BTS.header().eventsDisplay().size()), std::out_of_range);

    EXPECT_EQ(BTS.header().firstFrame(), 0);
    EXPECT_EQ(BTS.header().lastFrame(), 674);
    EXPECT_EQ(BTS.header().nbFrames(), 675);

    // Parameter tests
    EXPECT_EQ(BTS.parameters().checksum(), 80);
    EXPECT_EQ(BTS.parameters().nbGroups(), 20);


    EXPECT_EQ(BTS.parameters().group("POINT").parameter("USED").type(), ezc3d::INT);
    EXPECT_EQ(BTS.parameters().group("POINT").parameter("USED").valuesAsInt().size(), 1);
    EXPECT_EQ(BTS.parameters().group("POINT").parameter("USED").valuesAsInt()[0], 22);
    EXPECT_EQ(BTS.parameters().group("POINT").parameter("SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(BTS.parameters().group("POINT").parameter("SCALE").valuesAsDouble().size(), 1);
    EXPECT_FLOAT_EQ(BTS.parameters().group("POINT").parameter("SCALE").valuesAsDouble()[0], static_cast<float>(-0.1));
    EXPECT_EQ(BTS.parameters().group("POINT").parameter("RATE").type(), ezc3d::FLOAT);
    EXPECT_EQ(BTS.parameters().group("POINT").parameter("RATE").valuesAsDouble().size(), 1);
    EXPECT_FLOAT_EQ(BTS.parameters().group("POINT").parameter("RATE").valuesAsDouble()[0], 100);
    EXPECT_EQ(BTS.parameters().group("POINT").parameter("FRAMES").valuesAsInt()[0], 675);
    EXPECT_EQ(BTS.parameters().group("POINT").parameter("FRAMES").type(), ezc3d::INT);
    EXPECT_EQ(BTS.parameters().group("POINT").parameter("LABELS").type(), ezc3d::CHAR);
    EXPECT_EQ(BTS.parameters().group("POINT").parameter("LABELS").valuesAsString().size(), 22);
    EXPECT_EQ(BTS.parameters().group("POINT").parameter("DESCRIPTIONS").type(), ezc3d::CHAR);
    EXPECT_EQ(BTS.parameters().group("POINT").parameter("DESCRIPTIONS").valuesAsString().size(), 22);
    EXPECT_EQ(BTS.parameters().group("POINT").parameter("UNITS").type(), ezc3d::CHAR);
    EXPECT_EQ(BTS.parameters().group("POINT").parameter("UNITS").valuesAsString().size(), 1); //This might be weird. Shouldn't there be 22 as well?

    EXPECT_EQ(BTS.parameters().group("ANALOG").parameter("USED").type(), ezc3d::INT);
    EXPECT_EQ(BTS.parameters().group("ANALOG").parameter("USED").valuesAsInt().size(), 1);
    EXPECT_EQ(BTS.parameters().group("ANALOG").parameter("USED").valuesAsInt()[0], 44);
    EXPECT_EQ(BTS.parameters().group("ANALOG").parameter("LABELS").type(), ezc3d::CHAR);
    EXPECT_EQ(BTS.parameters().group("ANALOG").parameter("LABELS").valuesAsString().size(), 44);
    EXPECT_EQ(BTS.parameters().group("ANALOG").parameter("DESCRIPTIONS").type(), ezc3d::CHAR);
    EXPECT_EQ(BTS.parameters().group("ANALOG").parameter("DESCRIPTIONS").valuesAsString().size(), 44);
    EXPECT_EQ(BTS.parameters().group("ANALOG").parameter("GEN_SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(BTS.parameters().group("ANALOG").parameter("GEN_SCALE").valuesAsDouble().size(), 1);
    EXPECT_EQ(BTS.parameters().group("ANALOG").parameter("GEN_SCALE").valuesAsDouble()[0], 1);
    EXPECT_EQ(BTS.parameters().group("ANALOG").parameter("SCALE").type(), ezc3d::FLOAT);
    EXPECT_EQ(BTS.parameters().group("ANALOG").parameter("SCALE").valuesAsDouble().size(), 44);
    EXPECT_EQ(BTS.parameters().group("ANALOG").parameter("OFFSET").type(), ezc3d::INT);
    EXPECT_EQ(BTS.parameters().group("ANALOG").parameter("OFFSET").valuesAsInt().size(), 44);
    EXPECT_EQ(BTS.parameters().group("ANALOG").parameter("UNITS").type(), ezc3d::CHAR);
    EXPECT_EQ(BTS.parameters().group("ANALOG").parameter("UNITS").valuesAsString().size(), 44);
    EXPECT_EQ(BTS.parameters().group("ANALOG").parameter("RATE").type(), ezc3d::FLOAT);
    EXPECT_EQ(BTS.parameters().group("ANALOG").parameter("RATE").valuesAsDouble().size(), 1);
    EXPECT_FLOAT_EQ(BTS.parameters().group("ANALOG").parameter("RATE").valuesAsDouble()[0], 1000);


    EXPECT_EQ(BTS.parameters().group("FORCE_PLATFORM").parameter("USED").type(), ezc3d::INT);
    EXPECT_EQ(BTS.parameters().group("FORCE_PLATFORM").parameter("USED").valuesAsInt().size(), 1);
    EXPECT_EQ(BTS.parameters().group("FORCE_PLATFORM").parameter("USED").valuesAsInt()[0], 6);
    EXPECT_EQ(BTS.parameters().group("FORCE_PLATFORM").parameter("TYPE").type(), ezc3d::INT);
    EXPECT_EQ(BTS.parameters().group("FORCE_PLATFORM").parameter("TYPE").valuesAsInt().size(), 6);
    EXPECT_EQ(BTS.parameters().group("FORCE_PLATFORM").parameter("ZERO").type(), ezc3d::INT);
    EXPECT_EQ(BTS.parameters().group("FORCE_PLATFORM").parameter("ZERO").valuesAsInt().size(), 2);
    EXPECT_EQ(BTS.parameters().group("FORCE_PLATFORM").parameter("ZERO").valuesAsInt()[0], 0);
    EXPECT_EQ(BTS.parameters().group("FORCE_PLATFORM").parameter("ZERO").valuesAsInt()[1], 0);
    EXPECT_EQ(BTS.parameters().group("FORCE_PLATFORM").parameter("CORNERS").type(), ezc3d::FLOAT);
    EXPECT_EQ(BTS.parameters().group("FORCE_PLATFORM").parameter("CORNERS").valuesAsDouble().size(), 72);
    EXPECT_EQ(BTS.parameters().group("FORCE_PLATFORM").parameter("ORIGIN").type(), ezc3d::FLOAT);
    EXPECT_EQ(BTS.parameters().group("FORCE_PLATFORM").parameter("ORIGIN").valuesAsDouble().size(), 18);
    EXPECT_EQ(BTS.parameters().group("FORCE_PLATFORM").parameter("CHANNEL").type(), ezc3d::INT);
    EXPECT_EQ(BTS.parameters().group("FORCE_PLATFORM").parameter("CHANNEL").valuesAsInt().size(), 36);

        // DATA
    for (size_t f = 0; f < 675; ++f){
        EXPECT_EQ(BTS.data().frame(f).points().nbPoints(), 22);
        for (size_t sf = 0; sf < 10; ++sf)
            EXPECT_EQ(BTS.data().frame(f).analogs().subframe(sf).nbChannels(), 44);
    }
}

TEST(c3dFileIO, comparedIdenticalFilesSample1){
    ezc3d::c3d c3d_pr("c3dTestFiles/Eb015pr.c3d"); // Intel floating format
    ezc3d::c3d c3d_pi("c3dTestFiles/Eb015pi.c3d"); // Intel integer format
    ezc3d::c3d c3d_vr("c3dTestFiles/Eb015vr.c3d"); // DEC floating format
    ezc3d::c3d c3d_vi("c3dTestFiles/Eb015vi.c3d"); // DEC integer format
    EXPECT_THROW(ezc3d::c3d c3d_sr("c3dTestFiles/Eb015dr.c3d"), std::runtime_error); // MIPS float format
    EXPECT_THROW(ezc3d::c3d c3d_si("c3dTestFiles/Eb015di.c3d"), std::runtime_error); // MIPS integer format

    // The header should be the same for relevant informations
    compareHeader(c3d_pr, c3d_pi);
    compareHeader(c3d_pr, c3d_vr);
    compareHeader(c3d_pr, c3d_vi);

    // All the data should be the same
    compareData(c3d_pr, c3d_pi, true);
    compareData(c3d_pr, c3d_vr, true);
    compareData(c3d_pr, c3d_vi, true);
}

TEST(c3dFileIO, comparedIdenticalFilesSample2){
    ezc3d::c3d c3d_pr("c3dTestFiles/pc_real.c3d"); // Intel floating format
    ezc3d::c3d c3d_pi("c3dTestFiles/pc_int.c3d"); // Intel integer format
    ezc3d::c3d c3d_vr("c3dTestFiles/dec_real.c3d"); // DEC floating format
    ezc3d::c3d c3d_vi("c3dTestFiles/dec_int.c3d"); // DEC integer format
    EXPECT_THROW(ezc3d::c3d c3d_sr("c3dTestFiles/sgi_real.c3d"), std::runtime_error); // MIPS float format
    EXPECT_THROW(ezc3d::c3d c3d_si("c3dTestFiles/sgi_int.c3d"), std::runtime_error); // MIPS integer format

    // The header should be the same for relevant informations
    compareHeader(c3d_pr, c3d_pi);
    compareHeader(c3d_pr, c3d_vr);
    // compareHeader(c3d_pr, c3d_vi); // Header is actually different

    // All the data should be the same
    // compareData(c3d_pr, c3d_pi); // Data are actually sligthly different
    compareData(c3d_pr, c3d_vr, true);
    compareData(c3d_pr, c3d_vi, true);
}

TEST(c3dFileIO, parseAndBuildSameFileBTS){
    ezc3d::c3d original("c3dTestFiles/BTS.c3d");
    std::string savePath("c3dTestFiles/BTS_after.c3d");
    original.write(savePath.c_str());
    ezc3d::c3d rebuilt(savePath.c_str());

    compareHeader(original, rebuilt);
    compareData(original, rebuilt);

    remove(savePath.c_str());        
}

TEST(c3dFileIO, parseAndBuildSameFileQualisys){
    ezc3d::c3d original("c3dTestFiles/Qualisys.c3d");
    std::string savePath("c3dTestFiles/Qualisys_after.c3d");
    original.write(savePath.c_str());
    ezc3d::c3d rebuilt(savePath.c_str());

    compareHeader(original, rebuilt);
    compareData(original, rebuilt);

    remove(savePath.c_str());        
}
TEST(c3dFileIO, parseAndBuildSameFileOptoTrack){
    ezc3d::c3d original("c3dTestFiles/Optotrak.c3d");
    std::string savePath("c3dTestFiles/Optotrak_after.c3d");
    original.write(savePath.c_str());
    ezc3d::c3d rebuilt(savePath.c_str());

    compareHeader(original, rebuilt);
    compareData(original, rebuilt);

    remove(savePath.c_str());        
}
TEST(c3dFileIO, parseAndBuildSameFileVicon){
    ezc3d::c3d original("c3dTestFiles/Vicon.c3d");
    std::string savePath("c3dTestFiles/Vicon_after.c3d");
    original.write(savePath.c_str());
    ezc3d::c3d rebuilt(savePath.c_str());

    compareHeader(original, rebuilt);
    compareData(original, rebuilt);

    remove(savePath.c_str());        
}


TEST(c3dShow, printIt){
    // Create an empty c3d and print it
    c3dTestStruct new_c3d;
    fillC3D(new_c3d, true, true);

    EXPECT_NO_THROW(new_c3d.c3d.print());
}

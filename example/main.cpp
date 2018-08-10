
#define BUILD_SANDBOX
#include <vector>
#include "ezc3d.h"



#ifdef BUILD_SANDBOX
 // SANDBOX FOR DEVELOPER
int main()
{
    {
        ezc3d::c3d c3d("markers_analogs.c3d");

        // Add two new markers to the c3d (one filled with zeros, the other one with data)
        c3d.addMarker("new_marker1"); // Add empty
        std::vector<ezc3d::DataNS::Frame> frames_point;
        ezc3d::DataNS::Points3dNS::Points pts_new;
        ezc3d::DataNS::Points3dNS::Point pt_new;
        pt_new.name("new_marker2");
        pt_new.x(1.0);
        pt_new.y(2.0);
        pt_new.z(3.0);
        pts_new.add(pt_new);
        for (int i=0; i<c3d.data().frames().size(); ++i){
            ezc3d::DataNS::Frame frame;
            frame.add(pts_new);
            frames_point.push_back(frame);
        }
        c3d.addMarker(frames_point); // Add the previously created

        // Add a new analog to the c3d (one filled with zeros, the other one with data)
        c3d.addAnalog("new_analog1"); // add the empty
        std::vector<ezc3d::DataNS::Frame> frames_analog;
        ezc3d::DataNS::AnalogsNS::SubFrame subframes_analog;
        ezc3d::DataNS::AnalogsNS::Channel emptyChannel;
        emptyChannel.name("new_analog2");
        ezc3d::DataNS::Frame frame;
        subframes_analog.channels_nonConst().push_back(emptyChannel);
        for (int sf=0; sf<c3d.header().nbAnalogByFrame(); ++sf){
            subframes_analog.channels_nonConst()[0].value(sf+1);
            frame.analogs_nonConst().addSubframe(subframes_analog);
        }
        for (int f=0; f<c3d.data().frames().size(); ++f)
            frames_analog.push_back(frame);
        c3d.addAnalog(frames_analog);

        // Add a new frame
        ezc3d::DataNS::Frame f;
        std::vector<std::string>labels(c3d.parameters().group("POINT").parameter("LABELS").valuesAsString());
        int nPoints(c3d.parameters().group("POINT").parameter("USED").valuesAsInt()[0]);
        ezc3d::DataNS::Points3dNS::Points pts;
        for (int i=0; i<nPoints; ++i){
            ezc3d::DataNS::Points3dNS::Point pt;
            pt.name(labels[i]);
            pt.x(1.0);
            pt.y(2.0);
            pt.z(3.0);
            pts.add(pt);
        }
        ezc3d::DataNS::AnalogsNS::Analogs analog;
        ezc3d::DataNS::AnalogsNS::SubFrame subframe;
        for (int i=0; i<c3d.header().nbAnalogs(); ++i){
            ezc3d::DataNS::AnalogsNS::Channel c;
            c.value(i+1);
            subframe.addChannel(c);
        }
        for (int i=0; i<c3d.header().nbAnalogByFrame(); ++i)
            analog.addSubframe(subframe);
        f.add(pts, analog);
        c3d.addFrame(f);


        // Add new parameters
        ezc3d::ParametersNS::GroupNS::Parameter p1("new_param_1");
        p1.set(std::vector<int>() = {2}, {1});
        c3d.addParameter("new_group_1", p1);
        ezc3d::ParametersNS::GroupNS::Parameter p2("new_param_2");
        p2.set(std::vector<int>() = {1, 1, 2, 3, 5, 8}, {3, 2}); // TESTER LE READER DANS MATLAB!
        c3d.addParameter("new_group_1", p2);
        ezc3d::ParametersNS::GroupNS::Parameter p3("new_param_1");
        p3.set(std::vector<std::string>() = {"value1", "longer_value1", "value2", "longer_value2"}, {20, 2, 2});
        c3d.addParameter("new_group_2", p3);

        // write the changed c3d
        c3d.write("augmentedC3d.c3d");

        // Read it back!
        ezc3d::c3d augmentedC3d("augmentedC3d.c3d");
        augmentedC3d.parameters().print();
    }

    {
        // Create an empty fresh c3d
        ezc3d::c3d c3d_empty;
        ezc3d::ParametersNS::GroupNS::Parameter t("SCALE");
        t.set(std::vector<float>(), {0});


        // Fill it with some values
        ezc3d::ParametersNS::GroupNS::Parameter pointRate("RATE");
        pointRate.set(std::vector<float>() = {100}, {1});
        c3d_empty.addParameter("POINT", pointRate);

        ezc3d::ParametersNS::GroupNS::Parameter analogRate("RATE");
        analogRate.set(std::vector<float>() = {1000}, {1});
        c3d_empty.addParameter("ANALOG", analogRate);

        c3d_empty.addMarker("new_marker1"); // Add empty
        c3d_empty.addMarker("new_marker2"); // Add empty
        c3d_empty.addAnalog("new_analog1"); // add the empty
        c3d_empty.addAnalog("new_analog2"); // add the empty
        // Add a new frame
        ezc3d::DataNS::Frame f;
        std::vector<std::string>labels(c3d_empty.parameters().group("POINT").parameter("LABELS").valuesAsString());
        int nPoints(c3d_empty.parameters().group("POINT").parameter("USED").valuesAsInt()[0]);
        ezc3d::DataNS::Points3dNS::Points pts;
        for (int i=0; i<nPoints; ++i){
            ezc3d::DataNS::Points3dNS::Point pt;
            pt.name(labels[i]);
            pt.x(1.0);
            pt.y(2.0);
            pt.z(3.0);
            pts.add(pt);
        }
        ezc3d::DataNS::AnalogsNS::Analogs analog;
        ezc3d::DataNS::AnalogsNS::SubFrame subframe;
        for (int i=0; i<c3d_empty.header().nbAnalogs(); ++i){
            ezc3d::DataNS::AnalogsNS::Channel c;
            c.value(i+1);
            subframe.addChannel(c);
        }
        for (int i=0; i<c3d_empty.header().nbAnalogByFrame(); ++i)
            analog.addSubframe(subframe);
        f.add(pts, analog);
        c3d_empty.addFrame(f);
        c3d_empty.addFrame(f);

        // Write the brand new c3d
        c3d_empty.write("emptyC3d.c3d");

        // Read it back!
        ezc3d::c3d emptyC3d("emptyC3d.c3d");
        emptyC3d.parameters().print();
    }
    return 0;
}

#else
// EXAMPLE TO BUILD
int main()
{
    {
        ezc3d::c3d c3d("markers_analogs.c3d");

        // Add two new markers to the c3d (one filled with zeros, the other one with data)
        c3d.addMarker("new_marker1"); // Add empty
        std::vector<ezc3d::DataNS::Frame> frames_point;
        ezc3d::DataNS::Points3dNS::Points pts_new;
        ezc3d::DataNS::Points3dNS::Point pt_new;
        pt_new.name("new_marker2");
        pt_new.x(1.0);
        pt_new.y(2.0);
        pt_new.z(3.0);
        pts_new.add(pt_new);
        for (int i=0; i<c3d.data().frames().size(); ++i){
            ezc3d::DataNS::Frame frame;
            frame.add(pts_new);
            frames_point.push_back(frame);
        }
        c3d.addMarker(frames_point); // Add the previously created

        // Add a new analog to the c3d (one filled with zeros, the other one with data)
        c3d.addAnalog("new_analog1"); // add the empty
        std::vector<ezc3d::DataNS::Frame> frames_analog;
        ezc3d::DataNS::AnalogsNS::SubFrame subframes_analog;
        ezc3d::DataNS::AnalogsNS::Channel emptyChannel;
        emptyChannel.name("new_analog2");
        ezc3d::DataNS::Frame frame;
        subframes_analog.channels_nonConst().push_back(emptyChannel);
        for (int sf=0; sf<c3d.header().nbAnalogByFrame(); ++sf){
            subframes_analog.channels_nonConst()[0].value(sf+1);
            frame.analogs_nonConst().addSubframe(subframes_analog);
        }
        for (int f=0; f<c3d.data().frames().size(); ++f)
            frames_analog.push_back(frame);
        c3d.addAnalog(frames_analog);

        // Add a new frame
        ezc3d::DataNS::Frame f;
        std::vector<std::string>labels(c3d.parameters().group("POINT").parameter("LABELS").valuesAsString());
        int nPoints(c3d.parameters().group("POINT").parameter("USED").valuesAsInt()[0]);
        ezc3d::DataNS::Points3dNS::Points pts;
        for (int i=0; i<nPoints; ++i){
            ezc3d::DataNS::Points3dNS::Point pt;
            pt.name(labels[i]);
            pt.x(1.0);
            pt.y(2.0);
            pt.z(3.0);
            pts.add(pt);
        }
        ezc3d::DataNS::AnalogsNS::Analogs analog;
        ezc3d::DataNS::AnalogsNS::SubFrame subframe;
        for (int i=0; i<c3d.header().nbAnalogs(); ++i){
            ezc3d::DataNS::AnalogsNS::Channel c;
            c.value(i+1);
            subframe.addChannel(c);
        }
        for (int i=0; i<c3d.header().nbAnalogByFrame(); ++i)
            analog.addSubframe(subframe);
        f.add(pts, analog);
        c3d.addFrame(f);


        // Add new parameters
        ezc3d::ParametersNS::GroupNS::Parameter p1("new_param_1");
        p1.set(std::vector<int>() = {2}, {1});
        c3d.addParameter("new_group_1", p1);
        ezc3d::ParametersNS::GroupNS::Parameter p2("new_param_2");
        p2.set(std::vector<int>() = {1, 1, 2, 3, 5, 8}, {3, 2}); // TESTER LE READER DANS MATLAB!
        c3d.addParameter("new_group_1", p2);
        ezc3d::ParametersNS::GroupNS::Parameter p3("new_param_1");
        p3.set(std::vector<std::string>() = {"value1", "longer_value1", "value2", "longer_value2"}, {20, 2, 2});
        c3d.addParameter("new_group_2", p3);

        // write the changed c3d
        c3d.write("augmentedC3d.c3d");

        // Read it back!
        ezc3d::c3d augmentedC3d("augmentedC3d.c3d");
        augmentedC3d.parameters().print();
    }

    {
        // Create an empty fresh c3d
        ezc3d::c3d c3d_empty;
        ezc3d::ParametersNS::GroupNS::Parameter t("SCALE");
        t.set(std::vector<float>(), {0});


        // Fill it with some values
        ezc3d::ParametersNS::GroupNS::Parameter pointRate("RATE");
        pointRate.set(std::vector<float>() = {100}, {1});
        c3d_empty.addParameter("POINT", pointRate);

        ezc3d::ParametersNS::GroupNS::Parameter analogRate("RATE");
        analogRate.set(std::vector<float>() = {1000}, {1});
        c3d_empty.addParameter("ANALOG", analogRate);

        c3d_empty.addMarker("new_marker1"); // Add empty
        c3d_empty.addMarker("new_marker2"); // Add empty
        c3d_empty.addAnalog("new_analog1"); // add the empty
        c3d_empty.addAnalog("new_analog2"); // add the empty
        // Add a new frame
        ezc3d::DataNS::Frame f;
        std::vector<std::string>labels(c3d_empty.parameters().group("POINT").parameter("LABELS").valuesAsString());
        int nPoints(c3d_empty.parameters().group("POINT").parameter("USED").valuesAsInt()[0]);
        ezc3d::DataNS::Points3dNS::Points pts;
        for (int i=0; i<nPoints; ++i){
            ezc3d::DataNS::Points3dNS::Point pt;
            pt.name(labels[i]);
            pt.x(1.0);
            pt.y(2.0);
            pt.z(3.0);
            pts.add(pt);
        }
        ezc3d::DataNS::AnalogsNS::Analogs analog;
        ezc3d::DataNS::AnalogsNS::SubFrame subframe;
        for (int i=0; i<c3d_empty.header().nbAnalogs(); ++i){
            ezc3d::DataNS::AnalogsNS::Channel c;
            c.value(i+1);
            subframe.addChannel(c);
        }
        for (int i=0; i<c3d_empty.header().nbAnalogByFrame(); ++i)
            analog.addSubframe(subframe);
        f.add(pts, analog);
        c3d_empty.addFrame(f);
        c3d_empty.addFrame(f);

        // Write the brand new c3d
        c3d_empty.write("emptyC3d.c3d");

        // Read it back!
        ezc3d::c3d emptyC3d("emptyC3d.c3d");
        emptyC3d.parameters().print();
    }
    return 0;
}
#endif

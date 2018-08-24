#ifndef __DATA_H__
#define __DATA_H__

#include <sstream>
#include "ezc3d.h"



class EZC3D_API ezc3d::DataNS::Data{
public:
    Data();
    Data(ezc3d::c3d &file);
    void print() const;
    void write(std::fstream &f) const;

    // Getter
    void frame(const ezc3d::DataNS::Frame& f, int j = -1);
    std::vector<ezc3d::DataNS::Frame>& frames_nonConst();
    const std::vector<ezc3d::DataNS::Frame>& frames() const;
    const ezc3d::DataNS::Frame& frame(int idx) const;

protected:
    std::vector<ezc3d::DataNS::Frame> _frames;
};

class EZC3D_API ezc3d::DataNS::Frame{
public:
    Frame();
    void print() const;
    void write(std::fstream &f) const;

    void add(const ezc3d::DataNS::AnalogsNS::Analogs &analog_frame);
    void add(const ezc3d::DataNS::Points3dNS::Points &point3d_frame);
    void add(const ezc3d::DataNS::Frame &frame);
    void add(const ezc3d::DataNS::Points3dNS::Points &point3d_frame, const ezc3d::DataNS::AnalogsNS::Analogs &analog_frame);

    ezc3d::DataNS::Points3dNS::Points& points_nonConst() const;
    const ezc3d::DataNS::Points3dNS::Points& points() const;
    ezc3d::DataNS::AnalogsNS::Analogs& analogs_nonConst() const;
    const ezc3d::DataNS::AnalogsNS::Analogs& analogs() const;

protected:

    std::shared_ptr<ezc3d::DataNS::Points3dNS::Points> _points; // All points for this frame
    std::shared_ptr<ezc3d::DataNS::AnalogsNS::Analogs> _analogs; // All subframe for all analogs
};

class EZC3D_API ezc3d::DataNS::Points3dNS::Points{
public:
    Points();
    Points(int nMarkers);

    void add(const ezc3d::DataNS::Points3dNS::Point& p);
    void replace(int idx, const ezc3d::DataNS::Points3dNS::Point& p);
    void print() const;
    void write(std::fstream &f) const;

    const std::vector<ezc3d::DataNS::Points3dNS::Point>& points() const;
    std::vector<ezc3d::DataNS::Points3dNS::Point>& points_nonConst();
    int pointIdx(const std::string& pointName) const;
    const ezc3d::DataNS::Points3dNS::Point& point(int idx) const;
    const ezc3d::DataNS::Points3dNS::Point& point(const std::string& pointName) const;

protected:
    std::vector<ezc3d::DataNS::Points3dNS::Point> _points;
};

class EZC3D_API ezc3d::DataNS::Points3dNS::Point{
public:
    void print() const;
    Point();
    void write(std::fstream &f) const;

    float x() const;
    void x(float x);

    float y() const;
    void y(float y);

    float z() const;
    void z(float z);

	const std::vector<float> data() const;

    float residual() const;
    void residual(float residual);
    const std::string& name() const;
    void name(const std::string &name);

protected:
	std::vector<float> _data;
    std::string _name;
};

class EZC3D_API ezc3d::DataNS::AnalogsNS::Analogs{
public:
    Analogs();
    Analogs(int nSubframes);
    void print() const;
    void write(std::fstream &f) const;

    const std::vector<ezc3d::DataNS::AnalogsNS::SubFrame>& subframes() const;
    std::vector<ezc3d::DataNS::AnalogsNS::SubFrame>& subframes_nonConst();
    const ezc3d::DataNS::AnalogsNS::SubFrame& subframe(int idx) const;
    void addSubframe(const ezc3d::DataNS::AnalogsNS::SubFrame& subframe);
    void replaceSubframe(int idx, const SubFrame& subframe);

protected:
    std::vector<ezc3d::DataNS::AnalogsNS::SubFrame> _subframe;
};

class EZC3D_API ezc3d::DataNS::AnalogsNS::SubFrame{
public:
    SubFrame();
    SubFrame(int nChannels);
    void print() const;
    void write(std::fstream &f) const;

    void addChannel(const ezc3d::DataNS::AnalogsNS::Channel& channel);
    void replaceChannel(int idx, const ezc3d::DataNS::AnalogsNS::Channel& channel);
    void addChannels(const std::vector<ezc3d::DataNS::AnalogsNS::Channel>& allChannelsData);
    std::vector<ezc3d::DataNS::AnalogsNS::Channel>& channels_nonConst();
    const std::vector<ezc3d::DataNS::AnalogsNS::Channel>& channels() const;
    const ezc3d::DataNS::AnalogsNS::Channel& channel(int idx) const;
    const ezc3d::DataNS::AnalogsNS::Channel& channel(std::string channelName) const;
protected:
    std::vector<ezc3d::DataNS::AnalogsNS::Channel> _channels;
};

class EZC3D_API ezc3d::DataNS::AnalogsNS::Channel{
public:
    void print() const;
    void write(std::fstream &f) const;

    float value() const;
    void value(float v);

    const std::string& name() const;
    void name(const std::string &name);

protected:
    std::string _name;
    float _value;
};




#endif

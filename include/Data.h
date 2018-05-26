#ifndef __DATA_H__
#define __DATA_H__

#include "ezc3d.h"



class EZC3D_API ezc3d::DataNS::Data{
public:
    Data(ezc3d::c3d &file);
    void print() const;

    // Getter
    const std::vector<ezc3d::DataNS::Frame>& frames() const;
    const ezc3d::DataNS::Frame& frame(int idx) const;

protected:
    std::vector<ezc3d::DataNS::Frame> _frames;
};

class EZC3D_API ezc3d::DataNS::Frame{
public:

    void print() const;

    void add(ezc3d::DataNS::AnalogsNS::Analogs analog_frame);
    void add(ezc3d::DataNS::Points3dNS::Points point3d_frame);
    void add(ezc3d::DataNS::Points3dNS::Points point3d_frame, ezc3d::DataNS::AnalogsNS::Analogs analog_frame);

    const ezc3d::DataNS::Points3dNS::Points& points() const;
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

    const std::vector<Point>& points() const;
    std::vector<Point>& points_nonConst();
    const ezc3d::DataNS::Points3dNS::Point& point(int idx) const;
    const ezc3d::DataNS::Points3dNS::Point& point(const std::string& pointName) const;

protected:
    std::vector<ezc3d::DataNS::Points3dNS::Point> _points;
};

class EZC3D_API ezc3d::DataNS::Points3dNS::Point{
public:
    void print() const;
    Point();

    float x() const;
    void x(float x);

    float y() const;
    void y(float y);

    float z() const;
    void z(float z);

    const std::shared_ptr<float[]> data() const;

    float residual() const;
    void residual(float residual);
    const std::string& name() const;
    void name(const std::string &name);

protected:
    std::shared_ptr<float[]> _data;
    std::string _name;
    int _idxInData;
};

class EZC3D_API ezc3d::DataNS::AnalogsNS::Analogs{
public:
    Analogs();
    Analogs(int nSubframes);
    void print() const;

    const std::vector<ezc3d::DataNS::AnalogsNS::SubFrame>& subframes() const;
    std::vector<ezc3d::DataNS::AnalogsNS::SubFrame>& subframes_nonConst();
    const ezc3d::DataNS::AnalogsNS::SubFrame& subframe(int idx) const;
    void addSubframe(const SubFrame& subframe);
    void replaceSubframe(int idx, const SubFrame& subframe);

protected:
    std::vector<ezc3d::DataNS::AnalogsNS::SubFrame> _subframe;
};

class EZC3D_API ezc3d::DataNS::AnalogsNS::SubFrame{
public:
    SubFrame();
    SubFrame(int nChannels);
    void print() const;

    void addChannel(const ezc3d::DataNS::AnalogsNS::Channel& channel);
    void replaceChannel(int idx, const ezc3d::DataNS::AnalogsNS::Channel& channel);
    void addChannels(const std::vector<Channel>& allChannelsData);
    const std::vector<Channel>& channels() const;
    const ezc3d::DataNS::AnalogsNS::Channel& channel(int idx) const;
    const ezc3d::DataNS::AnalogsNS::Channel& channel(std::string channelName) const;
protected:
    std::vector<Channel> _channels;
};

class EZC3D_API ezc3d::DataNS::AnalogsNS::Channel{
public:
    void print() const;

    float value() const;
    void value(float v);

    const std::string& name() const;
    void name(const std::string &name);

protected:
    std::string _name;
    float _value;
};




#endif

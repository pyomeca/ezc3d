#ifndef __DATA_H__
#define __DATA_H__

#include "ezC3D.h"
#include <stdexcept>
#include <iostream>

class ezC3D::DataNS::Data{
public:
    Data(ezC3D::C3D &file);
    void print() const;

    // Getter
    const std::vector<ezC3D::DataNS::Frame>& frames() const;
    const ezC3D::DataNS::Frame& frame(int idx) const;

protected:
    std::vector<ezC3D::DataNS::Frame> _frames;
};

class ezC3D::DataNS::Frame{
public:

    void print() const;

    void add(ezC3D::DataNS::AnalogsNS::Analogs analog_frame);
    void add(ezC3D::DataNS::Points3dNS::Points3d point3d_frame);
    void add(ezC3D::DataNS::Points3dNS::Points3d point3d_frame, ezC3D::DataNS::AnalogsNS::Analogs analog_frame);

    const std::shared_ptr<ezC3D::DataNS::Points3dNS::Points3d>& points() const;
    const std::shared_ptr<ezC3D::DataNS::AnalogsNS::Analogs>& analogs() const;

protected:

    std::shared_ptr<ezC3D::DataNS::Points3dNS::Points3d> _points; // All points for this frame
    std::shared_ptr<ezC3D::DataNS::AnalogsNS::Analogs> _analogs; // All subframe for all analogs
};

class ezC3D::DataNS::Points3dNS::Points3d{
public:
    void add(ezC3D::DataNS::Points3dNS::Point p);
    void print() const;

    const std::vector<Point>& points() const;
    const Point& point(int idx) const;
    const Point& point(const std::string& pointName) const;

protected:
    std::vector<Point> _points;
};

class ezC3D::DataNS::Points3dNS::Point{
public:
    void print() const;
    Point();

    float x() const;
    void x(float x);

    float y() const;
    void y(float y);

    float z() const;
    void z(float z);

    float residual() const;
    void residual(float residual);
    const std::string& name() const;
    void name(const std::string &name);

    int idxInData() const;
    void idxInData(int idxInData);

protected:
    float _x;
    float _y;
    float _z;
    float _residual;

    std::string _name;
    int _idxInData;
};

class ezC3D::DataNS::AnalogsNS::Analogs{
public:
    void print();

    const std::vector<SubFrame>& subframes() const;
    const SubFrame& subframe(int idx) const;
    void addSubframe(const SubFrame& subframe);

protected:
    std::vector<SubFrame> _subframe;
};

class ezC3D::DataNS::AnalogsNS::SubFrame{
public:
    class Channel;

    void print() const;

    void addChannel(Channel allChannelsData);
    void addChannels(const std::vector<Channel>& allChannelsData);
    const std::vector<Channel>& channels() const;
    const Channel& channel(int idx) const;
    const Channel& channel(std::string channelName) const;
protected:
    std::vector<Channel> _channels;
};

class ezC3D::DataNS::AnalogsNS::SubFrame::Channel{
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

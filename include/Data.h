#ifndef __DATA_H__
#define __DATA_H__

#include "ezC3D.h"
#include <stdexcept>
#include <iostream>
#include <memory>

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
    void add(ezC3D::DataNS::Points3dNS::Points point3d_frame);
    void add(ezC3D::DataNS::Points3dNS::Points point3d_frame, ezC3D::DataNS::AnalogsNS::Analogs analog_frame);

    const ezC3D::DataNS::Points3dNS::Points& points() const;
    const ezC3D::DataNS::AnalogsNS::Analogs& analogs() const;

protected:

    std::shared_ptr<ezC3D::DataNS::Points3dNS::Points> _points; // All points for this frame
    std::shared_ptr<ezC3D::DataNS::AnalogsNS::Analogs> _analogs; // All subframe for all analogs
};

class ezC3D::DataNS::Points3dNS::Points{
public:
    void add(ezC3D::DataNS::Points3dNS::Point p);
    void print() const;

    const std::vector<Point>& points() const;
    const ezC3D::DataNS::Points3dNS::Point& point(int idx) const;
    const ezC3D::DataNS::Points3dNS::Point& point(const std::string& pointName) const;

protected:
    std::vector<ezC3D::DataNS::Points3dNS::Point> _points;
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

class ezC3D::DataNS::AnalogsNS::Analogs{
public:
    void print() const;

    const std::vector<ezC3D::DataNS::AnalogsNS::SubFrame>& subframes() const;
    const ezC3D::DataNS::AnalogsNS::SubFrame& subframe(int idx) const;
    void addSubframe(const SubFrame& subframe);

protected:
    std::vector<ezC3D::DataNS::AnalogsNS::SubFrame> _subframe;
};

class ezC3D::DataNS::AnalogsNS::SubFrame{
public:
    void print() const;

    void addChannel(Channel allChannelsData);
    void addChannels(const std::vector<Channel>& allChannelsData);
    const std::vector<Channel>& channels() const;
    const ezC3D::DataNS::AnalogsNS::Channel& channel(int idx) const;
    const ezC3D::DataNS::AnalogsNS::Channel& channel(std::string channelName) const;
protected:
    std::vector<Channel> _channels;
};

class ezC3D::DataNS::AnalogsNS::Channel{
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

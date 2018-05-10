#ifndef __DATA_H__
#define __DATA_H__

#include "ezC3D.h"
#include <stdexcept>
#include <iostream>

class ezC3D::Data::Data{
public:
    Data(ezC3D::C3D &file);

    class Frame;
    class Points3d;
    class Analogs;

    void print() const;

    // Getter
    const std::vector<ezC3D::Data::Data::Frame>& frames() const;
    const ezC3D::Data::Data::Frame& frame(int idx) const;

protected:
    std::vector<ezC3D::Data::Data::Frame> _frames;
};

class ezC3D::Data::Data::Frame{
public:

    void print() const;

    void add(ezC3D::Data::Data::Analogs analog_frame);
    void add(ezC3D::Data::Data::Points3d point3d_frame);
    void add(ezC3D::Data::Data::Points3d point3d_frame, ezC3D::Data::Data::Analogs analog_frame);

    const std::shared_ptr<ezC3D::Data::Data::Points3d>& points() const;
    const std::shared_ptr<ezC3D::Data::Data::Analogs>& analogs() const;

protected:

    std::shared_ptr<ezC3D::Data::Data::Points3d> _points; // All points for this frame
    std::shared_ptr<ezC3D::Data::Data::Analogs> _analogs; // All subframe for all analogs
};

class ezC3D::Data::Data::Points3d{
public:
    class Point;
    void add(ezC3D::Data::Data::Points3d::Point p);
    void print() const;

    const std::vector<Point>& points() const;
    const Point& point(int idx) const;
    const Point& point(const std::string& pointName) const;

protected:
    std::vector<Point> _points;
};

class ezC3D::Data::Data::Points3d::Point{
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

class ezC3D::Data::Data::Analogs{
public:
    class SubFrame;

    void print();

    const std::vector<SubFrame>& subframes() const;
    const SubFrame& subframe(int idx) const;
    void addSubframe(const SubFrame& subframe);

protected:
    std::vector<SubFrame> _subframe;
};

class ezC3D::Data::Data::Analogs::SubFrame{
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

class ezC3D::Data::Data::Analogs::SubFrame::Channel{
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

#ifndef __DATA_H__
#define __DATA_H__

#include "ezC3D.h"
#include <stdexcept>
#include <iostream>

class ezC3D::Data{
public:
    Data(ezC3D &file);

    class Frame;
    class Points3d;
    class Analogs;

    void print() const;

    // Getter
    const std::vector<ezC3D::Data::Frame>& frames() const;
    const ezC3D::Data::Frame& frame(int idx) const;

protected:
    std::vector<ezC3D::Data::Frame> _frames;
};

class ezC3D::Data::Frame{
public:

    void print() const;

    void add(ezC3D::Data::Analogs analog_frame);
    void add(ezC3D::Data::Points3d point3d_frame);
    void add(ezC3D::Data::Points3d point3d_frame, ezC3D::Data::Analogs analog_frame);

protected:

    std::shared_ptr<ezC3D::Data::Points3d> _points; // All points for this frame
    std::shared_ptr<ezC3D::Data::Analogs> _analogs; // All subframe for all analogs
};

class ezC3D::Data::Points3d{
public:
    class Point;
    void add(ezC3D::Data::Points3d::Point p);
    void print() const;

    const std::vector<Point>& points() const;
    const Point& point(int idx) const;

protected:
    std::vector<Point> _points;
};

class ezC3D::Data::Points3d::Point{
public:
    void print() const;

    float x() const;
    void x(float x);

    float y() const;
    void y(float y);

    float z() const;
    void z(float z);

    float residual() const;
    void residual(float residual);
protected:
    float _x;
    float _y;
    float _z;
    float _residual;
};

class ezC3D::Data::Analogs{
public:
    class Channel;

    void print() const;

    void addChannel(Channel allChannelsData);
    void addChannels(const std::vector<Channel>& allChannelsData);
    const std::vector<Channel>& channels() const;
    Channel channel(int channel) const;
protected:
    std::vector<Channel> _channels;
};

class ezC3D::Data::Analogs::Channel{
public:
    float value() const;
    void value(float v);

    void print() const;
protected:
    float _value;
};




#endif

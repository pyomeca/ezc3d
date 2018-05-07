#ifndef __DATA_HOLDER_H__
#define __DATA_HOLDER_H__

#include "ezC3D.h"
#include <stdexcept>

class ezC3D::Point3d{
public:
    void print();

    float x();
    void x(float x);

    float y();
    void y(float y);

    float z();
    void z(float z);

    float residual();
    void residual(float residual);
protected:
    float _x;
    float _y;
    float _z;
    float _residual;
};

class ezC3D::Analog{
public:
    void print();

    void addChannel(float allChannelsData);
    void addChannels(const std::vector<float>& allChannelsData);
    const std::vector<float>& data() const;
    float data(int channel) const;
protected:
    std::vector<float> _channels;
};



class ezC3D::Frame{
public:
    void print();

    void add(ezC3D::Analog analog_frame);
    void add(ezC3D::Point3d point3d_frame);
    void add(ezC3D::Point3d point3d_frame, ezC3D::Analog analog_frame);

protected:
    ezC3D::Analog _analogs; // All subframe for all analogs
    ezC3D::Point3d _points; // All points for this frame
};

#endif

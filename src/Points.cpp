#define EZC3D_API_EXPORTS
#include "Points.h"
// Implementation of Points class

// Point3d data
ezc3d::DataNS::Points3dNS::Points::Points()
{

}

ezc3d::DataNS::Points3dNS::Points::Points(size_t nbMarkers)
{
    _points.resize(nbMarkers);
}

void ezc3d::DataNS::Points3dNS::Points::point(const ezc3d::DataNS::Points3dNS::Point &point, size_t idx)
{
    if (idx == SIZE_MAX)
        _points.push_back(point);
    else{
        if (idx >= nbPoints())
            _points.resize(idx+1);
        _points[idx] = point;
    }
}

void ezc3d::DataNS::Points3dNS::Points::print() const
{
    for (size_t i = 0; i < nbPoints(); ++i)
        point(i).print();
}

void ezc3d::DataNS::Points3dNS::Points::write(std::fstream &f) const
{
    for (size_t i = 0; i < nbPoints(); ++i)
        point(i).write(f);
}

size_t ezc3d::DataNS::Points3dNS::Points::nbPoints() const
{
    return _points.size();
}

size_t ezc3d::DataNS::Points3dNS::Points::pointIdx(const std::string &pointName) const
{
    for (size_t i = 0; i < nbPoints(); ++i)
        if (!point(i).name().compare(pointName))
            return i;
    throw std::invalid_argument(pointName + " was not found in points data");
}
const ezc3d::DataNS::Points3dNS::Point& ezc3d::DataNS::Points3dNS::Points::point(size_t idx) const
{
    return _points.at(idx);
}
const ezc3d::DataNS::Points3dNS::Point &ezc3d::DataNS::Points3dNS::Points::point(const std::string &pointName) const
{
    return point(pointIdx(pointName));
}
void ezc3d::DataNS::Points3dNS::Point::print() const
{
    std::cout << name() << " = [" << x() << ", " << y() << ", " << z() << "]; Residual = " << residual() << std::endl;
}

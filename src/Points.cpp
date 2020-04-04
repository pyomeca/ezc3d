#define EZC3D_API_EXPORTS
///
/// \file Points.cpp
/// \brief Implementation of Points class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "Points.h"

// Point3d data
ezc3d::DataNS::Points3dNS::Points::Points() {
}

ezc3d::DataNS::Points3dNS::Points::Points(
        size_t nbPoints) {
    _points.resize(nbPoints);
}

void ezc3d::DataNS::Points3dNS::Points::print() const {
    for (size_t i = 0; i < nbPoints(); ++i)
        point(i).print();
}

void ezc3d::DataNS::Points3dNS::Points::write(
        std::fstream &f,
        float scaleFactor) const {
    for (size_t i = 0; i < nbPoints(); ++i)
        point(i).write(f, scaleFactor);
}

size_t ezc3d::DataNS::Points3dNS::Points::nbPoints() const {
    return _points.size();
}

const ezc3d::DataNS::Points3dNS::Point&
ezc3d::DataNS::Points3dNS::Points::point(
        size_t idx) const {
    try {
        return _points.at(idx);
    } catch(std::out_of_range) {
        throw std::out_of_range(
                    "Points::point method is trying to access the point "
                    + std::to_string(idx) +
                    " while the maximum number of points is "
                    + std::to_string(nbPoints()) + ".");
    }
}

ezc3d::DataNS::Points3dNS::Point&
ezc3d::DataNS::Points3dNS::Points::point(
        size_t idx) {
    try {
        return _points.at(idx);
    } catch(std::out_of_range) {
        throw std::out_of_range(
                    "Points::point method is trying to access the point "
                    + std::to_string(idx) +
                    " while the maximum number of points is "
                    + std::to_string(nbPoints()) + ".");
    }
}

void ezc3d::DataNS::Points3dNS::Points::point(
        const ezc3d::DataNS::Points3dNS::Point &point,
        size_t idx) {
    if (idx == SIZE_MAX) {
        _points.push_back(point);
    }
    else {
        if (idx >= nbPoints()) {
            _points.resize(idx+1);
        }
        _points[idx] = point;
    }
}

const std::vector<ezc3d::DataNS::Points3dNS::Point>&
ezc3d::DataNS::Points3dNS::Points::points() const {
    return _points;
}

bool ezc3d::DataNS::Points3dNS::Points::isEmpty() const {
    for (Point point : points())
        if (!point.isEmpty())
            return false;
    return true;
}

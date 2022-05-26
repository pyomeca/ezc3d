#define EZC3D_API_EXPORTS
///
/// \file Analogs.cpp
/// \brief Implementation of Analogs class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "ezc3d/Analogs.h"
#include "ezc3d/ezc3d.h"
#include "ezc3d/Header.h"
#include <iostream>
#include <stdexcept>

ezc3d::DataNS::AnalogsNS::Analogs::Analogs() {

}

ezc3d::DataNS::AnalogsNS::Analogs::Analogs(
        ezc3d::c3d &c3d,
        std::fstream &file,
        const AnalogsNS::Info& info)
{
    nbSubframes(c3d.header().nbAnalogByFrame());
    for (size_t k = 0; k < c3d.header().nbAnalogByFrame(); ++k){
        subframe(ezc3d::DataNS::AnalogsNS::SubFrame(c3d, file, info), k);
    }
}

void ezc3d::DataNS::AnalogsNS::Analogs::print() const {
    for (size_t i = 0; i < nbSubframes(); ++i) {
        std::cout << "Subframe = " << i << "\n";
        subframe(i).print();
        std::cout << "\n";
    }
}

void ezc3d::DataNS::AnalogsNS::Analogs::write(
        std::fstream &f,
        std::vector<double> scaleFactors) const {
    for (size_t i = 0; i < nbSubframes(); ++i) {
        subframe(i).write(f, scaleFactors);
    }
}

size_t ezc3d::DataNS::AnalogsNS::Analogs::nbSubframes() const {
    return _subframe.size();
}

void ezc3d::DataNS::AnalogsNS::Analogs::nbSubframes(
        size_t nbSubframes) {
    _subframe.resize(nbSubframes);
}

const ezc3d::DataNS::AnalogsNS::SubFrame&
ezc3d::DataNS::AnalogsNS::Analogs::subframe(
        size_t idx) const {
    try {
        return _subframe.at(idx);
    } catch(std::out_of_range) {
        throw std::out_of_range("Analogs::subframe method is trying to access the subframe "
                                + std::to_string(idx) +
                                " while the maximum number of subframes is "
                                + std::to_string(nbSubframes()) + ".");
    }
}

ezc3d::DataNS::AnalogsNS::SubFrame&
ezc3d::DataNS::AnalogsNS::Analogs::subframe(
        size_t idx) {
    try {
        return _subframe.at(idx);
    } catch(std::out_of_range) {
        throw std::out_of_range("Analogs::subframe method is trying to access the subframe "
                                + std::to_string(idx) +
                                " while the maximum number of subframes is "
                                + std::to_string(nbSubframes()) + ".");
    }
}

void ezc3d::DataNS::AnalogsNS::Analogs::subframe(
        const ezc3d::DataNS::AnalogsNS::SubFrame& subframe,
        size_t idx) {
    if (idx == SIZE_MAX) {
        _subframe.push_back(subframe);
    }
    else {
        if (idx >= nbSubframes()) {
            _subframe.resize(idx+1);
        }
        _subframe[idx] = subframe;
    }
}

const std::vector<ezc3d::DataNS::AnalogsNS::SubFrame>&
ezc3d::DataNS::AnalogsNS::Analogs::subframes() const {
    return _subframe;
}

bool ezc3d::DataNS::AnalogsNS::Analogs::isEmpty() const {
    for (SubFrame subframe : subframes()) {
        if (!subframe.isEmpty()) {
            return false;
        }
    }
    return true;
}

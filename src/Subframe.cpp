#define EZC3D_API_EXPORTS
///
/// \file Subframe.cpp
/// \brief Implementation of Subframe class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "Subframe.h"

ezc3d::DataNS::AnalogsNS::SubFrame::SubFrame() {

}

void ezc3d::DataNS::AnalogsNS::SubFrame::print() const {
    for (size_t i = 0; i < nbChannels(); ++i){
        channel(i).print();
    }
}

void ezc3d::DataNS::AnalogsNS::SubFrame::write(
        std::fstream &f,
        std::vector<double> scaleFactors) const {
    for (size_t i = 0; i < nbChannels(); ++i){
        if (scaleFactors.size() > 1){
            channel(i).write(f, scaleFactors[i]);
        }
        else {
            channel(i).write(f, scaleFactors[0]);
        }
    }
}

size_t ezc3d::DataNS::AnalogsNS::SubFrame::nbChannels() const {
    return _channels.size();
}

void ezc3d::DataNS::AnalogsNS::SubFrame::nbChannels(
        size_t nChannels) {
    _channels.resize(nChannels);
}

const ezc3d::DataNS::AnalogsNS::Channel&
ezc3d::DataNS::AnalogsNS::SubFrame::channel(size_t idx) const {
    try {
        return _channels.at(idx);
    } catch(std::out_of_range) {
        throw std::out_of_range(
                    "Subframe::channel method is trying to access the channel "
                    + std::to_string(idx) +
                    " while the maximum number of channels is "
                    + std::to_string(nbChannels()) + ".");
    }
}

ezc3d::DataNS::AnalogsNS::Channel&
ezc3d::DataNS::AnalogsNS::SubFrame::channel(
        size_t idx) {
    try {
        return _channels.at(idx);
    } catch(std::out_of_range) {
        throw std::out_of_range(
                    "Subframe::channel method is trying to access the channel "
                    + std::to_string(idx) +
                    " while the maximum number of channels is "
                    + std::to_string(nbChannels()) + ".");
    }
}

void ezc3d::DataNS::AnalogsNS::SubFrame::channel(
        const ezc3d::DataNS::AnalogsNS::Channel &channel,
        size_t idx) {
    if (idx == SIZE_MAX)
        _channels.push_back(channel);
    else{
        if (idx >= nbChannels())
            _channels.resize(idx+1);
        _channels[idx] = channel;
    }
}

const std::vector<ezc3d::DataNS::AnalogsNS::Channel>&
ezc3d::DataNS::AnalogsNS::SubFrame::channels() const {
    return _channels;
}

bool ezc3d::DataNS::AnalogsNS::SubFrame::isEmpty() const {
    for (Channel channel : channels())
        if (!channel.isEmpty())
            return false;
    return true;
}

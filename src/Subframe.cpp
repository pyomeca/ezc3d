#define EZC3D_API_EXPORTS
#include "Subframe.h"
// Implementation of Subframe class

ezc3d::DataNS::AnalogsNS::SubFrame::SubFrame()
{

}

ezc3d::DataNS::AnalogsNS::SubFrame::SubFrame(size_t nChannels)
{
    _channels.resize(nChannels);
}

void ezc3d::DataNS::AnalogsNS::SubFrame::print() const
{
    for (size_t i = 0; i < nbChannels(); ++i){
        channel(i).print();
    }
}

void ezc3d::DataNS::AnalogsNS::SubFrame::write(std::fstream &f) const
{
    for (size_t i = 0; i < nbChannels(); ++i){
        channel(i).write(f);
    }
}

size_t ezc3d::DataNS::AnalogsNS::SubFrame::nbChannels() const
{
    return _channels.size();
}

size_t ezc3d::DataNS::AnalogsNS::SubFrame::channelIdx(const std::string &channelName) const
{
    for (size_t i = 0; i < nbChannels(); ++i)
        if (!channel(i).name().compare(channelName))
            return i;
    throw std::invalid_argument("Subframe::channelIdx could not find " + channelName +
                                " in the analogous data set");
}

const ezc3d::DataNS::AnalogsNS::Channel &ezc3d::DataNS::AnalogsNS::SubFrame::channel(size_t idx) const
{
    try {
        return _channels.at(idx);
    } catch(std::out_of_range) {
        throw std::out_of_range("Subframe::channel method is trying to access the channel "
                                + std::to_string(idx) +
                                " while the maximum number of channels is "
                                + std::to_string(nbChannels()) + ".");
    }
}

ezc3d::DataNS::AnalogsNS::Channel &ezc3d::DataNS::AnalogsNS::SubFrame::channel_nonConst(size_t idx)
{
    try {
        return _channels.at(idx);
    } catch(std::out_of_range) {
        throw std::out_of_range("Subframe::channel method is trying to access the channel "
                                + std::to_string(idx) +
                                " while the maximum number of channels is "
                                + std::to_string(nbChannels()) + ".");
    }
}

const ezc3d::DataNS::AnalogsNS::Channel &ezc3d::DataNS::AnalogsNS::SubFrame::channel(const std::string &channelName) const
{
    return _channels.at(channelIdx(channelName));
}

ezc3d::DataNS::AnalogsNS::Channel &ezc3d::DataNS::AnalogsNS::SubFrame::channel_nonConst(const std::string &channelName)
{
    return _channels.at(channelIdx(channelName));
}

void ezc3d::DataNS::AnalogsNS::SubFrame::channel(const ezc3d::DataNS::AnalogsNS::Channel &channel, size_t idx)
{
    if (idx == SIZE_MAX)
        _channels.push_back(channel);
    else{
        if (idx >= nbChannels())
            _channels.resize(idx+1);
        _channels[idx] = channel;
    }
}

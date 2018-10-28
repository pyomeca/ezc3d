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
    throw std::invalid_argument(channelName + " was not found in subframe");
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
const ezc3d::DataNS::AnalogsNS::Channel &ezc3d::DataNS::AnalogsNS::SubFrame::channel(size_t idx) const
{
    return _channels.at(idx);
}
const ezc3d::DataNS::AnalogsNS::Channel &ezc3d::DataNS::AnalogsNS::SubFrame::channel(const std::string &channelName) const
{
    return _channels.at(channelIdx(channelName));
}

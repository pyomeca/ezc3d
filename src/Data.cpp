#define EZC3D_API_EXPORTS
#include "Data.h"
// Implementation of data class


ezc3d::DataNS::Data::Data()
{

}

ezc3d::DataNS::Data::Data(ezc3d::c3d &file)
{
    // Firstly read a dummy value just prior to the data so it moves the pointer to the right place
    file.readInt(ezc3d::DATA_TYPE::BYTE, 256*ezc3d::DATA_TYPE::WORD*(file.header().parametersAddress()-1) + 256*ezc3d::DATA_TYPE::WORD*file.parameters().nbParamBlock() - ezc3d::DATA_TYPE::BYTE, std::ios::beg); // "- BYTE" so it is just prior

    // Initialize some variables
    if (file.header().nbFrames()>0)
        _frames.resize(static_cast<size_t>(file.header().nbFrames()));

    // Get names of the data
    std::vector<std::string> pointNames;
    if (file.header().nb3dPoints() > 0)
        pointNames = file.parameters().group("POINT").parameter("LABELS").valuesAsString();
    std::vector<std::string> analogNames;
    if (file.header().nbAnalogs() > 0)
        analogNames = file.parameters().group("ANALOG").parameter("LABELS").valuesAsString();

    // Read the actual data
    for (size_t j = 0; j < static_cast<size_t>(file.header().nbFrames()); ++j){
        ezc3d::DataNS::Frame f;
        if (file.header().scaleFactor() < 0){ // if it is float
            // Read point 3d
            ezc3d::DataNS::Points3dNS::Points ptsAtAFrame(file.header().nb3dPoints());
            for (size_t i = 0; i < static_cast<size_t>(file.header().nb3dPoints()); ++i){
                ezc3d::DataNS::Points3dNS::Point pt;
                pt.x(file.readFloat());
                pt.y(file.readFloat());
                pt.z(file.readFloat());
                pt.residual(file.readFloat());
                if (i < pointNames.size())
                    pt.name(pointNames[i]);
                else {
                    std::stringstream unlabel;
                    unlabel << "unlabeled_point_" << i;
                    pt.name(unlabel.str());
                }
                ptsAtAFrame.point(pt, i);
            }
            _frames[j].add(ptsAtAFrame); // modified by pts_tp which is an nonconst ref to internal points

            // Read analogs
            ezc3d::DataNS::AnalogsNS::Analogs analog(file.header().nbAnalogByFrame());
            for (int k = 0; k < file.header().nbAnalogByFrame(); ++k){
                ezc3d::DataNS::AnalogsNS::SubFrame sub(file.header().nbAnalogs());
                for (size_t i = 0; i < static_cast<size_t>(file.header().nbAnalogs()); ++i){
                    ezc3d::DataNS::AnalogsNS::Channel c;
                    c.value(file.readFloat());
                    if (i < analogNames.size())
                        c.name(analogNames[i]);
                    else {
                        std::stringstream unlabel;
                        unlabel << "unlabeled_analog_" << i;
                        c.name(unlabel.str());
                    }
                    sub.channel(c, i);
                }
                analog.replaceSubframe(k, sub);
            }
            _frames[j].add(analog);

        }
        else
            throw std::invalid_argument("Points were recorded using int number which is not implemented yet");
    }
}
void ezc3d::DataNS::Data::frame(const ezc3d::DataNS::Frame &frame, size_t idx)
{
    if (idx == SIZE_MAX)
        _frames.push_back(frame);
    else{
        if (idx >= _frames.size())
            _frames.resize(idx+1);
        _frames[static_cast<size_t>(idx)].add(frame);
    }
}
ezc3d::DataNS::Frame &ezc3d::DataNS::Data::frame_nonConst(size_t idx)
{
    try{
        return _frames.at(idx);
    } catch(std::out_of_range) {
        throw std::out_of_range("Wrong number of frames");
    }
}

const ezc3d::DataNS::Frame& ezc3d::DataNS::Data::frame(size_t idx) const
{
    try{
        return _frames.at(idx);
    } catch(std::out_of_range) {
        throw std::out_of_range("Wrong number of frames");
    }
}

size_t ezc3d::DataNS::Data::nbFrames() const
{
    return _frames.size();
}
void ezc3d::DataNS::Data::print() const
{
    for (size_t i = 0; i < nbFrames(); ++i){
        std::cout << "Frame " << i << std::endl;
        frame(i).print();
        std::cout << std::endl;
    }
}

void ezc3d::DataNS::Data::write(std::fstream &f) const
{
    for (size_t i = 0; i < nbFrames(); ++i)
        frame(i).write(f);
}









float ezc3d::DataNS::AnalogsNS::Channel::value() const
{
    return _value;
}
void ezc3d::DataNS::AnalogsNS::Channel::value(float v)
{
    _value = v;
}
void ezc3d::DataNS::AnalogsNS::Channel::print() const
{
    std::cout << "Analog[" << name() << "] = " << value() << std::endl;
}
void ezc3d::DataNS::AnalogsNS::Channel::write(std::fstream &f) const
{
    f.write(reinterpret_cast<const char*>(&_value), ezc3d::DATA_TYPE::FLOAT);
}
const std::string& ezc3d::DataNS::AnalogsNS::Channel::name() const
{
    return _name;
}

void ezc3d::DataNS::AnalogsNS::Channel::name(const std::string &name)
{
    std::string name_copy = name;
    ezc3d::removeTrailingSpaces(name_copy);
    _name = name_copy;
}





const std::vector<ezc3d::DataNS::AnalogsNS::SubFrame>& ezc3d::DataNS::AnalogsNS::Analogs::subframes() const
{
    return _subframe;
}
std::vector<ezc3d::DataNS::AnalogsNS::SubFrame> &ezc3d::DataNS::AnalogsNS::Analogs::subframes_nonConst()
{
    return _subframe;
}
const ezc3d::DataNS::AnalogsNS::SubFrame& ezc3d::DataNS::AnalogsNS::Analogs::subframe(int idx) const
{
    if (idx < 0 || idx >= static_cast<int>(_subframe.size()))
        throw std::out_of_range("Tried to access wrong subframe index for analog data");
    return _subframe[static_cast<size_t>(idx)];
}


ezc3d::DataNS::AnalogsNS::Analogs::Analogs()
{

}
ezc3d::DataNS::AnalogsNS::Analogs::Analogs(int nSubframes)
{
    if (nSubframes < 0)
        throw std::out_of_range("Number of subframes can't be under 0");
    _subframe.resize(static_cast<size_t>(nSubframes));
}
void ezc3d::DataNS::AnalogsNS::Analogs::print() const
{
    for (int i = 0; i < static_cast<int>(subframes().size()); ++i){
        std::cout << "Subframe = " << i << std::endl;
        subframe(i).print();
        std::cout << std::endl;
    }
}
void ezc3d::DataNS::AnalogsNS::Analogs::write(std::fstream &f) const
{
    for (int i = 0; i < static_cast<int>(subframes().size()); ++i){
        subframe(i).write(f);
    }
}
void ezc3d::DataNS::AnalogsNS::Analogs::addSubframe(const ezc3d::DataNS::AnalogsNS::SubFrame& subframe)
{
    _subframe.push_back(subframe);
}
void ezc3d::DataNS::AnalogsNS::Analogs::replaceSubframe(int idx, const ezc3d::DataNS::AnalogsNS::SubFrame& subframe)
{
    if (idx < 0 || idx >= static_cast<int>(_subframe.size()))
        throw std::out_of_range("Tried to access wrong subframe index for analog data");
    _subframe[static_cast<size_t>(idx)] = subframe;
}

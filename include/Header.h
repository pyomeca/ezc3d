#ifndef __HEAEDER_H__
#define __HEAEDER_H__

#include <fstream>

#include "ezc3d.h"

class EZC3D_API ezc3d::Header{
public:
    Header();
    Header(ezc3d::c3d &file);
    void read(ezc3d::c3d &file);
    void print() const;
    void write(std::fstream &f) const;

    // Getter on the parameters
    int parametersAddress() const;
    int checksum() const;
    int nb3dPoints() const;
    int nbAnalogsMeasurement() const;
    int firstFrame() const;
    int lastFrame() const;
    int nbMaxInterpGap() const;
    int scaleFactor() const;
    int dataStartAnalog() const;
    int nbAnalogByFrame() const;
    double frameRate() const;
    int emptyBlock1() const;
    int keyLabelPresent() const;
    int firstBlockKeyLabel() const;
    int fourCharPresent() const;
    int nbEvents() const;
    int emptyBlock2() const;
    const std::vector<float>& eventsTime() const;
    float eventsTime(int i) const;
    std::vector<int> eventsDisplay() const;
    int eventsDisplay(int idx) const;
    int emptyBlock3() const;
    const std::vector<std::string>& eventsLabel() const;
    const std::string& eventsLabel(int idx) const;
    int emptyBlock4() const;
    int nbFrames() const;
    int nbAnalogs() const;

protected:
    // Read the Header
    int _parametersAddress;         // Byte 1.1
    int _checksum;                 // Byte 1.2 ==> 80 if it is a c3d
    int _nb3dPoints;                // Byte 2 ==> number of stored trajectories
    int _nbAnalogsMeasurement;      // Byte 3 ==> number of analog data per point frame
    int _firstFrame; // 1-based!    // Byte 4 ==> first frame in the file
    int _lastFrame;                 // Byte 5 ==> last frame in the file
    int _nbMaxInterpGap;            // Byte 6 ==> maximal gap for interpolation
    int _scaleFactor;               // Byte 7-8 ==> convert int to 3d reference frame, floating point if negative
    int _dataStartAnalog;           // Byte 9 ==> Number of first block for 3D and analog data
    int _nbAnalogByFrame;           // Byte 10 ==> Number of analog by frame
    double _frameRate;              // Byte 11-12 ==> 3d frame rate in Hz (floating point)
    int _emptyBlock1;               // Byte 13-147
    int _keyLabelPresent;           // Byte 148 ==> 12345 if Label and range are present
    int _firstBlockKeyLabel;        // Byte 149 ==> First block of key labels (if present)
    int _fourCharPresent;           // Byte 150 ==> 12345 if 4 char event labels are supported (otherwise 2 char)
    int _nbEvents;                  // Byte 151 ==> Number of defined time events (0 to 18)
    int _emptyBlock2;               // Byte 152
    std::vector<float> _eventsTime; // Byte 153-188 ==> Event times (floating-point) in seconds
    std::vector<int> _eventsDisplay;// Byte 189-197 ==> Event display (0x00 = ON, 0x01 = OFF)
    int _emptyBlock3;               // Byte 198
    std::vector<std::string> _eventsLabel;       // Byte 199-234 ==> Event labels (4 char by label)
    int _emptyBlock4;               // Byte 235-256
};

#endif

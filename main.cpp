#include <iostream>
#include <fstream>
#include <cmath>
#include <string.h>

int hex2int(const char * val){
    int ret(0);
    for (int i=0; i<strlen(val); i++)
        ret |= int((unsigned char)val[i]) * int(pow(0x100, i));
    return ret;
}

int hex2long(const char * val){
    long ret(0);
    for (int i=0; i<strlen(val); i++)
        ret |= long((unsigned char)val[i]) * long(pow(0x100, i));
    return ret;
}

void readChar(std::fstream &file, int nByteToRead, int nByteFromPrevious, const std::ios_base::seekdir &pos, char * c){
    file.seekg (nByteFromPrevious, pos); // Move to number analogs
    file.read (c, nByteToRead);
    c[nByteToRead] = '\0'; // Make sure last char is NULL
}

int readInt(std::fstream &file, int nByteToRead, int nByteFromPrevious, const std::ios_base::seekdir &pos = std::ios::cur){
    char c[nByteToRead + 1];
    readChar(file, nByteToRead, nByteFromPrevious, pos, c);
    return hex2int(c);
}

long readLong(std::fstream &file, int nByteToRead, int nByteFromPrevious,const  std::ios_base::seekdir &pos = std::ios::cur){
    char c[nByteToRead + 1];
    readChar(file, nByteToRead, nByteFromPrevious, pos, c);
    return hex2long(c);
}

int main()
{

    std::fstream file("../markers_analogs.c3d",
                      std::ios::in | std::ios::binary);
    if (file.is_open())
    {
        // Find file size
        std::streampos fileSize = file.tellg();

        // Declare a data receiver a big as the file
        // char* data_tp = new char[fileSize + long(1)];

        // Read the first byte
        char cParametersAddress[50];
        file.read(cParametersAddress, 50);
        int iParametersAddress(cParametersAddress[0]);

        // Read the parameters
        int nBytes3dPoints = 2;
        int nBytes3dAnalogs = 2;
        int nBytesFirstFrame = 2;
        int nBytesLastFrame = 2;
        int nBytesMaxInterpGap = 2;
        int nBytesScaleFactor = 4;
        int nBytesDataStartAnalog = 2;
        int nBytesAnalogByFrame = 2;
        int nBytesFrameRate = 4;
        int nBytesEmptyBlock = 135;

        int nb3dPoints = readInt(file, nBytes3dPoints, iParametersAddress, std::ios::beg);
        int nbAnalogs = readInt(file, nBytes3dAnalogs, nBytes3dPoints);
        int nbFirstFrame = readInt(file, nBytesFirstFrame, nBytes3dAnalogs);
        int nbLastFrame = readInt(file, nBytesLastFrame, nBytesFirstFrame);
        int nbMaxInterpGap = readInt(file, nBytesMaxInterpGap, nBytesLastFrame);
        int scaleFactor = readInt(file, nBytesScaleFactor, nBytesMaxInterpGap);
        int dataStartAnalog = readInt(file, nBytesDataStartAnalog, nBytesScaleFactor);
        int nbAnalogByFrame = readInt(file, nBytesAnalogByFrame, nBytesDataStartAnalog);
        int frameRate = readInt(file, nBytesFrameRate, nBytesAnalogByFrame);
        int emptyBlock = readInt(file, nBytesEmptyBlock, nBytesFrameRate);


        std::cout << "Nb points = " << nb3dPoints << std::endl;
        std::cout << "Nb analogs = " << nbAnalogs << std::endl;
        std::cout << "Nb first frame = " << nbFirstFrame << std::endl;
        std::cout << "Nb last frame = " << nbLastFrame << std::endl;
        std::cout << "Max interp gap = " << nbMaxInterpGap << std::endl;
        std::cout << "Scale factor = " << scaleFactor << std::endl;
        std::cout << "Data start analog = " << dataStartAnalog << std::endl;
        std::cout << "Nb analog by frame = " << nbAnalogByFrame << std::endl;
        std::cout << "Frame rate = " << frameRate << std::endl;
        std::cout << "Empty = " << emptyBlock << std::endl;

        // Terminate
        file.close();

    }
    return 0;
}

#include <iostream>
#include <fstream>
#include <cmath>
#include <string.h>
#include <vector>

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

void readFile(std::fstream &file,
              int nByteToRead,
              char * c,
              int nByteFromPrevious = 0,
              const  std::ios_base::seekdir &pos = std::ios::cur)
{
    file.seekg (nByteFromPrevious, pos); // Move to number analogs
    file.read (c, nByteToRead);
    c[nByteToRead] = '\0'; // Make sure last char is NULL
}

void readChar(std::fstream &file,
              int nByteToRead,
              char * c,
              int nByteFromPrevious = 0,
              const  std::ios_base::seekdir &pos = std::ios::cur)
{
    c = new char[nByteToRead + 1];
    readFile(file, nByteToRead, c, nByteFromPrevious, pos);
}

std::string readString(std::fstream &file,
            int nByteToRead,
            int nByteFromPrevious = 0,
            const std::ios_base::seekdir &pos = std::ios::cur)
{
    char c[nByteToRead + 1];
    readFile(file, nByteToRead, c, nByteFromPrevious, pos);
    return std::string(c);
}

int readInt(std::fstream &file,
            int nByteToRead,
            int nByteFromPrevious = 0,
            const std::ios_base::seekdir &pos = std::ios::cur)
{
    char c[nByteToRead + 1];
    readFile(file, nByteToRead, c, nByteFromPrevious, pos);
    return hex2int(c);
}

double readDouble(std::fstream &file,
                  int nByteToRead,
                  int nByteFromPrevious = 0,
                  const std::ios_base::seekdir &pos = std::ios::cur)
{
    std::cout << "READ DOUBLE IS WRONG" << std::endl;
    char c[nByteToRead + 1];
    readFile(file, nByteToRead, c, nByteFromPrevious, pos);
    return hex2int(c);
}

long readLong(std::fstream &file,
              int nByteToRead,
              int nByteFromPrevious = 0,
              const  std::ios_base::seekdir &pos = std::ios::cur)
{
    char c[nByteToRead + 1];
    readFile(file, nByteToRead, c, nByteFromPrevious, pos);
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

        int BYTE(1);
        int WORD(2*BYTE);

        // Read the Header
        int parametersAddress  (readInt(file, 1*BYTE, 0, std::ios::beg));   // Byte 1.1
        int iChecksum           (readInt(file, 1*BYTE));                    // Byte 1.2 ==> 80 if it is a C3D
        int nb3dPoints          (readInt(file, 1*WORD));                    // Byte 2 ==> number of stored trajectories
        int nbAnalogs           (readInt(file, 1*WORD));                    // Byte 3 ==> number of analog data
        int firstFrame          (readInt(file, 1*WORD) - 1); // 1-based!    // Byte 4 ==> first frame in the file
        int lastFrame           (readInt(file, 1*WORD));                    // Byte 5 ==> last frame in the file
        int nbMaxInterpGap      (readInt(file, 1*WORD));                    // Byte 6 ==> maximal gap for interpolation
        int scaleFactor         (readInt(file, 2*WORD));                    // Byte 7-8 ==> convert int to 3d reference frame, floating point if negative
        int dataStartAnalog     (readInt(file, 1*WORD));                    // Byte 9 ==> Number of first block for 3D and analog data
        int nbAnalogByFrame     (readInt(file, 1*WORD));                    // Byte 10 ==> Number of analog by frame
        double frameRate        (readDouble(file, 2*WORD));                 // Byte 11-12 ==> 3d frame rate in Hz (floating point)
        int emptyBlock1         (readInt(file, 135*WORD));                  // Byte 13-147
        int keyLabelPresent     (readInt(file, 1*WORD));                    // Byte 148 ==> 12345 if Label and range are present
        int firstBlockKeyLabel  (readInt(file, 1*WORD));                    // Byte 149 ==> First block of key labels (if present)
        int fourCharPresent     (readInt(file, 1*WORD));                    // Byte 150 ==> 12345 if 4 char event labels are supported (otherwise 2 char)
        int nbEvents            (readInt(file, 1*WORD));                    // Byte 151 ==> Number of defined time events (0 to 18)
        int emptyBlock2         (readInt(file, 1*WORD));                    // Byte 152
        double eventsTime       (readDouble(file, 36*WORD));                // Byte 153-188 ==> Event times (floating-point) in seconds
        int eventsDisplay       (readInt(file, 9*WORD));                    // Byte 189-197 ==> Event display (0x00 = ON, 0x01 = OFF)
        int emptyBlock3         (readInt(file, 1*WORD));                    // Byte 198
        std::string eventsLabel (readString(file, 36*WORD));                // Byte 199-234 ==> Event labels (4 char by label)
        int emptyBlock4         (readInt(file, 22*WORD));                   // Byte 235-256

        std::cout << "HEADER" << std::endl;
        std::cout << "nb3dPoints = " << nb3dPoints << std::endl;
        std::cout << "nbAnalogs = " << nbAnalogs << std::endl;
        std::cout << "firstFrame = " << firstFrame << std::endl;
        std::cout << "lastFrame = " << lastFrame << std::endl;
        std::cout << "nbMaxInterpGap = " << nbMaxInterpGap << std::endl;
        std::cout << "scaleFactor = " << scaleFactor << std::endl;
        std::cout << "dataStartAnalog = " << dataStartAnalog << std::endl;
        std::cout << "nbAnalogByFrame = " << nbAnalogByFrame << std::endl;
        std::cout << "frameRate = " << frameRate << std::endl;
        std::cout << "emptyBlock1 = " << emptyBlock1 << std::endl;
        std::cout << "keyLabelPresent = " << keyLabelPresent << std::endl;
        std::cout << "firstBlockKeyLabel = " << firstBlockKeyLabel << std::endl;
        std::cout << "fourCharPresent = " << fourCharPresent << std::endl;
        std::cout << "nbEvents = " << nbEvents << std::endl;
        std::cout << "emptyBlock2 = " << emptyBlock2 << std::endl;
        std::cout << "eventsTime = " << eventsTime << std::endl;
        std::cout << "eventsDisplay = " << eventsDisplay << std::endl;
        std::cout << "emptyBlock3 = " << emptyBlock3 << std::endl;
        std::cout << "eventsLabel = " << eventsLabel << std::endl;
        std::cout << "emptyBlock4 = " << emptyBlock4 << std::endl;
        std::cout << std::endl;


        // Read the Parameters Header
        int parametersStart     (readInt(file, 1*BYTE, 256*WORD*(parametersAddress-1), std::ios::beg));   // Byte 1 ==> if 1 then it starts at byte 3 otherwise at byte 512*parametersStart
        int iChecksum2          (readInt(file, 1*BYTE));            // Byte 2 ==> should be 80 if it is a c3d
        int nbParamBlock        (readInt(file, 1*BYTE));            // Byte 3 ==> Number of parameter blocks to follow
        int processorType       (readInt(file, 1*BYTE));            // Byte 4 ==> Processor type (83 + [1 Inter, 2 DEC, 3 MIPS])

        std::cout << "Parameters header" << std::endl;
        std::cout << "reservedParam1 = " << parametersStart << std::endl;
        std::cout << "reservedParam2 = " << iChecksum2 << std::endl;
        std::cout << "nbParamBlock = " << nbParamBlock << std::endl;
        std::cout << "processorType = " << processorType << std::endl;
        std::cout << std::endl;

        // Parameters group
        for (int i = 0; i < 20; ++i)
        {
            int nbCharInName       (readInt(file, 1*BYTE));                // Nb of char in the group name, locked if negative
            int id           (readInt(file, 1*BYTE));                       // Groupe ID always negative
            std::string name   (readString(file, nbCharInName*BYTE));      // Name of the group
            int offSetNext          (readInt(file, 2*BYTE));                // number of byte to the next group
            if (id < 0){
                std::cout << "Parameters group " << i << std::endl;
            } else {
                int lengthInByte        (readInt(file, 1*BYTE));    // -1 sizeof(char), 1 byte, 2 int, 4 float
                int nDimensions         (readInt(file, 1*BYTE));    // number of dimension of parameter (0 for scalar)
                std::vector<int> dimension;
                for (int j=0; j<nDimensions; ++j){
                    dimension.push_back (readInt(file, 1*BYTE));    // Read the dimension size
                }

                std::cout << "Parameter " << i << std::endl;
            }
            int nbCharInDesc   (readInt(file, 1*BYTE));                // Byte 5+nbCharInName ==> Number of characters in group description
            std::string desc   (readString(file, nbCharInDesc));  // Byte 6+nbCharInName ==> Group description

            std::cout << "nbCharInName = " << nbCharInName << std::endl;
            std::cout << "nbGroupID = " << id << std::endl;
            std::cout << "groupName = " << name << std::endl;
            std::cout << "offSetNextGroup = " << offSetNext << std::endl;
            std::cout << "nbCharInDesc = " << nbCharInDesc << std::endl;
            std::cout << "desc = " << desc << std::endl;

            std::cout << std::endl;
        }
        // Terminate
        file.close();

    }
    return 0;
}

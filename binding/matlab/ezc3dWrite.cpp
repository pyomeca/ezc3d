#include "mex.h"
#include <iostream>
#include <memory>

#include "ezc3d.h"
#include "utils.h"

void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
    // Check inputs and outputs
    if (nrhs != 2)
        mexErrMsgTxt("Input argument must be valids path and c3d structure.");
    if (mxIsChar(prhs[0]) != 1)
        mexErrMsgTxt("Input argument 1 must be a valid path to write.");
    if (mxIsStruct(prhs[1]) != 1)
        mexErrMsgTxt("Input argument 2 must be a valid c3d structure.");
    if (nlhs != 0)
        mexErrMsgTxt("Too many output arguments.");

    // Receive the pathmx
    mwSize pathlen = mxGetNumberOfElements(prhs[0]) + 1;
    char *path_tp = new char[pathlen];
    mxGetString(prhs[0], path_tp, pathlen);
    std::string path(path_tp);
    delete[] path_tp;
    if (path.size() == 0)
        mexErrMsgTxt("Input argument 1 must be a valid path to write.");
    std::string extension = ".c3d";
    if (path.substr(path.find_last_of(".")).compare(extension)){
        path += extension;
    }

    // Receive the structure
    const mxArray * c3dStruct(prhs[1]);

    // Get pointer on each struct (header, parameter, data)
    mxArray *header = mxGetField(c3dStruct, 0, "header");
    if (!header)
        mexErrMsgTxt("'header' is not accessible in the structure.");
    mxArray *parameter = mxGetField(c3dStruct, 0, "parameter");
    if (!parameter)
        mexErrMsgTxt("'parameter' is not accessible in the structure.");
    mxArray *data = mxGetField(c3dStruct, 0, "data");
    if (!data)
        mexErrMsgTxt("'data' is not accessible in the structure.");
    mxArray *dataPoints = mxGetField(data, 0, "points");
    mxArray *dataAnalogs = mxGetField(data, 0, "analogs");

    // Setup important factors
    if (!dataPoints)
        mexErrMsgTxt("'data.points' is not accessible in the structure.");
    if (mxGetNumberOfDimensions(dataPoints) != 3 )
        mexErrMsgTxt("'data.points' should be in format XYZ x nPoints x nFrames.");
    const mwSize *dimsPoints = mxGetDimensions(dataPoints);
    size_t nPointsComponents(dimsPoints[0]);
    size_t nPoints(dimsPoints[1]);
    size_t nFramesPoints(dimsPoints[2]);
    if (nPointsComponents < 3 || nPointsComponents > 4)
        mexErrMsgTxt("'data.points' should be in format XYZ x nPoints x nFrames.");

    if (!dataAnalogs)
        mexErrMsgTxt("'data.analogs' is not accessible in the structure.");
    if (mxGetNumberOfDimensions(dataAnalogs) != 2)
        mexErrMsgTxt("'data.analogs' should be in format nFrames x nAnalogs.");
    const mwSize *dimsAnalogs = mxGetDimensions(dataAnalogs);
    size_t nAnalogs(dimsAnalogs[1]);
    size_t nFramesAnalogs(dimsAnalogs[0]);

    size_t nSubframes(0);
    if (nFramesPoints != 0){
        if (nFramesAnalogs % nFramesPoints != 0)
            mexErrMsgTxt("Number of frames of Points and Analogs should be a multiple of an integer");
        nSubframes = nFramesAnalogs/nFramesPoints;
    }

    // Create a fresh c3d which will be fill with c3d struct
    ezc3d::c3d c3d;

    // Fill it with some values
    std::cout << "100 Hz hard-coded" << std::endl;
    ezc3d::ParametersNS::GroupNS::Parameter pointRate("RATE");
    pointRate.set(std::vector<float>() = {100}, {1});
    c3d.addParameter("POINT", pointRate);

    std::cout << "1000 Hz hard-coded" << std::endl;
    ezc3d::ParametersNS::GroupNS::Parameter analogRate("RATE");
    analogRate.set(std::vector<float>() = {500}, {1});
    c3d.addParameter("ANALOG", analogRate);
    std::cout << "No parameters are copied! To be implemented" << std::endl;


    // Get the name of the markers and the analogs
    mxArray *parameterPoints = mxGetField(parameter, 0, "POINT");
    if (!parameterPoints)
        mexErrMsgTxt("'parameter.POINT' is not accessible in the structure.");
    mxArray *parameterPointsLabels = mxGetField(parameterPoints, 0, "LABELS");
    if (!parameterPointsLabels)
        mexErrMsgTxt("'parameter.POINT.LABELS' parameters is not accessible in the structure.");
    if (nPoints != mxGetM(parameterPointsLabels) * mxGetN(parameterPointsLabels))
        mexErrMsgTxt("'parameter.POINT.LABELS' must have the same length as nPoints of the data.");
    std::vector<std::string> pointLabels;
    for (int i=0; i<nPoints; ++i){
        mxArray *pointLabelsPtr = mxGetCell(parameterPointsLabels, i);
        mwSize namelen = mxGetNumberOfElements(pointLabelsPtr) + 1;
        char *name = new char[namelen];
        mxGetString(pointLabelsPtr, name, namelen);
        pointLabels.push_back(name);
        delete[] name;
    }
    // Add them to the c3d
    for (int i=0; i<pointLabels.size(); ++i)
        c3d.addMarker(pointLabels[i]);

    mxArray *parameterAnalogs = mxGetField(parameter, 0, "ANALOG");
    if (!parameterAnalogs)
        mexErrMsgTxt("'parameter.ANALOG' is not accessible in the structure.");
    mxArray *parameterAnalogsLabels = mxGetField(parameterAnalogs, 0, "LABELS");
    if (!parameterAnalogsLabels)
        mexErrMsgTxt("'parameter.ANALOG.LABELS' parameters is not accessible in the structure.");
    if (nAnalogs != mxGetN(parameterAnalogsLabels))
        mexErrMsgTxt("'parameter.ANALOG.LABELS' must have the same length as nAnalogs of the data.");
    std::vector<std::string> analogsLabels;
    for (int i=0; i<nAnalogs; ++i){
        mxArray *analogsLabelsPtr = mxGetCell(parameterAnalogsLabels, i);
        mwSize namelen = mxGetNumberOfElements(analogsLabelsPtr) + 1;
        char *name = new char[namelen];
        mxGetString(analogsLabelsPtr, name, namelen);
        analogsLabels.push_back(name);
        delete[] name;
    }
    // Add them to the c3d
    for (int i=0; i<analogsLabels.size(); ++i)
        c3d.addAnalog(analogsLabels[i]);

    // Fill the data
    mxDouble* allDataPoints = mxGetDoubles(dataPoints);
    mxDouble* allDataAnalogs = mxGetDoubles(dataAnalogs);
    for (int f=0; f<nFramesPoints; ++f){
        ezc3d::DataNS::Frame frame;
        ezc3d::DataNS::Points3dNS::Points pts;
        for (int i=0; i<nPoints; ++i){
            ezc3d::DataNS::Points3dNS::Point pt;
            pt.name(pointLabels[i]);
            pt.x(allDataPoints[nPointsComponents*i+0+f*3*nPoints]);
            pt.y(allDataPoints[nPointsComponents*i+1+f*3*nPoints]);
            pt.z(allDataPoints[nPointsComponents*i+2+f*3*nPoints]);
            pts.add(pt);
        }

        ezc3d::DataNS::AnalogsNS::Analogs analog;
        for (int sf=0; sf<nSubframes; ++sf){
            ezc3d::DataNS::AnalogsNS::SubFrame subframes;
            for (int i=0; i<nAnalogs; ++i){
                ezc3d::DataNS::AnalogsNS::Channel c;
                c.value(allDataAnalogs[nFramesAnalogs*i+sf+f*nSubframes]);
                c.name(analogsLabels[i]);
                subframes.addChannel(c);
            }
            analog.addSubframe(subframes);
        }
        frame.add(pts, analog);
        c3d.addFrame(frame);// Add the previously created frame
    }
    c3d.write(path);
    return;
}

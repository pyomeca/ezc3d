#include "mex.h"
#include <iostream>
#include <memory>

#include "ezc3d.h"
#include "utils.h"

void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
    // Check inputs and outputs
    if (nrhs != 1)
        mexErrMsgTxt("Input argument must be a valid c3d structure.");
    if (mxIsStruct(prhs[0]) != 1)
        mexErrMsgTxt("Input argument must be a valid c3d structure.");
    if (nlhs != 0)
        mexErrMsgTxt("Too many output arguments.");

    // Receive the path
    const mxArray * c3dStruct(prhs[0]);

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

    if (nFramesAnalogs % nFramesPoints != 0)
        mexErrMsgTxt("Number of frames of Points and Analogs should be a multiple of an integer");
    size_t nSubframes(nFramesAnalogs/nFramesPoints);


    // Create a fresh c3d which will be fill with c3d struct
    ezc3d::c3d c3d;

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
    if (nAnalogs != mxGetM(parameterAnalogsLabels))
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


    // Fill the data
    mxDouble* allDataPoints = mxGetDoubles(dataPoints);
    mxDouble* allDataAnalogs = mxGetDoubles(dataAnalogs);
    for (int f=0; f<nFramesPoints; ++f){
        ezc3d::DataNS::Frame frame;
        ezc3d::DataNS::Points3dNS::Points pts_new;
        for (int i=0; i<nPoints; ++i){
            ezc3d::DataNS::Points3dNS::Point pt_new;
            pt_new.name(pointLabels[i]);
            pt_new.x(allDataPoints[nPointsComponents*i+0+f*3*nPoints]);
            pt_new.y(allDataPoints[nPointsComponents*i+1+f*3*nPoints]);
            pt_new.z(allDataPoints[nPointsComponents*i+2+f*3*nPoints]);
            pts_new.add(pt_new);
        }

        ezc3d::DataNS::AnalogsNS::Analogs analog;
        for (int sf=0; sf<nSubframes; ++sf){
            ezc3d::DataNS::AnalogsNS::SubFrame subframes;
            for (int i=0; i<2; ++i){
                ezc3d::DataNS::AnalogsNS::Channel c;
                c.value(allDataAnalogs[nFramesAnalogs*i+sf+f*nSubframes]);
                c.name(analogsLabels[i]);
                subframes.addChannel(c);
            }
            analog.addSubframe(subframes);
        }
        frame.add(pts_new, analog);
        c3d.addFrame(frame);// Add the previously created frame
    }

    return;
}

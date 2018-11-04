#include "mex.h"
#include <iostream>
#include <memory>

#include "ezc3d.h"
#include "utils.h"

void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
    // Check inputs and outputs
    if (nrhs > 1)
        mexErrMsgTxt("Input argument must be a file path string or no input for a valid empty structure.");
    if (nrhs == 1 && mxIsChar(prhs[0]) != 1)
        mexErrMsgTxt("Input argument must be a file path string or no input for a valid empty structure.");
    if (nlhs > 1)
        mexErrMsgTxt("Only one output is available");

    // Receive the path
    std::string path;
    if (nrhs == 1){
        char *buffer = mxArrayToString(prhs[0]);
        path = buffer;
        mxFree(buffer);
    }

    // Preparer the first layer of the output structure
    const char *globalFieldsNames[] = {"header", "parameters","data"};
    int headerIdx = 0;
    int parametersIdx = 1;
    int dataIdx = 2;
    mwSize globalFieldsDims[2] = {1, 1};
    plhs[0] = mxCreateStructArray(2, globalFieldsDims, sizeof(globalFieldsNames)/sizeof(*globalFieldsNames), globalFieldsNames);

    // Populate the c3d
    ezc3d::c3d *c3d;
    try{
        // Read the c3d
        if (path.size() == 0)
            c3d = new ezc3d::c3d;
        else
            c3d = new ezc3d::c3d(path);

        // Fill the header
        {
        const char *headerFieldsNames[] = {"points", "analogs", "events"};
        mwSize headerFieldsDims[2] = {1, 1};
        mxArray * headerStruct = mxCreateStructArray(2, headerFieldsDims, sizeof(headerFieldsNames)/sizeof(*headerFieldsNames), headerFieldsNames);
        mxSetFieldByNumber(plhs[0], 0, headerIdx, headerStruct);
            // fill points
            {
                const char *pointsFieldsNames[] = {"size", "frameRate", "firstFrame", "lastFrame"};
                mwSize pointFieldsDims[2] = {1, 1};
                mxArray * pointsStruct = mxCreateStructArray(2, pointFieldsDims, sizeof(pointsFieldsNames)/sizeof(*pointsFieldsNames), pointsFieldsNames);
                mxSetFieldByNumber(headerStruct, 0, 0, pointsStruct);

                fillMatlabField(pointsStruct, 0, c3d->header().nb3dPoints());
                fillMatlabField(pointsStruct, 1, static_cast<mxDouble>(c3d->header().frameRate()));
                fillMatlabField(pointsStruct, 2, c3d->header().firstFrame()+1);
                fillMatlabField(pointsStruct, 3, c3d->header().lastFrame()+1);
            }
            // fill analogs
            {
                const char *analogsFieldsNames[] = {"size", "frameRate", "firstFrame", "lastFrame"};
                mwSize analogsFieldsDims[2] = {1, 1};
                mxArray * analogsStruct = mxCreateStructArray(2, analogsFieldsDims, sizeof(analogsFieldsNames)/sizeof(*analogsFieldsNames), analogsFieldsNames);
                mxSetFieldByNumber(headerStruct, 0, 1, analogsStruct);

                fillMatlabField(analogsStruct, 0, c3d->header().nbAnalogs());
                fillMatlabField(analogsStruct, 1, static_cast<mxDouble>(c3d->header().nbAnalogByFrame() * c3d->header().frameRate()) );
                fillMatlabField(analogsStruct, 2, c3d->header().nbAnalogByFrame() * c3d->header().firstFrame()+1);
                fillMatlabField(analogsStruct, 3, c3d->header().nbAnalogByFrame() * (c3d->header().lastFrame()+1));
            }

            // fill events
            {
                const char *eventsFieldsNames[] = {"size", "eventsTime", "eventsLabel"};
                mwSize eventsFieldsDims[2] = {1, 1};
                mxArray * eventsStruct = mxCreateStructArray(2, eventsFieldsDims, sizeof(eventsFieldsNames)/sizeof(*eventsFieldsNames), eventsFieldsNames);
                mxSetFieldByNumber(headerStruct, 0, 2, eventsStruct);

                fillMatlabField(eventsStruct, 0, static_cast<int>(c3d->header().eventsTime().size()));
                fillMatlabField(eventsStruct, 1, c3d->header().eventsTime());
                fillMatlabField(eventsStruct, 2, c3d->header().eventsLabel());
            }
        }

        // Fill the parameters
        {
            size_t nbGroups(c3d->parameters().nbGroups());
            char **groupsFieldsNames = new char *[nbGroups];
            for (size_t g = 0; g < nbGroups; ++g){
                groupsFieldsNames[g] = new char[c3d->parameters().group(g).name().length() + 1];
                strcpy( groupsFieldsNames[g], c3d->parameters().group(g).name().c_str());
            }
            mwSize groupsFieldsDims[2] = {1, 1};
            mxArray * groupsStruct = mxCreateStructArray(2, groupsFieldsDims, static_cast<int>(nbGroups), const_cast<const char**>(groupsFieldsNames));
            mxSetFieldByNumber(plhs[0], 0, parametersIdx, groupsStruct);

            // Parse each parameters
            for (size_t g = 0; g < nbGroups; ++g){
                size_t nbParams(c3d->parameters().group(g).nbParameters());
                char **parametersFieldsNames = new char *[nbParams];
                for (size_t p = 0; p < nbParams; ++p){
                    parametersFieldsNames[p] = new char[c3d->parameters().group(g).parameter(p).name().length() + 1];
                    strcpy( parametersFieldsNames[p], c3d->parameters().group(g).parameter(p).name().c_str());
                }
                mwSize parametersFieldsDims[2] = {1, 1};
                mxArray * parametersStruct = mxCreateStructArray(2, parametersFieldsDims, static_cast<int>(nbParams), const_cast<const char**>(parametersFieldsNames));
                mxSetFieldByNumber(groupsStruct, 0, static_cast<int>(g), parametersStruct);

                // Fill each parameters
                for (size_t p = 0; p < nbParams; ++p){
                    ezc3d::ParametersNS::GroupNS::Parameter param = c3d->parameters().group(g).parameter(p);
                    if (param.type() == ezc3d::DATA_TYPE::INT)
                        fillMatlabField(parametersStruct, p, param.valuesAsInt(), param.dimension());
                    else if (param.type() == ezc3d::DATA_TYPE::FLOAT)
                        fillMatlabField(parametersStruct, p, param.valuesAsFloat(), param.dimension());
                    else if (param.type() == ezc3d::DATA_TYPE::CHAR)
                        fillMatlabField(parametersStruct, p, param.valuesAsString(), param.dimension());
                }
            }
        }


        // Fill the data
        {
        const char *dataFieldsNames[] = {"points", "analogs"};
        mwSize dataFieldsDims[2] = {1, 1};
        mxArray * dataStruct = mxCreateStructArray(2, dataFieldsDims, 2, dataFieldsNames);
        mxSetFieldByNumber(plhs[0], 0, dataIdx, dataStruct);

            // Fill the point data and analogous data
            {
            mwSize nFramesPoints(static_cast<mwSize>(c3d->header().nbFrames()));
            mwSize nPoints(static_cast<mwSize>(c3d->header().nb3dPoints()));
            mwSize dataPointsFieldsDims[3] = {3, nPoints, nFramesPoints};
            mxArray * dataPoints = mxCreateNumericArray(3, dataPointsFieldsDims, mxDOUBLE_CLASS, mxREAL);
            double * valPoints = mxGetPr(dataPoints);

            size_t nFramesAnalogs(static_cast<size_t>(nFramesPoints * c3d->header().nbAnalogByFrame()));
            size_t nAnalogs(c3d->header().nbAnalogs());
            size_t nSubFrames(c3d->header().nbAnalogByFrame());
            mxArray * dataAnalogs = mxCreateDoubleMatrix(nFramesAnalogs, nAnalogs, mxREAL);
            double * valAnalogs = mxGetPr(dataAnalogs);

            for (size_t f=0; f<nFramesPoints; ++f){
                ezc3d::DataNS::Frame frame(c3d->data().frame(f));
                // Points side
                for (size_t p = 0; p < frame.points().nbPoints(); ++p){
                    ezc3d::DataNS::Points3dNS::Point point(frame.points().point(p));
                    valPoints[f*nPoints*3+3*p+0] = static_cast<double>(point.x());
                    valPoints[f*nPoints*3+3*p+1] = static_cast<double>(point.y());
                    valPoints[f*nPoints*3+3*p+2] = static_cast<double>(point.z());
                }

                // Analogs side
                for (size_t sf=0; sf<frame.analogs().nbSubframes(); ++sf)
                    for (size_t c=0; c<frame.analogs().subframe(sf).nbChannels(); ++c)
                        valAnalogs[c*nSubFrames*nFramesPoints + sf + f*nSubFrames] = static_cast<double>(frame.analogs().subframe(sf).channel(c).data());
            }
            mxSetFieldByNumber(dataStruct, 0, 0, dataPoints);
            mxSetFieldByNumber(dataStruct, 0, 1, dataAnalogs);
            }
        }
    }
    catch (std::string m){
        delete c3d;
        mexErrMsgTxt(m.c_str());
    }

    delete c3d;
    return;
}

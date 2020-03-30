#include "mex.h"
#include <iostream>
#include <memory>

#include "ezc3d.h"
#include "utils.h"
#include "Header.h"
#include "Parameters.h"
#include "Data.h"

void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
    // Check inputs and outputs
    if (nrhs > 1)
        mexErrMsgTxt("Input argument must be a file path string or "
                     "no input for a valid empty structure.");
    if (nrhs == 1 && mxIsChar(prhs[0]) != 1)
        mexErrMsgTxt("Input argument must be a file path string or "
                     "no input for a valid empty structure.");
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
    plhs[0] = mxCreateStructArray(
                2, globalFieldsDims,
                sizeof(globalFieldsNames) / sizeof(*globalFieldsNames),
                globalFieldsNames);

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
        mxArray * headerStruct = mxCreateStructArray(
                    2, headerFieldsDims,
                    sizeof(headerFieldsNames) / sizeof(*headerFieldsNames),
                    headerFieldsNames);
        mxSetFieldByNumber(plhs[0], 0, headerIdx, headerStruct);
            // fill points
            {
                const char *pointsFieldsNames[] = {"size", "frameRate",
                                                   "firstFrame", "lastFrame"};
                mwSize pointFieldsDims[2] = {1, 1};
                mxArray * pointsStruct = mxCreateStructArray(
                            2, pointFieldsDims,
                            sizeof(pointsFieldsNames) / sizeof(*pointsFieldsNames),
                            pointsFieldsNames);
                mxSetFieldByNumber(headerStruct, 0, 0, pointsStruct);

                fillMatlabField(pointsStruct, 0, c3d->header().nb3dPoints());
                fillMatlabField(pointsStruct, 1,
                                static_cast<mxDouble>(
                                    c3d->header().frameRate()));
                fillMatlabField(pointsStruct, 2, c3d->header().firstFrame()+1);
                fillMatlabField(pointsStruct, 3, c3d->header().lastFrame()+1);
            }
            // fill analogs
            {
                const char *analogsFieldsNames[] = {"size", "frameRate",
                                                    "firstFrame", "lastFrame"};
                mwSize analogsFieldsDims[2] = {1, 1};
                mxArray * analogsStruct = mxCreateStructArray(
                            2, analogsFieldsDims,
                            sizeof(analogsFieldsNames) / sizeof(*analogsFieldsNames),
                            analogsFieldsNames);
                mxSetFieldByNumber(headerStruct, 0, 1, analogsStruct);

                fillMatlabField(analogsStruct, 0, c3d->header().nbAnalogs());
                fillMatlabField(analogsStruct, 1,
                                static_cast<mxDouble>(
                                    c3d->header().nbAnalogByFrame()
                                    * c3d->header().frameRate()) );
                fillMatlabField(analogsStruct, 2,
                                c3d->header().nbAnalogByFrame()
                                * c3d->header().firstFrame()+1);
                fillMatlabField(analogsStruct, 3,
                                c3d->header().nbAnalogByFrame()
                                * (c3d->header().lastFrame()+1));
            }

            // fill events
            {
                const char *eventsFieldsNames[] = {"size", "eventsTime",
                                                   "eventsLabel"};
                mwSize eventsFieldsDims[2] = {1, 1};
                mxArray * eventsStruct = mxCreateStructArray(
                            2, eventsFieldsDims,
                            sizeof(eventsFieldsNames) / sizeof(*eventsFieldsNames),
                            eventsFieldsNames);
                mxSetFieldByNumber(headerStruct, 0, 2, eventsStruct);

                fillMatlabField(eventsStruct, 0, static_cast<int>(
                                    c3d->header().eventsTime().size()));
                fillMatlabField(eventsStruct, 1, c3d->header().eventsTime());
                fillMatlabField(eventsStruct, 2, c3d->header().eventsLabel());
            }
        }

        // Fill the parameters
        {
            size_t nbGroups(c3d->parameters().nbGroups());
            char **groupsFieldsNames = new char *[nbGroups];
            for (size_t g = 0; g < nbGroups; ++g){
                groupsFieldsNames[g] = new char[c3d->parameters()
                        .group(g).name().length() + 1];
                strcpy( groupsFieldsNames[g], c3d->parameters().group(g).name().c_str());
            }
            mwSize groupsFieldsDims[2] = {1, 1};
            mxArray * groupsStruct =
                    mxCreateStructArray(
                        2, groupsFieldsDims, static_cast<int>(nbGroups),
                        const_cast<const char**>(groupsFieldsNames));
            mxSetFieldByNumber(plhs[0], 0, parametersIdx, groupsStruct);

            // Parse each group
            for (size_t g = 0; g < nbGroups; ++g){
                size_t nbParams(c3d->parameters().group(g).nbParameters());
                // Create the holder for parameter + 1 for metadata
                char **parametersFieldsNames = new char *[nbParams + 1];


                ezc3d::ParametersNS::GroupNS::Group group(
                            c3d->parameters().group(g));
                std::string metadataName(METADATA_FIELD);
                parametersFieldsNames[0] = new char[metadataName.length() + 1];
                strcpy( parametersFieldsNames[0], metadataName.c_str());

                for (size_t p = 0; p < nbParams; ++p){
                    parametersFieldsNames[p + 1] =
                            new char[c3d->parameters().group(g).parameter(p)
                            .name().length() + 1];
                    strcpy( parametersFieldsNames[p + 1],
                            c3d->parameters().group(g).parameter(p)
                            .name().c_str());
                }
                mwSize parametersFieldsDims[2] = {1, 1};
                mxArray * parametersStruct = mxCreateStructArray(
                            2, parametersFieldsDims,
                            static_cast<int>(nbParams + 1),
                            const_cast<const char**>(parametersFieldsNames));
                mxSetFieldByNumber(groupsStruct, 0,
                                   static_cast<int>(g), parametersStruct);

                // Fill Metadata
                fillMetadata(parametersStruct, 0,
                             group.description(), group.isLocked(), false);

                // Fill each parameters
                mwSize idxValue(2);
                for (size_t p = 0; p < nbParams; ++p){
                    ezc3d::ParametersNS::GroupNS::Parameter param =
                            c3d->parameters().group(g).parameter(p);
                    mxArray * valueArray =
                            fillMetadata(
                                parametersStruct, p + 1,
                                param.description(), param.isLocked(), true);
                    if (param.type() == ezc3d::DATA_TYPE::INT)
                        fillMatlabField(
                                    valueArray, idxValue,
                                    param.valuesAsInt(), param.dimension());
                    else if (param.type() == ezc3d::DATA_TYPE::FLOAT)
                        fillMatlabField(
                                    valueArray, idxValue,
                                    param.valuesAsDouble(), param.dimension());
                    else if (param.type() == ezc3d::DATA_TYPE::CHAR)
                        fillMatlabField(
                                    valueArray, idxValue,
                                    param.valuesAsString(), param.dimension());
                }
            }
        }


        // Fill the data
        {
        const char *dataFieldsNames[] = {"points", "meta_points", "analogs"};
        mwSize dataFieldsDims[3] = {1, 1, 1};
        mxArray * dataStruct =
                mxCreateStructArray(3, dataFieldsDims, 3, dataFieldsNames);
        mxSetFieldByNumber(plhs[0], 0, dataIdx, dataStruct);

            // Fill the point data and analogous data
            {
            mwSize nFramesPoints(static_cast<mwSize>(c3d->header().nbFrames()));
            mwSize nPoints(static_cast<mwSize>(c3d->header().nb3dPoints()));
            mwSize dataPointsFieldsDims[3] = {3, nPoints, nFramesPoints};
            mxArray * dataPoints = mxCreateNumericArray(
                        3, dataPointsFieldsDims, mxDOUBLE_CLASS, mxREAL);
            double * valPoints = mxGetPr(dataPoints);

            const char *dataMetaPointsFieldsNames[] = {"residuals", "camera_masks"};
            mwSize dataMetaPointsFieldsDims[2] = {1, 1};
            mxArray * dataMetaPointsStruct =
                    mxCreateStructArray(2, dataMetaPointsFieldsDims, 2, dataMetaPointsFieldsNames);
            mwSize dataMetaResidualsFieldsDims[3] = {1, nPoints, nFramesPoints};
            mxArray * dataMetaResiduals = mxCreateNumericArray(
                        3, dataMetaResidualsFieldsDims, mxDOUBLE_CLASS, mxREAL);
            double * valMetaResiduals = mxGetPr(dataMetaResiduals);
            mwSize dataMetaCameraMasksFieldsDims[3] = {7, nPoints, nFramesPoints};
            mxArray * dataMetaCameraMasks = mxCreateNumericArray(
                        3, dataMetaCameraMasksFieldsDims, mxDOUBLE_CLASS, mxREAL);
            double * valMetaCameraMasks = mxGetPr(dataMetaCameraMasks);
            mxSetFieldByNumber(dataMetaPointsStruct, 0, 0, dataMetaResiduals);
            mxSetFieldByNumber(dataMetaPointsStruct, 0, 1, dataMetaCameraMasks);

            size_t nFramesAnalogs(
                        static_cast<size_t>(
                            nFramesPoints * c3d->header().nbAnalogByFrame()));
            size_t nAnalogs(c3d->header().nbAnalogs());
            size_t nSubFrames(c3d->header().nbAnalogByFrame());
            mxArray * dataAnalogs = mxCreateDoubleMatrix(
                        nFramesAnalogs, nAnalogs, mxREAL);
            double * valAnalogs = mxGetPr(dataAnalogs);

            for (size_t f=0; f<nFramesPoints; ++f) {
                ezc3d::DataNS::Frame frame(c3d->data().frame(f));

                // Points side
                for (size_t p = 0; p < frame.points().nbPoints(); ++p){
                    const ezc3d::DataNS::Points3dNS::Point& point(
                                frame.points().point(p));
                    if (point.residual() < 0){
                        valPoints[f*nPoints*3+3*p+0] =
                                static_cast<double>(NAN);
                        valPoints[f*nPoints*3+3*p+1] =
                                static_cast<double>(NAN);
                        valPoints[f*nPoints*3+3*p+2] =
                                static_cast<double>(NAN);
                    }
                    else {
                        valPoints[f*nPoints*3+3*p+0] =
                                static_cast<double>(point.x());
                        valPoints[f*nPoints*3+3*p+1] =
                                static_cast<double>(point.y());
                        valPoints[f*nPoints*3+3*p+2] =
                                static_cast<double>(point.z());
                    }

                    // Metadata for points
                    valMetaResiduals[f*nPoints+p] = static_cast<double>(point.residual());
                    std::vector<bool> cameraMasks(point.cameraMask());
                    for (size_t cam = 0; cam < 7; ++cam){
                        valMetaCameraMasks[f*nPoints*7+7*p+cam] = cameraMasks[cam];
                    }
                }


                // Analogs side
                for (size_t sf=0; sf<frame.analogs().nbSubframes(); ++sf)
                    for (size_t c=0; c<frame.analogs().subframe(sf).nbChannels()
                         ; ++c)
                        valAnalogs[c*nSubFrames*nFramesPoints + sf + f*nSubFrames] =
                                static_cast<double>(
                                    frame.analogs().subframe(sf)
                                    .channel(c).data());
            }
            mxSetFieldByNumber(dataStruct, 0, 0, dataPoints);
            mxSetFieldByNumber(dataStruct, 0, 1, dataMetaPointsStruct);
            mxSetFieldByNumber(dataStruct, 0, 2, dataAnalogs);
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

#include <iostream>
#include <memory>

#include <ezc3d/ezc3d.h>
#include "utils.h"
#include <ezc3d/Header.h>
#include <ezc3d/Parameters.h>
#include <ezc3d/Data.h>
#include <ezc3d/Frame.h>
#include "RotationsInfo.h"
#include <ezc3d/modules/ForcePlatforms.h>
#include <string.h>
#include <cmath>

void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
    // Check inputs and outputs
    if (nrhs > 2)
        mexErrMsgTxt("Input argument must be a file path string (with a boolean "
                     "for accepting bad formatting files) or no input for a "
                     "valid empty structure.");
    if (nrhs >= 1 && !mxIsChar(prhs[0]))
        mexErrMsgTxt("First input argument must be a file path string or "
                     "no input for a valid empty structure.");
    if (nrhs >= 2 && !mxIsLogical(prhs[1]))
        mexErrMsgTxt("Second input argument must be a bool for allowing to load "
                     "bad formatted c3d. Warning this can generate a segmentation "
                     "fault.");

    if (nlhs > 2)
        mexErrMsgTxt("Only two outputs are available");

    // Receive the path
    std::string path;
    if (nrhs >= 1){
        char *buffer = mxArrayToString(prhs[0]);
        path = buffer;
        mxFree(buffer);
    }
    bool ignoreBadFormatting = false;
    if (nrhs >= 2){
        ignoreBadFormatting = toBool(prhs[1]);
    }

    // Preparer the first layer of the output structure
    const char *globalFieldsNames[] = {"header", "parameters", "data"};
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
        if (path.size() == 0){
            if (nlhs > 1){
                mexErrMsgTxt("Force platform filter is not available when "
                             "creating an empty structure.");
            }
            c3d = new ezc3d::c3d;
        }
        else {
            c3d = new ezc3d::c3d(path, ignoreBadFormatting);
        }

        // Fill the header
        {
        const char *headerFieldsNames[] = {"points", "analogs", "rotations", "events"};
        mwSize headerFieldsDims[2] = {1, 1};
        mxArray * headerStruct = mxCreateStructArray(
                    2, headerFieldsDims,
                    sizeof(headerFieldsNames) / sizeof(*headerFieldsNames),
                    headerFieldsNames);
        mxSetFieldByNumber(plhs[0], 0, headerIdx, headerStruct);
            // fill points
            {
                const char *pointsFieldsNames[] = {
                    "size", "frameRate", "firstFrame", "lastFrame"};
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
                const char *analogsFieldsNames[] = {
                    "size", "frameRate", "firstFrame", "lastFrame"};
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
            // fill rotations
            {
                const char *rotationsFieldsNames[] = {
                    "size", "frameRate", "firstFrame", "lastFrame"};
                mwSize rotationsFieldsDims[2] = {1, 1};
                mxArray * rotationsStruct = mxCreateStructArray(
                            2, rotationsFieldsDims,
                            sizeof(rotationsFieldsNames) / sizeof(*rotationsFieldsNames),
                            rotationsFieldsNames);
                mxSetFieldByNumber(headerStruct, 0, 2, rotationsStruct);

                ezc3d::DataNS::RotationNS::Info rotationsInfo(*c3d);
                fillMatlabField(rotationsStruct, 0, rotationsInfo.used());
                fillMatlabField(rotationsStruct, 1,
                    static_cast<mxDouble>(rotationsInfo.ratio() * c3d->header().frameRate()));
                fillMatlabField(rotationsStruct, 2,
                    rotationsInfo.ratio() * c3d->header().firstFrame()+1);
                fillMatlabField(rotationsStruct, 3,
                    rotationsInfo.ratio() * (c3d->header().lastFrame()+1));
            }

            // fill events
            {
                const char *eventsFieldsNames[] = {
                    "size", "eventsTime", "eventsLabel"};
                mwSize eventsFieldsDims[2] = {1, 1};
                mxArray * eventsStruct = mxCreateStructArray(
                            2, eventsFieldsDims,
                            sizeof(eventsFieldsNames) / sizeof(*eventsFieldsNames),
                            eventsFieldsNames);
                mxSetFieldByNumber(headerStruct, 0, 3, eventsStruct);

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
        const char *dataFieldsNames[] = {"points", "meta_points", "analogs", "rotations"};
        mwSize dataFieldsDims[4] = {1, 1, 1, 1};
        mxArray * dataStruct =
                mxCreateStructArray(4, dataFieldsDims, 4, dataFieldsNames);
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
            size_t nSubFramesAnalogs (c3d->header().nbAnalogByFrame());
            mxArray * dataAnalogs = mxCreateDoubleMatrix(
                        nFramesAnalogs, nAnalogs, mxREAL);
            double * valAnalogs = mxGetPr(dataAnalogs);

            ezc3d::DataNS::RotationNS::Info rotationsInfo(*c3d);
            mwSize nRotations(static_cast<mwSize>(rotationsInfo.used()));
            mwSize nFramesRotations(static_cast<mwSize>(c3d->header().nbFrames() * rotationsInfo.ratio()));
            size_t nSubFramesRotations(static_cast<mwSize>(rotationsInfo.ratio()));
            mwSize nDataRotations[4] = {4, 4, nRotations, nFramesRotations};
            mxArray* dataRotations = mxCreateNumericArray(4, nDataRotations, mxDOUBLE_CLASS, mxREAL);
            double* valRotations = mxGetPr(dataRotations);

            for (size_t f = 0; f < nFramesPoints; ++f) {
                ezc3d::DataNS::Frame frame(c3d->data().frame(f));

                // Points side
                for (size_t p = 0; p < frame.points().nbPoints(); ++p) {
                    const ezc3d::DataNS::Points3dNS::Point& point(
                        frame.points().point(p));
                    if (point.residual() < 0) {
                        valPoints[f * nPoints * 3 + 3 * p + 0] =
                            static_cast<double>(NAN);
                        valPoints[f * nPoints * 3 + 3 * p + 1] =
                            static_cast<double>(NAN);
                        valPoints[f * nPoints * 3 + 3 * p + 2] =
                            static_cast<double>(NAN);
                    }
                    else {
                        valPoints[f * nPoints * 3 + 3 * p + 0] =
                            static_cast<double>(point.x());
                        valPoints[f * nPoints * 3 + 3 * p + 1] =
                            static_cast<double>(point.y());
                        valPoints[f * nPoints * 3 + 3 * p + 2] =
                            static_cast<double>(point.z());
                    }

                    // Metadata for points
                    valMetaResiduals[f * nPoints + p] = static_cast<double>(point.residual());
                    std::vector<bool> cameraMasks(point.cameraMask());
                    for (size_t cam = 0; cam < 7; ++cam) {
                        valMetaCameraMasks[f * nPoints * 7 + 7 * p + cam] = cameraMasks[cam];
                    }
                }


                // Analogs side
                for (size_t sf = 0; sf < frame.analogs().nbSubframes(); ++sf)
                    for (size_t c = 0; c < frame.analogs().subframe(sf).nbChannels(); ++c)
                        valAnalogs[c * nSubFramesAnalogs * nFramesPoints + sf + f * nSubFramesAnalogs] =
                        static_cast<double>(
                            frame.analogs().subframe(sf)
                            .channel(c).data());


                // Rotations side
                for (size_t sf = 0; sf < frame.rotations().nbSubframes(); ++sf)
                    for (size_t r = 0; r < frame.rotations().subframe(sf).nbRotations(); ++r)
                    {
                        const ezc3d::DataNS::RotationNS::Rotation& current(frame.rotations().subframe(sf).rotation(r));
                        for (size_t i = 0; i < 4; ++i)
                            for (size_t j = 0; j < 4; ++j) 
                                valRotations[
                                    f * nSubFramesRotations * 16 * nRotations +
                                    sf * 16 * nRotations +
                                    r * 16 + 
                                    i * 4 +
                                    j
                                ] = current(j, i);
                    }
            }
            mxSetFieldByNumber(dataStruct, 0, 0, dataPoints);
            mxSetFieldByNumber(dataStruct, 0, 1, dataMetaPointsStruct);
            mxSetFieldByNumber(dataStruct, 0, 2, dataAnalogs);
            mxSetFieldByNumber(dataStruct, 0, 3, dataRotations);
            }
        }

        // Fill force platform if needed
        if (nlhs > 1){
            const auto& all_pf = ezc3d::Modules::ForcePlatforms(*c3d);

            mwSize globalPlatFormDims[2] = {static_cast<mwSize>(all_pf.forcePlatforms().size()), 1};
            const char *forcePlatformNames[] =
                {"unit_force", "unit_moment", "unit_position",
                "cal_matrix", "corners", "origin", "force", "moment", "center_of_pressure", "Tz"};
            plhs[1] = mxCreateStructArray(
                        2, globalPlatFormDims,
                        sizeof(forcePlatformNames) / sizeof(*forcePlatformNames),
                        forcePlatformNames);

            for (size_t i=0; i<all_pf.forcePlatforms().size(); ++i){
                auto& pf(all_pf.forcePlatform(i));

                // Units
                mxSetFieldByNumber(plhs[1], i, 0, mxCreateString(pf.forceUnit().c_str()));
                mxSetFieldByNumber(plhs[1], i, 1, mxCreateString(pf.momentUnit().c_str()));
                mxSetFieldByNumber(plhs[1], i, 2, mxCreateString(pf.positionUnit().c_str()));

                // Force platform configuration
                mwSize pfCalMatrixSize[2] = {static_cast<mwSize>(pf.calMatrix().nbRows()), static_cast<mwSize>(pf.calMatrix().nbCols())};
                mxArray * pfCalMatrix = mxCreateNumericArray(2, pfCalMatrixSize, mxDOUBLE_CLASS, mxREAL);
                double * valCalMatrix = mxGetPr(pfCalMatrix);
                size_t cmp = 0;
                for (size_t col=0; col<pf.calMatrix().nbCols(); ++col){
                    for (size_t row=0; row<pf.calMatrix().nbRows(); ++row){
                        valCalMatrix[cmp++] = pf.calMatrix()(row, col);
                    }
                }
                mxSetFieldByNumber(plhs[1], i, 3, pfCalMatrix);

                mwSize pfCornersSize[2] = {3, static_cast<mwSize>(pf.corners().size())};
                mxArray * pfCorners = mxCreateNumericArray(2, pfCornersSize, mxDOUBLE_CLASS, mxREAL);
                double * valCorners = mxGetPr(pfCorners);
                ezc3d::Matrix corners(pf.corners());
                cmp = 0;
                for (size_t col=0; col<pf.corners().size(); ++col){
                    for (size_t row=0; row<3; ++row){
                        valCorners[cmp++] = corners(row, col);
                    }
                }
                mxSetFieldByNumber(plhs[1], i, 4, pfCorners);

                mwSize pfOriginSize[2] = {3, 1};
                mxArray * pfOrigin = mxCreateNumericArray(2, pfOriginSize, mxDOUBLE_CLASS, mxREAL);
                double * valOrigin = mxGetPr(pfOrigin);
                for (size_t row=0; row<3; ++row){
                    valOrigin[row] = pf.origin()(row);
                }
                mxSetFieldByNumber(plhs[1], i, 5, pfOrigin);

                // Data
                size_t nFrames(c3d->header().nbFrames() * c3d->header().nbAnalogByFrame());
                mwSize pfDataSize[2] = {3, static_cast<mwSize>(nFrames)};
                mxArray * pfForceMatrix = mxCreateNumericArray(2, pfDataSize, mxDOUBLE_CLASS, mxREAL);
                mxArray * pfMomentMatrix = mxCreateNumericArray(2, pfDataSize, mxDOUBLE_CLASS, mxREAL);
                mxArray * pfCoPMatrix = mxCreateNumericArray(2, pfDataSize, mxDOUBLE_CLASS, mxREAL);
                mxArray * pfTzMatrix = mxCreateNumericArray(2, pfDataSize, mxDOUBLE_CLASS, mxREAL);
                double * valForceMatrix = mxGetPr(pfForceMatrix);
                double * valMomentMatrix = mxGetPr(pfMomentMatrix);
                double * valCoPMatrix = mxGetPr(pfCoPMatrix);
                double * valTzMatrix = mxGetPr(pfTzMatrix);
                cmp = 0;
                for (size_t col=0; col<nFrames; ++col){
                    for (size_t row=0; row<3; ++row){
                        valForceMatrix[cmp] = pf.forces()[col](row);
                        valMomentMatrix[cmp] = pf.moments()[col](row);
                        valCoPMatrix[cmp] = pf.CoP()[col](row);
                        valTzMatrix[cmp++] = pf.Tz()[col](row);
                    }
                }
                mxSetFieldByNumber(plhs[1], i, 6, pfForceMatrix);
                mxSetFieldByNumber(plhs[1], i, 7, pfMomentMatrix);
                mxSetFieldByNumber(plhs[1], i, 8, pfCoPMatrix);
                mxSetFieldByNumber(plhs[1], i, 9, pfTzMatrix);
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

#include "mex.h"
#include <iostream>
#include <memory>

#include "ezc3d.h"
#include "utils.h"
#include "Header.h"
#include "Parameters.h"
#include "Data.h"

size_t parseParam(
        mxDouble* data,
        const std::vector<size_t> &dimension,
        std::vector<int> &param_data,
        size_t idxInData=0,
        size_t currentIdx=0) {
    if (dimension[currentIdx] == 0)
        return SIZE_MAX;

    for (size_t i=0; i<dimension[currentIdx]; ++i){
        if (currentIdx == dimension.size()-1){
            param_data.push_back (static_cast<int>(data[idxInData]));
            ++idxInData;
        }
        else
            idxInData = parseParam(data, dimension, param_data, idxInData, currentIdx + 1);
    }
    return idxInData;
}

size_t parseParam(
        mxDouble* data,
        const std::vector<size_t> &dimension,
        std::vector<float> &param_data,
        size_t idxInData=0,
        size_t currentIdx=0) {
    if (dimension[currentIdx] == 0)
        return SIZE_MAX;

    for (size_t i = 0; i<dimension[currentIdx]; ++i){
        if (currentIdx == dimension.size()-1){
            param_data.push_back (static_cast<float>(data[idxInData]));
            ++idxInData;
        }
        else
            idxInData = parseParam(data, dimension, param_data, idxInData, currentIdx + 1);
    }
    return idxInData;
}
size_t parseParam(
        mxArray* data,
        const std::vector<size_t> &dimension,
        std::vector<std::string> &param_data,
        size_t idxInData=0,
        size_t currentIdx=0) {
    if (dimension[currentIdx] == 0)
        return SIZE_MAX;

    for (size_t i = 0; i < dimension[currentIdx]; ++i){
        if (currentIdx == dimension.size()-1){
            mxArray *cell(mxGetCell(data, static_cast<mwIndex>(idxInData)));
            param_data.push_back (toString(cell));
            ++idxInData;
        }
        else
            idxInData = parseParam(data, dimension, param_data, idxInData, currentIdx + 1);
    }
    return idxInData;
}

size_t checkLongestStrParam(
        const std::vector<std::string> &param_data) {
    size_t longest(0);
    for (size_t i=0; i<param_data.size(); ++i){
        if (param_data[i].size() > longest)
            longest = param_data[i].size();
    }
    return longest;
}


void mexFunction(
        int nlhs,
        mxArray *[],
        int nrhs,
        const mxArray *prhs[]) {
    // Check inputs and outputs
    if (nrhs != 2)
        mexErrMsgTxt("Input argument must be valids path and c3d structure.");
    if (mxIsChar(prhs[0]) != 1)
        mexErrMsgTxt("Input argument 1 must be a valid path to write.");
    if (mxIsStruct(prhs[1]) != 1)
        mexErrMsgTxt("Input argument 2 must be a valid c3d structure.");
    if (nlhs != 0)
        mexErrMsgTxt("Too many output arguments.");

    // Receive the path
    std::string path(toString(prhs[0]));
    if (path.size() == 0)
        mexErrMsgTxt("Input argument 1 must be a valid path to write.");
    std::string extension = ".c3d";
    if (path.find_last_of(".") > path.size()
            || path.substr(path.find_last_of(".")).compare(extension)){
        path += extension;
    }

    // Receive the structure
    const mxArray * c3dStruct(prhs[1]);
    mxArray *header = mxGetField(c3dStruct, 0, "header");
    if (!header)
        mexErrMsgTxt("'header' is not accessible in the structure.");
    mxArray *parameters = mxGetField(c3dStruct, 0, "parameters");
    if (!parameters)
        mexErrMsgTxt("'parameters' is not accessible in the structure.");
    mxArray *data = mxGetField(c3dStruct, 0, "data");
    if (!data)
        mexErrMsgTxt("'data' is not accessible in the structure.");
    mxArray *dataPoints = mxGetField(data, 0, "points");
    mxArray *dataMetaPoints = mxGetField(data, 0, "meta_points");
    mxArray *dataAnalogs = mxGetField(data, 0, "analogs");

    // Setup important factors
    if (!dataPoints)
        mexErrMsgTxt("'data.points' is not accessible in the structure.");
    const mwSize *dimsPoints = mxGetDimensions(dataPoints);
    size_t nFramesPoints;
    if (mxGetNumberOfDimensions(dataPoints) == 3)
        nFramesPoints = dimsPoints[2];
    else if (mxGetNumberOfDimensions(dataPoints) == 2)
        nFramesPoints = 1;
    else {
        nFramesPoints = INT_MAX;
        mexErrMsgTxt("'data.points' should be in "
                     "format XYZ x nPoints x nFrames.");
    }
    size_t nPointsComponents(dimsPoints[0]);
    size_t nPoints(dimsPoints[1]);
    if (nPointsComponents < 3 || nPointsComponents > 4)
        mexErrMsgTxt("'data.points' should be in "
                     "format XYZ x nPoints x nFrames.");

    // Check if metadat exists and if so their dimensions
    mxArray *metadataResidual = nullptr;
    mxArray *metadataCamMasks = nullptr;
    if (dataMetaPoints) {
        metadataResidual = mxGetField(dataMetaPoints, 0, "residuals");
        if (metadataResidual){
            if (mxGetNumberOfDimensions(metadataResidual) == 3) {
                const mwSize *dimsResiduals = mxGetDimensions(metadataResidual);
                if (dimsResiduals[0] * dimsResiduals[1] * dimsResiduals[2] == 0) {
                    // Act as if there is no residual
                    metadataResidual = nullptr;
                }
                else if(!(dimsResiduals[0] == 1 && dimsResiduals[1] == nPoints
                          && dimsResiduals[2] == nFramesPoints) ){
                    mexErrMsgTxt("'data.meta_points.residuals' should be in "
                                 "format 1 x nPoints x nFrames.");
                }
            }
            else {
                mexErrMsgTxt("'data.meta_points.residuals' should be in "
                             "format 1 x nPoints x nFrames.");
            }
        }

        metadataCamMasks = mxGetField(dataMetaPoints, 0, "camera_masks");
        if (metadataCamMasks){
            if (mxGetNumberOfDimensions(metadataCamMasks) == 3) {
                const mwSize *dimsResiduals = mxGetDimensions(metadataCamMasks);
                if (dimsResiduals[0] * dimsResiduals[1] * dimsResiduals[2] == 0) {
                    // Act as if there is no masks
                    metadataCamMasks = nullptr;
                }
                else if(!(dimsResiduals[0] == 7 && dimsResiduals[1] == nPoints
                          && dimsResiduals[2] == nFramesPoints) ){
                    mexErrMsgTxt("'data.meta_points.camera_masks' should be in "
                                 "format 7 x nPoints x nFrames.");
                }
            }
            else {
                mexErrMsgTxt("'data.meta_points.camera_masks' should be in "
                             "format 7 x nPoints x nFrames.");
            }
        }
    }

    if (!dataAnalogs)
        mexErrMsgTxt("'data.analogs' is not accessible in the structure.");
    if (mxGetNumberOfDimensions(dataAnalogs) != 2)
        mexErrMsgTxt("'data.analogs' should be in format nFrames x nAnalogs.");
    const mwSize *dimsAnalogs = mxGetDimensions(dataAnalogs);
    size_t nAnalogs(dimsAnalogs[1]);
    size_t nFramesAnalogs(dimsAnalogs[0]);

    size_t nFrames(0);
    size_t nSubframes(0);
    if (nFramesPoints != 0){
        if (nFramesAnalogs % nFramesPoints != 0)
            mexErrMsgTxt("Number of frames of Points and Analogs "
                         "should be a multiple of an integer");
        nFrames = nFramesPoints;
        nSubframes = nFramesAnalogs/nFramesPoints;
    } else {
        nFrames = nFramesAnalogs;
        nSubframes = 1;
    }



    // Create a fresh c3d which will be fill with c3d struct
    ezc3d::c3d c3d;

    // Fill the header field that won't autoadjusts
    size_t firstFrame(0);
    mxArray *headerPoint = mxGetField(header, 0, "points");
    if (headerPoint){
        mxArray *firstFrameField = mxGetField(headerPoint, 0, "firstFrame");
        if (firstFrameField){
            mxDouble* firstFrameData = mxGetDoubles(firstFrameField);
            if (firstFrameData)
                // 1-based
                firstFrame = static_cast<size_t>(firstFrameData[0] - 1);
        }
    }
    c3d.setFirstFrame(firstFrame);

    // Get the names of the points
    mxArray *parametersPoints = mxGetField(parameters, 0, "POINT");
    if (!parametersPoints) {
        mexErrMsgTxt("'parameters.POINT' is not accessible in the structure.");
    }
    mxArray *parametersPointsLabels = mxGetField(parametersPoints, 0, "LABELS");
    if (!parametersPointsLabels) {
        mexErrMsgTxt("'parameters.POINT.LABELS' parameters "
                     "is not accessible in the structure.");
    }
    mxArray *valuePointsLabels = mxGetField(
                parametersPointsLabels, 0, DATA_FIELD);
    if (!valuePointsLabels) {
        mexErrMsgTxt(("'parameters.POINT.LABELS." + std::string(DATA_FIELD)
                     + " parameters is not accessible in the structure.")
                     .c_str());
    }
    if (nPoints != mxGetM(valuePointsLabels) * mxGetN(valuePointsLabels)) {
        mexErrMsgTxt("'parameters.POINT.LABELS' must have "
                     "the same length as nPoints of the data.");
    }

    std::vector<std::string> pointLabels;
    for (size_t i=0; i<nPoints; ++i) {
        mxArray *pointLabelsPtr = mxGetCell(valuePointsLabels, i);
        pointLabels.push_back(toString(pointLabelsPtr));
    }
    // Add them to the c3d
    for (size_t i=0; i<pointLabels.size(); ++i)
        c3d.point(pointLabels[i]);

    // Get the names of the analogs
    mxArray *groupAnalogs = mxGetField(parameters, 0, "ANALOG");
    if (!groupAnalogs)
        mexErrMsgTxt("'parameters.ANALOG' is not accessible in the structure.");
    mxArray *groupAnalogsLabels = mxGetField(groupAnalogs, 0, "LABELS");
    if (!groupAnalogsLabels)
        mexErrMsgTxt("'parameters.ANALOG.LABELS' parameters "
                     "is not accessible in the structure.");
    mxArray *valueAnalogsLabels = mxGetField(groupAnalogsLabels, 0, DATA_FIELD);
    if (!valueAnalogsLabels) {
        mexErrMsgTxt(("'parameters.ANALOG.LABELS." + std::string(DATA_FIELD)
                     + " parameters is not accessible in the structure.")
                     .c_str());
    }
    if (nAnalogs != mxGetM(valueAnalogsLabels) * mxGetN(valueAnalogsLabels))
        mexErrMsgTxt("'parameters.ANALOG.LABELS' must have "
                     "the same length as nAnalogs of the data.");
    std::vector<std::string> analogsLabels;
    for (size_t i=0; i<nAnalogs; ++i){
        mxArray *analogsLabelsPtr = mxGetCell(valueAnalogsLabels, i);
        analogsLabels.push_back(toString(analogsLabelsPtr));
    }
    // Add them to the c3d
    for (size_t i=0; i<analogsLabels.size(); ++i) {
        c3d.analog(analogsLabels[i]);
    }

    //  Fill the parameters
    for (int g=0; g<mxGetNumberOfFields(parameters); ++g){ // top level
        std::string groupName(mxGetFieldNameByNumber(parameters, g));
        mxArray* groupField(mxGetFieldByNumber(parameters, 0, g));
        if (!groupField) {
            mexErrMsgTxt("Unexplained error while gathering the parameters. "
                         "Please report this");
        }

        mxArray* metadataField(mxGetField(groupField, 0, METADATA_FIELD));
        if (metadataField) {
            std::string description;
            bool isLocked(false);
            mxArray* descriptionField(
                        mxGetField(metadataField, 0, DESCRIPTION_FIELD));
            if (descriptionField) {
                description = toString(descriptionField);
            }

            mxArray* isLockedField(
                        mxGetField(metadataField, 0, IS_LOCKED_FIELD));
            if (isLockedField) {
                isLocked = toBool(isLockedField);
            }
            c3d.setGroupMetadata(groupName, description, isLocked);
        }

        for (int p=0; p<mxGetNumberOfFields(groupField); ++p){
            std::string paramName(mxGetFieldNameByNumber(groupField, p));
            if (!paramName.compare(METADATA_FIELD)) {
                continue;
            }

            mxArray* paramField(mxGetFieldByNumber(groupField, 0, p));
            // Copy the parameters into the c3d,
            // but skip those who are already done
            if ( !(!groupName.compare("POINT") && !paramName.compare("USED"))
                 && !(!groupName.compare("POINT") && !paramName.compare("FRAMES"))
                 && !(!groupName.compare("POINT") && !paramName.compare("LABELS"))
                 && !(!groupName.compare("ANALOG") && !paramName.compare("USED"))
                 && !(!groupName.compare("ANALOG") && !paramName.compare("LABELS"))
                 && !(!groupName.compare("ANALOG") && !paramName.compare("SCALE"))
                 && !(!groupName.compare("ANALOG") && !paramName.compare("OFFSET"))
                 && !(!groupName.compare("ANALOG") && !paramName.compare("UNITS"))) {
                std::vector<size_t> dimension;
                size_t nDim;
                mxArray* valueField(mxGetField(paramField, 0, DATA_FIELD));

                if (!valueField)
                    nDim = 0;
                else
                    nDim =mxGetNumberOfDimensions(valueField);

                if (nDim == 0)
                    dimension.push_back(0);
                else if (nDim == 2 && mxGetDimensions(valueField)[0]
                         * mxGetDimensions(valueField)[1] == 0)
                    dimension.push_back(0);
                else if (nDim == 2 && mxGetDimensions(valueField)[0]
                         * mxGetDimensions(valueField)[1] == 1)
                    dimension.push_back(1);
                else
                    for (size_t i = 0; i < nDim; ++i)
                        dimension.push_back(mxGetDimensions(valueField)[i]);

                // Special cases
                if ( (!groupName.compare("POINT")
                      && !paramName.compare("DESCRIPTIONS"))
                     && dimension[0] != nPoints)
                    continue;
                if ( (!groupName.compare("ANALOG")
                      && !paramName.compare("DESCRIPTIONS"))
                     && dimension[0] != nAnalogs)
                    continue;

                ezc3d::ParametersNS::GroupNS::Parameter newParam(paramName);
                try {
                    ezc3d::DATA_TYPE type(
                                c3d.parameters().group(groupName)
                                .parameter(paramName).type());

                    if  (type == ezc3d::DATA_TYPE::INT
                         || type == ezc3d::DATA_TYPE::BYTE) {
                        std::vector<int> data;
                        parseParam(mxGetDoubles(valueField), dimension, data);
                        newParam.set(data, dimension);
                    } else if (type == ezc3d::DATA_TYPE::FLOAT) {
                        std::vector<float> data;
                        parseParam(mxGetDoubles(valueField), dimension, data);
                        newParam.set(
                            std::vector<double>(data.begin(), data.end()),
                            dimension);
                    } else if (type == ezc3d::DATA_TYPE::CHAR) {
                        std::vector<std::string> data;
                        parseParam(valueField, dimension, data);
                        dimension.pop_back();
                        newParam.set(data, dimension);
                    } else
                        mexErrMsgTxt(std::string(
                                         "Unrecognized type for parameter."
                                         + groupName + "." + paramName + ".")
                                     .c_str());
                    } catch (std::invalid_argument) {
                        if (!valueField || mxIsDouble(valueField)) {
                            std::vector<float> data;
                            parseParam(mxGetDoubles(valueField), dimension, data);
                            newParam.set(
                                std::vector<double>(data.begin(), data.end()),
                                dimension);
                        } else if (mxIsCell(valueField)) {
                            std::vector<std::string> data;
                            parseParam(valueField, dimension, data);
                            dimension.pop_back();
                            newParam.set(data, dimension);
                        } else if (mxIsChar(valueField)) {
                            std::vector<std::string> data;
                            data.push_back (toString(valueField));
                            dimension.pop_back();  // Matlab inserts length already
                            newParam.set(data, dimension);
                        } else
                            mexErrMsgTxt(std::string(
                                             "Unrecognized type for parameter."
                                             + groupName + "." + paramName + ".")
                                         .c_str());
                    }

                // Get the metadata for this parameter
                mxArray* descriptionField(
                            mxGetField(paramField, 0, DESCRIPTION_FIELD));
                if (descriptionField) {
                    newParam.description(toString(descriptionField));
                }

                mxArray* isLockedField(
                            mxGetField(paramField, 0, IS_LOCKED_FIELD));
                if (isLockedField) {
                    if (toBool(isLockedField)) {
                        newParam.lock();
                    }
                    else {
                        newParam.unlock();
                    }
                }

                c3d.parameter(groupName, newParam);
            }
        }
    }

    // Fill the data
    mxDouble* allDataPoints = mxGetDoubles(dataPoints);
    mxDouble* allResiduals = nullptr;
    mxDouble* allMasks = nullptr;
    if (dataMetaPoints){
        if (metadataResidual){
            allResiduals = mxGetDoubles(metadataResidual);
        }
        if (metadataCamMasks){
            allMasks = mxGetDoubles(metadataCamMasks);
        }
    }
    mxDouble* allDataAnalogs = mxGetDoubles(dataAnalogs);
    for (size_t f=0; f<nFrames; ++f){
        ezc3d::DataNS::Frame frame;
        ezc3d::DataNS::Points3dNS::Points pts;
        for (size_t i=0; i<nPoints; ++i){
            ezc3d::DataNS::Points3dNS::Point pt;
            pt.x(static_cast<float>(
                     allDataPoints[nPointsComponents*i+0+f*3*nPoints]));
            pt.y(static_cast<float>(
                     allDataPoints[nPointsComponents*i+1+f*3*nPoints]));
            pt.z(static_cast<float>(
                     allDataPoints[nPointsComponents*i+2+f*3*nPoints]));
            if (allResiduals) {
                pt.residual(static_cast<float>(allResiduals[i+f*nPoints]));
            }
            if (allMasks) {
                std::vector<bool> camMasks;
                for (size_t c = 0; c<7; ++c){
                    camMasks.push_back(
                                static_cast<bool>(allMasks[c+7*i+f*7*nPoints]));
                }
                pt.cameraMask(camMasks);
            }
            pts.point(pt);
        }

        ezc3d::DataNS::AnalogsNS::Analogs analogs;
        for (size_t sf=0; sf<nSubframes; ++sf){
            ezc3d::DataNS::AnalogsNS::SubFrame subframe;
            for (size_t i=0; i<nAnalogs; ++i){
                ezc3d::DataNS::AnalogsNS::Channel c;
                c.data(static_cast<float>(
                           allDataAnalogs[nFramesAnalogs*i+sf+f*nSubframes]));
                subframe.channel(c);
            }
            analogs.subframe(subframe);
        }
        frame.add(pts, analogs);
        c3d.frame(frame);// Add the previously created frame
    }
    c3d.write(path);
    return;
}

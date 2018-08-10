#include "mex.h"
#include <iostream>
#include <memory>

#include "ezc3d.h"
#include "utils.h"

int parseParam(mxDouble* data, const std::vector<int> &dimension,
               std::vector<int> &param_data, int idxInData=0, int currentIdx=0)
{
    if (dimension[currentIdx] == 0)
        return -1;

    for (int i=0; i<dimension[currentIdx]; ++i){
        if (currentIdx == dimension.size()-1){
            param_data.push_back (data[idxInData]);
            ++idxInData;
        }
        else
            idxInData = parseParam(data, dimension, param_data, idxInData, currentIdx + 1);
    }
    return idxInData;
}
int parseParam(mxDouble* data, const std::vector<int> &dimension,
               std::vector<float> &param_data, int idxInData=0, int currentIdx=0)
{
    if (dimension[currentIdx] == 0)
        return -1;

    for (int i=0; i<dimension[currentIdx]; ++i){
        if (currentIdx == dimension.size()-1){
            param_data.push_back (data[idxInData]);
            ++idxInData;
        }
        else
            idxInData = parseParam(data, dimension, param_data, idxInData, currentIdx + 1);
    }
    return idxInData;
}
int parseParam(mxArray* data, const std::vector<int> &dimension,
               std::vector<std::string> &param_data, int idxInData=0, int currentIdx=0)
{
    if (dimension[currentIdx] == 0)
        return -1;

    for (int i=0; i<dimension[currentIdx]; ++i){
        if (currentIdx == dimension.size()-1){
            mxArray *cell(mxGetCell(data, idxInData));
            mwSize pathlen = mxGetNumberOfElements(cell) + 1;
            char *path_tp = new char[pathlen];
            mxGetString(cell, path_tp, pathlen);
            std::string path(path_tp);
            delete[] path_tp;
            param_data.push_back (path);
            ++idxInData;
        }
        else
            idxInData = parseParam(data, dimension, param_data, idxInData, currentIdx + 1);
    }
    return idxInData;
}
int checkLongestStrParam(const std::vector<std::string> &param_data){
    int longest(0);
    for (int i=0; i<param_data.size(); ++i){
        if (param_data[i].size() > longest)
            longest = param_data[i].size();
    }
    return longest;
}


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

    // Receive the path
    mwSize pathlen = mxGetNumberOfElements(prhs[0]) + 1;
    char *path_tp = new char[pathlen];
    mxGetString(prhs[0], path_tp, pathlen);
    std::string path(path_tp);
    delete[] path_tp;
    if (path.size() == 0)
        mexErrMsgTxt("Input argument 1 must be a valid path to write.");
    std::string extension = ".c3d";
    if (path.find_last_of(".") > path.size() || path.substr(path.find_last_of(".")).compare(extension)){
        path += extension;
    }

    // Receive the structure
    const mxArray * c3dStruct(prhs[1]);
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

    //  Fill the parameters
    for (int g=0; g<mxGetNumberOfFields(parameter); ++g){ // top level
        std::string groupName(mxGetFieldNameByNumber(parameter, g));
        mxArray* groupField(mxGetFieldByNumber(parameter, 0, g));
        for (int p=0; p<mxGetNumberOfFields(groupField); ++p){
            std::string paramName(mxGetFieldNameByNumber(groupField, p));
            mxArray* paramField(mxGetFieldByNumber(groupField, 0, p));
            // Copy the parameters into the c3d, but skip those who will be updated later
            if ( !(!groupName.compare("POINT") && !paramName.compare("USED")) &&
                     !(!groupName.compare("POINT") && !paramName.compare("FRAMES")) &&
                     !(!groupName.compare("POINT") && !paramName.compare("LABELS")) &&
                     !(!groupName.compare("POINT") && !paramName.compare("DESCRIPTIONS")) &&

                     !(!groupName.compare("ANALOG") && !paramName.compare("USED")) &&
                     !(!groupName.compare("ANALOG") && !paramName.compare("LABELS")) &&
                     !(!groupName.compare("ANALOG") && !paramName.compare("DESCRIPTIONS")) &&
                     !(!groupName.compare("ANALOG") && !paramName.compare("SCALE")) &&
                     !(!groupName.compare("ANALOG") && !paramName.compare("OFFSET")) &&
                     !(!groupName.compare("ANALOG") && !paramName.compare("UNITS"))
                     ){
                std::vector<int> dimension;
                int nDim;
                if (!paramField)
                    nDim = 0;
                else
                    nDim =(mxGetNumberOfDimensions(paramField));
                if (nDim == 0)
                    dimension.push_back(0);
                else if (nDim == 2 && mxGetDimensions(paramField)[0] * mxGetDimensions(paramField)[1] == 0)
                    dimension.push_back(0);
                else if (nDim == 2 && mxGetDimensions(paramField)[0] * mxGetDimensions(paramField)[1] == 1)
                    dimension.push_back(1);
                else
                    for (int i=0; i<nDim; ++i)
                        dimension.push_back(mxGetDimensions(paramField)[i]);

                ezc3d::ParametersNS::GroupNS::Parameter newParam(paramName);
                if (c3d.parameters().groupIdx(groupName) != -1 &&
                        c3d.parameters().group(groupName).parameterIdx(paramName) != -1){
                    ezc3d::DATA_TYPE type(c3d.parameters().group(groupName).parameter(paramName).type());
                    if  (type == ezc3d::DATA_TYPE::INT || type == ezc3d::DATA_TYPE::BYTE) {
                        std::vector<int> data;
                        parseParam(mxGetDoubles(paramField), dimension, data);
                        newParam.set(data, dimension);
                    } else if (type == ezc3d::DATA_TYPE::FLOAT) {
                        std::vector<float> data;
                        parseParam(mxGetDoubles(paramField), dimension, data);
                        newParam.set(data, dimension);
                    } else if (type == ezc3d::DATA_TYPE::CHAR) {
                        std::vector<std::string> data;
                        parseParam(paramField, dimension, data);
                        int longest(checkLongestStrParam(data));
                        dimension.insert(dimension.begin(), {longest});
                        newParam.set(data, dimension);
                    }
                } else {
                    if (!paramField || mxIsDouble(paramField)){
                        std::vector<float> data;
                        parseParam(mxGetDoubles(paramField), dimension, data);
                        newParam.set(data, dimension);
                    } else if (mxIsCell(paramField)){
                        std::vector<std::string> data;
                        parseParam(paramField, dimension, data);
                        int longest(checkLongestStrParam(data));
                        dimension.insert(dimension.begin(), {longest});
                        newParam.set(data, dimension);
                    }
                }
                c3d.addParameter(groupName, newParam);
            }
        }
    }

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
    if (nAnalogs != mxGetN(parameterAnalogsLabels) * mxGetM(parameterAnalogsLabels))
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

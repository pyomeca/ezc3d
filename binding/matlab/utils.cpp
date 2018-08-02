#include "mex.h"
#include <iostream>
#include <memory>

#include "utils.h"

void fillMatlabField(mxArray *field, mwIndex idx, int value){
    // markers_Number
    mxArray* ptr = mxCreateDoubleMatrix(1, 1, mxREAL);
    double * val = mxGetPr(ptr);
    val[0] = (double)value;
    mxSetFieldByNumber(field, 0, idx, ptr);
}
void fillMatlabField(mxArray *field, mwIndex idx, const std::vector<int>& values){
    // markers_Number
    mxArray* ptr = mxCreateDoubleMatrix(values.size(), 1, mxREAL);
    double * val = mxGetPr(ptr);
    for (int i =0; i < values.size(); ++i)
        val[i] = (double)values[i];
    mxSetFieldByNumber(field, 0, idx, ptr);
}

void fillMatlabField(mxArray *field, mwIndex idx, double value){
    // markers_Number
    mxArray* ptr = mxCreateDoubleMatrix(1, 1, mxREAL);
    double * val = mxGetPr(ptr);
    val[0] = value;
    mxSetFieldByNumber(field, 0, idx, ptr);
}
void fillMatlabField(mxArray *field, mwIndex idx, const std::vector<float>& values){
    // markers_Number
    mxArray* ptr = mxCreateDoubleMatrix(values.size(), 1, mxREAL);
    double * val = mxGetPr(ptr);
    for (int i =0; i < values.size(); ++i)
        val[i] = (double)values[i];
    mxSetFieldByNumber(field, 0, idx, ptr);
}
void fillMatlabField(mxArray *field, mwIndex idx, const std::vector<double>& values){
    // markers_Number
    mxArray* ptr = mxCreateDoubleMatrix(values.size(), 1, mxREAL);
    double * val = mxGetPr(ptr);
    for (int i =0; i < values.size(); ++i)
        val[i] = values[i];
    mxSetFieldByNumber(field, 0, idx, ptr);
}

void fillMatlabField(mxArray *field, mwIndex idx, const std::string &value){
    // markers_Number
    mxSetFieldByNumber(field, 0, idx, mxCreateString(value.c_str()));
}
void fillMatlabField(mxArray *field, mwIndex idx, const std::vector<std::string>& values){
    // markers_Number
    mxArray* ptr = mxCreateCellMatrix(values.size(), 1);

    // Fill cell matrix with input arguments
    for (int i =0; i < values.size(); ++i)
        mxSetCell(ptr, i, mxCreateString(values[i].c_str()) );
    mxSetFieldByNumber(field, 0, idx, ptr);
}


int toInteger(const mxArray * prhs){
    return int( toDouble(prhs) );
}
double toDouble(const mxArray * prhs){
    return mxGetPr(prhs)[0];
}
bool toBool(const mxArray * prhs){
    if (mxIsDouble(prhs))
        return bool(mxGetPr(prhs)[0]);
    else if (mxIsLogical(prhs))
        return bool(mxGetLogicals(prhs)[0]);

}
std::string toString(const mxArray * prhs){
    char *str_char = mxArrayToString(prhs);
    std::string str(str_char);
    mxFree(str_char);
    return str;
}


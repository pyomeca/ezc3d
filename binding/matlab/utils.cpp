#include "mex.h"
#include <iostream>
#include <memory>

#include "utils.h"

void fillMatlabField(mxArray *field, mwIndex idx, size_t value){
    mxArray* ptr = mxCreateDoubleMatrix(1, 1, mxREAL);
    double * val = mxGetPr(ptr);
    val[0] = static_cast<double>(value);
    mxSetFieldByNumber(field, 0, static_cast<int>(idx), ptr);
}
void fillMatlabField(mxArray *field, mwIndex idx, int value){
    mxArray* ptr = mxCreateDoubleMatrix(1, 1, mxREAL);
    double * val = mxGetPr(ptr);
    val[0] = static_cast<double>(value);
    mxSetFieldByNumber(field, 0, static_cast<int>(idx), ptr);
}
void fillMatlabField(mxArray *field, mwIndex idx, const std::vector<int>& values, const std::vector<size_t>& dimension){
    mxArray* ptr;
    if (dimension.size() == 0)
        ptr = mxCreateDoubleMatrix(values.size(), 1, mxREAL);
    else {
        mwSize ndim(dimension.size());
        mwSize *dims = new mwSize[ndim];
        for (size_t i = 0; i < ndim; ++i)
            dims[i] = dimension[i];
        ptr = mxCreateNumericArray(ndim, dims, mxDOUBLE_CLASS, mxREAL);
        delete[] dims;
    }
    double * val = mxGetDoubles(ptr);
    for (size_t i =0; i < values.size(); ++i)
        val[i] = values[i];
    mxSetFieldByNumber(field, 0, static_cast<int>(idx), ptr);
}

void fillMatlabField(mxArray *field, mwIndex idx, double value){
    mxArray* ptr = mxCreateDoubleMatrix(1, 1, mxREAL);
    double * val = mxGetPr(ptr);
    val[0] = value;
    mxSetFieldByNumber(field, 0, static_cast<int>(idx), ptr);
}
void fillMatlabField(mxArray *field, mwIndex idx, const std::vector<float>& values, const std::vector<size_t>& dimension){
    mxArray* ptr;
    if (dimension.size() == 0)
        ptr = mxCreateDoubleMatrix(values.size(), 1, mxREAL);
    else {
        mwSize ndim(dimension.size());
        mwSize *dims = new mwSize[ndim];
        for (size_t i = 0; i < ndim; ++i)
            dims[i] = dimension[i];
        ptr = mxCreateNumericArray(ndim, dims, mxDOUBLE_CLASS, mxREAL);
        delete[] dims;
    }
    double * val = mxGetDoubles(ptr);
    for (size_t i =0; i < values.size(); ++i)
        val[i] = static_cast<double>(values[i]);
    mxSetFieldByNumber(field, 0, static_cast<int>(idx), ptr);
}
void fillMatlabField(mxArray *field, mwIndex idx, const std::vector<double>& values, const std::vector<size_t>& dimension){
    mxArray* ptr;
    if (dimension.size() == 0)
        ptr = mxCreateDoubleMatrix(values.size(), 1, mxREAL);
    else {
        mwSize ndim(dimension.size());
        mwSize *dims = new mwSize[ndim];
        for (size_t i = 0; i < ndim; ++i)
            dims[i] = dimension[i];
        ptr = mxCreateNumericArray(ndim, dims, mxDOUBLE_CLASS, mxREAL);
        delete[] dims;
    }
    double * val = mxGetDoubles(ptr);
    for (size_t i =0; i < values.size(); ++i)
        val[i] = values[i];
    mxSetFieldByNumber(field, 0, static_cast<int>(idx), ptr);
}

void fillMatlabField(mxArray *field, mwIndex idx, const std::string &value){
    mxSetFieldByNumber(field, 0, static_cast<int>(idx), mxCreateString(value.c_str()));
}
void fillMatlabField(mxArray *field, mwIndex idx, const std::vector<std::string>& values, const std::vector<size_t>& dimension){
    mxArray* ptr;
    if (dimension.size() <= 1)
            ptr = mxCreateCellMatrix(values.size(), 1);
    else {
        mwSize ndim(dimension.size()-1);
        mwSize *dims = new mwSize[ndim];
        for (size_t i = 0; i < ndim; ++i)
            dims[i] = dimension[i+1];
        ptr = mxCreateCellArray(ndim, dims);
        delete[] dims;
    }

    // Fill cell matrix with input arguments
    for (size_t i =0; i < values.size(); ++i)
        mxSetCell(ptr, i, mxCreateString(values[i].c_str()) );
    mxSetFieldByNumber(field, 0, static_cast<int>(idx), ptr);
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
    throw std::invalid_argument("Unrecognized type for parameter.");
}
std::string toString(const mxArray * prhs){
    char *str_char = mxArrayToString(prhs);
    std::string str(str_char);
    mxFree(str_char);
    return str;
}


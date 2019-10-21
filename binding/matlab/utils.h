#include "mex.h"
#include <iostream>
#include <memory>

#include "ezc3d.h"

#define METADATA_FIELD "META_DATA"
#define DESCRIPTION_FIELD "DESCRIPTION"
#define IS_LOCKED_FIELD "IS_LOCKED"
#define DATA_FIELD "DATA"

mxArray * fillMetadata(
        mxArray *field,
        mwIndex idx,
        const std::string& description,
        bool isLocked,
        bool withValueField);

// From values to matlab
void fillMatlabField(
        mxArray *field,
        mwIndex idx,
        size_t value);
void fillMatlabField(
        mxArray *field,
        mwIndex idx,
        int value);
void fillMatlabField(
        mxArray *field,
        mwIndex idx,
        const std::vector<int>& values,
        const std::vector<size_t>& dimension = {});

void fillMatlabField(
        mxArray *field,
        mwIndex idx,
        double value);
void fillMatlabField(
        mxArray *field,
        mwIndex idx,
        const std::vector<float>& values,
        const std::vector<size_t>& dimension = {});
void fillMatlabField(
        mxArray *field,
        mwIndex idx,
        const std::vector<double>& values,
        const std::vector<size_t>& dimension = {});

void fillMatlabField(
        mxArray *field,
        mwIndex idx,
        const std::string &value);
void fillMatlabField(
        mxArray *field,
        mwIndex idx,
        const std::vector<std::string>& values,
        const std::vector<size_t>& dimension = {});

// From matlab to values
int toInteger(
        const mxArray * prhs);
double toDouble(
        const mxArray * prhs);
bool toBool(
        const mxArray * prhs);
std::string toString(
        const mxArray * prhs);

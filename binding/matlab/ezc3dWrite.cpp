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

    // Make sure the top level is a valid c3d structure
    bool valid(true);
    std::vector<std::string> mandatoryFields = {"header", "parameter","data"};
    int nFields(mxGetNumberOfFields(c3dStruct));
    if (nFields != 3)
        valid = false;
    for (int i=0; i<nFields; ++i){
        bool found(false);
        std::string fieldName(mxGetFieldNameByNumber(c3dStruct, i));
        for (int j=0; j<mandatoryFields.size(); ++j){
            if (!fieldName.compare(mandatoryFields[j])){
                found = true;
                break;
            }
        }
        if (!found){
            valid = false;
            break;
        }
    }
    if (!valid){
        mexErrMsgTxt("Input argument must be a valid c3d structure.");
    }

    // Create a fresh c3d which will be fill with c3d struct
    //ezc3d c3d;

    return;
}

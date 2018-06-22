#include "mexClassHandler.h"
#include "ezc3d.h"
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	// Check inputs
	if(nrhs != 1 && (!mxIsChar(prhs[0]) || mxIsEmpty(prhs[0]) ))
		mexErrMsgTxt("Input argument must be a file path string.");

	if (nlhs != 1)
		mexErrMsgTxt("You must catch the pointer!");
		
	// Receive the path
	char *buffer = mxArrayToString(prhs[0]);
    std::string path(buffer);
    mxFree(buffer);

	try{
		plhs[0] = convertPtr2Mat<ezc3d::c3d>(new ezc3d::c3d(path));
	}
	catch (std::string m){
		mexErrMsgTxt(m.c_str());
	}
	
	return;
}
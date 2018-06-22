#include "mexClassHandler.h"
#include "ezc3d.h"
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	// Destroy
	destroyObject<ezc3d::c3d>(prhs[0]);
	
	// Warn if other commands were ignored
	if (nlhs != 0)
		mexWarnMsgTxt("Delete: Unexpected output arguments ignored.");
return;
}
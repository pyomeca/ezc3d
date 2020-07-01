import ezc3d

c3d = ezc3d.c3d("../c3dFiles/ezc3d-testFiles-master/ezc3d-testFiles-master/Qualisys.c3d", extract_forceplat_data=True)
c3d["data"]["platform"][0]["Tz"]

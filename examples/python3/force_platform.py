import ezc3d

# This example reads a file that contains 2 force platforms. It thereafter print some metadata and data for one them

c3d = ezc3d.c3d("../c3dFiles/ezc3d-testFiles-master/ezc3d-testFiles-master/Qualisys.c3d", extract_forceplat_data=True)

print(f"Number of force platform = {len(c3d['data']['platform'])}")
print("")
print("Printing information and data for force platform 0")
print("")
pf0 = c3d["data"]["platform"][0]
# Units
print(f"Force unit = {pf0['unit_force']}")
print(f"Moment unit = {pf0['unit_moment']}")
print(f"Center of pressure unit = {pf0['unit_position']}")
print("")
# Position of pf
print(f"Position of origin = {pf0['origin']}")
print(f"Position of corners = \n{pf0['corners']}")
print("")
# Calibration matrix
print(f"Calibation matrix = \n{pf0['cal_matrix']}")
print("")
# Data at 3 different time
frames = [0, 10, 1000, -1]
print(f"Data (in global reference frame) at frames = {frames}")
print(f"Force = \n{pf0['force'][:, frames]}")
print(f"Moment = \n{pf0['moment'][:, frames]}")
print(f"Center of pressure = \n{pf0['center_of_pressure'][:, frames]}")
print(f"Moment at CoP = \n{pf0['Tz'][:, frames]}")

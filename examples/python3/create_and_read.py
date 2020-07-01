import numpy as np

import ezc3d

# Load an empty c3d structure
c3d = ezc3d.c3d()

# Adjust some mandatory values of the parameters and fill the data with random values
c3d['parameters']['POINT']['RATE']['value'] = [100]
c3d['parameters']['POINT']['LABELS']['value'] = ('point1', 'point2', 'point3', 'point4', 'point5')
c3d['data']['points'] = np.random.rand(3, 5, 100)

c3d['parameters']['ANALOG']['RATE']['value'] = [1000]
c3d['parameters']['ANALOG']['LABELS']['value'] = ('analog1', 'analog2', 'analog3', 'analog4', 'analog5', 'analog6')
c3d['data']['analogs'] = np.random.rand(1, 6, 1000)

# Create a custom parameter to the POINT group
c3d.add_parameter("POINT", "newParam", [1, 2, 3])

# Create a custom parameter a new group
c3d.add_parameter("NewGroup", "newParam", ["MyParam1", "MyParam2"])

# Write a new modified C3D and read back the data
c3d.write("temporary.c3d")
c3d_to_compare = ezc3d.c3d("temporary.c3d")

# Print the header
print("# ---- HEADER ---- #")
print(f"Number of points = {c3d_to_compare['header']['points']['size']}")
print(f"Point frame rate = {c3d_to_compare['header']['points']['frame_rate']}")
print(f"Index of the first point frame = {c3d_to_compare['header']['points']['first_frame']}")
print(f"Index of the last point frame = {c3d_to_compare['header']['points']['last_frame']}")
print("")
print(f"Number of analogs = {c3d_to_compare['header']['analogs']['size']}")
print(f"Analog frame rate = {c3d_to_compare['header']['analogs']['frame_rate']}")
print(f"Index of the first analog frame = {c3d_to_compare['header']['analogs']['first_frame']}")
print(f"Index of the last analog frame = {c3d_to_compare['header']['analogs']['last_frame']}")
print("")
print("")
# Print the parameters
print("# ---- PARAMETERS ---- #")
print(f"Number of points = {c3d_to_compare['parameters']['POINT']['USED']['value'][0]}")
print(f"Name of the points = {c3d_to_compare['parameters']['POINT']['LABELS']['value']}")
print(f"Point frame rate = {c3d_to_compare['parameters']['POINT']['RATE']['value'][0]}")
print(f"Number of frames = {c3d_to_compare['parameters']['POINT']['FRAMES']['value'][0]}")
print(f"My point new Param = {c3d_to_compare['parameters']['POINT']['NEWPARAM']['value']}")
print("")
print(f"Number of analogs = {c3d_to_compare['parameters']['ANALOG']['USED']['value'][0]}")
print(f"Name of the analogs = {c3d_to_compare['parameters']['ANALOG']['LABELS']['value']}")
print(f"Analog frame rate = {c3d_to_compare['parameters']['ANALOG']['RATE']['value'][0]}")
print("")
print(f"My NewGroup new Param = {c3d_to_compare['parameters']['NEWGROUP']['NEWPARAM']['value']}")
print("")
print("")
# Print the data
print("# ---- DATA ---- #")
print(f" = {c3d_to_compare['data']['points'][0:3, :, :]}")
print(f" = {c3d_to_compare['data']['analogs']}")


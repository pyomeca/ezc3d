"""
Test for file IO
"""
from pathlib import Path
from copy import deepcopy

import numpy as np
import pytest

import ezc3d


def test_create_c3d():
    c3d = ezc3d.c3d()

    # Test the header
    assert c3d["header"]["points"]["size"] == 0
    assert c3d["header"]["points"]["frame_rate"] == 0.0
    assert c3d["header"]["points"]["first_frame"] == 0
    assert c3d["header"]["points"]["last_frame"] == 0

    assert c3d["header"]["analogs"]["size"] == 0
    assert c3d["header"]["analogs"]["frame_rate"] == 0.0
    assert c3d["header"]["analogs"]["first_frame"] == 0
    assert c3d["header"]["analogs"]["last_frame"] == -1

    assert c3d["header"]["events"]["size"] == 18
    assert c3d["header"]["events"]["events_time"] == (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    assert c3d["header"]["events"]["events_label"] == (
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    )

    # Test the parameters
    assert c3d["parameters"]["POINT"]["USED"]["value"][0] == 0
    assert c3d["parameters"]["POINT"]["SCALE"]["value"][0] == -1
    assert c3d["parameters"]["POINT"]["RATE"]["value"][0] == 0.0
    assert c3d["parameters"]["POINT"]["FRAMES"]["value"][0] == 0
    assert len(c3d["parameters"]["POINT"]["LABELS"]["value"]) == 0
    assert len(c3d["parameters"]["POINT"]["DESCRIPTIONS"]["value"]) == 0
    assert len(c3d["parameters"]["POINT"]["UNITS"]["value"]) == 0

    assert c3d["parameters"]["ANALOG"]["USED"]["value"][0] == 0
    assert len(c3d["parameters"]["ANALOG"]["LABELS"]["value"]) == 0
    assert len(c3d["parameters"]["ANALOG"]["DESCRIPTIONS"]["value"]) == 0
    assert c3d["parameters"]["ANALOG"]["GEN_SCALE"]["value"][0] == 1
    assert len(c3d["parameters"]["ANALOG"]["SCALE"]["value"]) == 0
    assert len(c3d["parameters"]["ANALOG"]["OFFSET"]["value"]) == 0
    assert len(c3d["parameters"]["ANALOG"]["UNITS"]["value"]) == 0
    assert c3d["parameters"]["ANALOG"]["RATE"]["value"][0] == 0.0
    assert len(c3d["parameters"]["ANALOG"]["FORMAT"]["value"]) == 0
    assert len(c3d["parameters"]["ANALOG"]["BITS"]["value"]) == 0

    assert c3d["parameters"]["FORCE_PLATFORM"]["USED"]["value"][0] == 0
    assert len(c3d["parameters"]["FORCE_PLATFORM"]["TYPE"]["value"]) == 0
    assert np.all(c3d["parameters"]["FORCE_PLATFORM"]["ZERO"]["value"] == (1, 0))
    assert len(c3d["parameters"]["FORCE_PLATFORM"]["CORNERS"]["value"]) == 0
    assert len(c3d["parameters"]["FORCE_PLATFORM"]["ORIGIN"]["value"]) == 0
    assert len(c3d["parameters"]["FORCE_PLATFORM"]["CHANNEL"]["value"]) == 0
    assert len(c3d["parameters"]["FORCE_PLATFORM"]["CAL_MATRIX"]["value"]) == 0

    # Test the data
    assert c3d["data"]["points"].shape == (4, 0, 0)
    assert c3d["data"]["analogs"].shape == (1, 0, 0)


def test_deepcopy():
    # Load an empty c3d structure
    c3d = ezc3d.c3d()

    # Fill it with random data
    point_names = ("point1", "point2", "point3", "point4", "point5")
    point_frame_rate = 100
    n_second = 2
    points = np.random.rand(4, len(point_names), point_frame_rate * n_second)
    points[3, :, :] = 1

    analog_names = ("analog1", "analog2", "analog3", "analog4", "analog5", "analog6")
    analog_frame_rate = 1000
    analogs = np.random.rand(1, len(analog_names), analog_frame_rate * n_second)

    c3d["parameters"]["POINT"]["RATE"]["value"] = [100]
    c3d["parameters"]["POINT"]["LABELS"]["value"] = point_names
    c3d["data"]["points"] = points

    c3d["parameters"]["ANALOG"]["RATE"]["value"] = [1000]
    c3d["parameters"]["ANALOG"]["LABELS"]["value"] = analog_names
    c3d["data"]["analogs"] = analogs

    # Add a custom parameter to the POINT group
    point_new_param = ("POINT", "newPointParam", (1.0, 2.0, 3.0))
    c3d.add_parameter(point_new_param[0], point_new_param[1], point_new_param[2])

    # Add a custom parameter a new group
    new_group_param = ("NewGroup", "newGroupParam", ["MyParam1", "MyParam2"])
    c3d.add_parameter(new_group_param[0], new_group_param[1], new_group_param[2])

    # Deepcopy the c3d
    c3d_deepcopied = deepcopy(c3d)

    # Change some of its values
    change_new_group_param = ("NewGroup", "newGroupParam", ["MyParam3", "MyParam4"])
    c3d_deepcopied.add_parameter(change_new_group_param[0], change_new_group_param[1], change_new_group_param[2])
    c3d_deepcopied["data"]["points"][:3, :, :] = 0

    # Write the new file and read it back
    c3d_deepcopied.write("temporary.c3d")
    c3d_loaded = ezc3d.c3d("temporary.c3d")

    # Check that the new value changed, but not the old one
    assert c3d["parameters"]["NewGroup"]["newGroupParam"]["value"] == ["MyParam1", "MyParam2"]
    assert c3d_deepcopied["parameters"]["NewGroup"]["newGroupParam"]["value"] == ["MyParam3", "MyParam4"]
    assert c3d_loaded["parameters"]["NewGroup"]["newGroupParam"]["value"] == ["MyParam3", "MyParam4"]

    np.testing.assert_almost_equal(
        c3d["data"]["points"][:3, :, :] - c3d_deepcopied["data"]["points"][:3, :, :], c3d["data"]["points"][:3, :, :]
    )
    np.testing.assert_almost_equal(c3d["data"]["points"][3, :, :], c3d_deepcopied["data"]["points"][3, :, :])
    np.testing.assert_almost_equal(
        c3d["data"]["points"][:3, :, :] - c3d_loaded["data"]["points"][:3, :, :], c3d["data"]["points"][:3, :, :]
    )
    np.testing.assert_almost_equal(c3d["data"]["points"][3, :, :], c3d_loaded["data"]["points"][3, :, :])


def test_create_and_read_c3d():
    # Load an empty c3d structure
    c3d = ezc3d.c3d()

    # Fill it with random data
    point_names = ("point1", "point2", "point3", "point4", "point5")
    point_frame_rate = 100
    n_second = 2
    points = np.random.rand(3, len(point_names), point_frame_rate * n_second)

    analog_names = ("analog1", "analog2", "analog3", "analog4", "analog5", "analog6")
    analog_frame_rate = 1000
    analogs = np.random.rand(1, len(analog_names), analog_frame_rate * n_second)

    c3d["parameters"]["POINT"]["RATE"]["value"] = [100]
    c3d["parameters"]["POINT"]["LABELS"]["value"] = point_names
    c3d["data"]["points"] = points

    c3d["parameters"]["ANALOG"]["RATE"]["value"] = [1000]
    c3d["parameters"]["ANALOG"]["LABELS"]["value"] = analog_names
    c3d["data"]["analogs"] = analogs

    # Add a custom parameter to the POINT group
    point_new_param = ("POINT", "newPointParam", (1.0, 2.0, 3.0))
    c3d.add_parameter(point_new_param[0], point_new_param[1], point_new_param[2])

    # Add a custom parameter a new group
    new_group_param = ("NewGroup", "newGroupParam", ["MyParam1", "MyParam2"])
    c3d.add_parameter(new_group_param[0], new_group_param[1], new_group_param[2])

    # Write and read back the data
    c3d.write("temporary.c3d")
    c3d_to_compare = ezc3d.c3d("temporary.c3d")

    # Test the header
    assert c3d_to_compare["header"]["points"]["size"] == len(point_names)
    assert c3d_to_compare["header"]["points"]["frame_rate"] == point_frame_rate
    assert c3d_to_compare["header"]["points"]["first_frame"] == 0
    assert c3d_to_compare["header"]["points"]["last_frame"] == point_frame_rate * n_second - 1

    assert c3d_to_compare["header"]["analogs"]["size"] == len(analog_names)
    assert c3d_to_compare["header"]["analogs"]["frame_rate"] == analog_frame_rate
    assert c3d_to_compare["header"]["analogs"]["first_frame"] == 0
    assert c3d_to_compare["header"]["analogs"]["last_frame"] == analog_frame_rate * n_second - 1

    assert c3d_to_compare["header"]["events"]["size"] == 18
    assert c3d_to_compare["header"]["events"]["events_time"] == (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    assert c3d_to_compare["header"]["events"]["events_label"] == (
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    )

    # Test the parameters
    assert c3d_to_compare["parameters"]["POINT"]["USED"]["value"][0] == len(point_names)
    assert c3d_to_compare["parameters"]["POINT"]["SCALE"]["value"][0] == -1.0
    assert c3d_to_compare["parameters"]["POINT"]["RATE"]["value"][0] == point_frame_rate
    assert c3d_to_compare["parameters"]["POINT"]["FRAMES"]["value"][0] == point_frame_rate * n_second
    assert c3d_to_compare["parameters"]["POINT"]["LABELS"]["value"] == list(point_names)
    assert c3d_to_compare["parameters"]["POINT"]["DESCRIPTIONS"]["value"] == ["" for _ in point_names]
    assert len(c3d_to_compare["parameters"]["POINT"]["UNITS"]["value"]) == 0
    assert np.all(c3d_to_compare["parameters"][point_new_param[0]][point_new_param[1]]["value"] == point_new_param[2])

    assert c3d_to_compare["parameters"]["ANALOG"]["USED"]["value"][0] == len(analog_names)
    assert c3d_to_compare["parameters"]["ANALOG"]["LABELS"]["value"] == list(analog_names)
    assert c3d_to_compare["parameters"]["ANALOG"]["DESCRIPTIONS"]["value"] == ["" for _ in analog_names]
    assert c3d_to_compare["parameters"]["ANALOG"]["GEN_SCALE"]["value"][0] == 1
    assert np.all(c3d_to_compare["parameters"]["ANALOG"]["SCALE"]["value"] == tuple([1.0 for _ in analog_names]))
    assert np.all(c3d_to_compare["parameters"]["ANALOG"]["OFFSET"]["value"] == tuple([0 for _ in analog_names]))
    assert c3d_to_compare["parameters"]["ANALOG"]["UNITS"]["value"] == ["" for _ in analog_names]
    assert c3d_to_compare["parameters"]["ANALOG"]["RATE"]["value"][0] == analog_frame_rate
    assert len(c3d_to_compare["parameters"]["ANALOG"]["FORMAT"]["value"]) == 0
    assert len(c3d_to_compare["parameters"]["ANALOG"]["BITS"]["value"]) == 0

    assert c3d_to_compare["parameters"]["FORCE_PLATFORM"]["USED"]["value"][0] == 0
    assert len(c3d_to_compare["parameters"]["FORCE_PLATFORM"]["TYPE"]["value"]) == 0
    assert np.all(c3d_to_compare["parameters"]["FORCE_PLATFORM"]["ZERO"]["value"] == (1, 0))
    assert len(c3d_to_compare["parameters"]["FORCE_PLATFORM"]["CORNERS"]["value"]) == 0
    assert len(c3d_to_compare["parameters"]["FORCE_PLATFORM"]["ORIGIN"]["value"]) == 0
    assert len(c3d_to_compare["parameters"]["FORCE_PLATFORM"]["CHANNEL"]["value"]) == 0
    assert len(c3d_to_compare["parameters"]["FORCE_PLATFORM"]["CAL_MATRIX"]["value"]) == 0

    assert c3d_to_compare["parameters"][new_group_param[0]][new_group_param[1]]["value"] == new_group_param[2]

    # Test the data
    assert c3d_to_compare["data"]["points"].shape == (4, len(point_names), point_frame_rate * n_second)
    assert c3d_to_compare["data"]["analogs"].shape == (1, len(analog_names), analog_frame_rate * n_second)

    # Compare the read c3d
    np.testing.assert_almost_equal(c3d_to_compare["data"]["points"][0:3, :, :], points)
    np.testing.assert_almost_equal(c3d_to_compare["data"]["analogs"], analogs)


def test_create_and_read_c3d_with_nan():
    # Load an empty c3d structure
    c3d = ezc3d.c3d()

    # Fill it with random data
    point_names = ("point1", "point2")
    point_frame_rate = 100
    n_second = 2
    points = np.random.rand(3, len(point_names), point_frame_rate * n_second) * np.nan

    analog_names = ("analog1", "analog2")
    analog_frame_rate = 1000
    analogs = np.random.rand(1, len(analog_names), analog_frame_rate * n_second) * np.nan

    c3d["parameters"]["POINT"]["RATE"]["value"] = [100]
    c3d["parameters"]["POINT"]["LABELS"]["value"] = point_names
    c3d["data"]["points"] = points

    c3d["parameters"]["ANALOG"]["RATE"]["value"] = [1000]
    c3d["parameters"]["ANALOG"]["LABELS"]["value"] = analog_names
    c3d["data"]["analogs"] = analogs

    # Write and read back the data
    c3d.write("temporary.c3d")
    c3d_to_compare = ezc3d.c3d("temporary.c3d")

    # Compare the read c3d
    np.testing.assert_equal(
        np.sum(np.isnan(c3d_to_compare["data"]["points"])), 3 * len(point_names) * point_frame_rate * n_second
    )
    np.testing.assert_equal(
        np.sum(np.isnan(c3d_to_compare["data"]["analogs"])), len(analog_names) * analog_frame_rate * n_second
    )


def test_add_events():
    # Add an event to a file that does not have any before
    c3d = ezc3d.c3d("test/c3dTestFiles/Optotrak.c3d")
    c3d.add_event(
        [0, 0.1],
        label="MyNewEvent",
        context="Left",
        icon_id=2,
        subject="Me",
        description="Hey! This is new!",
        generic_flag=1,
    )
    c3d.add_event([0, 0.2])
    c3d.write("temporary.c3d")
    c3d_to_compare = ezc3d.c3d("temporary.c3d")
    np.testing.assert_equal(c3d_to_compare["parameters"]["EVENT"]["USED"]["value"][0], 2)
    np.testing.assert_almost_equal(
        c3d_to_compare["parameters"]["EVENT"]["TIMES"]["value"], [[0.0, 0.0], [0.1, 0.2]], decimal=6
    )
    np.testing.assert_equal(c3d_to_compare["parameters"]["EVENT"]["CONTEXTS"]["value"], ["Left", ""])
    np.testing.assert_equal(c3d_to_compare["parameters"]["EVENT"]["LABELS"]["value"], ["MyNewEvent", ""])
    np.testing.assert_equal(c3d_to_compare["parameters"]["EVENT"]["DESCRIPTIONS"]["value"], ["Hey! This is new!", ""])
    np.testing.assert_equal(c3d_to_compare["parameters"]["EVENT"]["SUBJECTS"]["value"], ["Me", ""])
    np.testing.assert_equal(c3d_to_compare["parameters"]["EVENT"]["ICON_IDS"]["value"], [2, 0])
    np.testing.assert_equal(c3d_to_compare["parameters"]["EVENT"]["GENERIC_FLAGS"]["value"], [1, 0])

    # Add an event to a file did have events before
    c3d = c3d_to_compare
    c3d.add_event(
        [0, 0.3],
        label="MySecondNewEvent",
        context="Right",
        icon_id=3,
        subject="You",
        description="Hey! This is new again!",
        generic_flag=2,
    )
    c3d.add_event([0, 0.4])
    c3d.write("temporary.c3d")
    c3d_to_compare = ezc3d.c3d("temporary.c3d")
    np.testing.assert_equal(c3d_to_compare["parameters"]["EVENT"]["USED"]["value"][0], 4)
    np.testing.assert_almost_equal(
        c3d_to_compare["parameters"]["EVENT"]["TIMES"]["value"], [[0.0, 0.0, 0.0, 0.0], [0.1, 0.2, 0.3, 0.4]], decimal=6
    )
    np.testing.assert_equal(c3d_to_compare["parameters"]["EVENT"]["CONTEXTS"]["value"], ["Left", "", "Right", ""])
    np.testing.assert_equal(
        c3d_to_compare["parameters"]["EVENT"]["LABELS"]["value"], ["MyNewEvent", "", "MySecondNewEvent", ""]
    )
    np.testing.assert_equal(
        c3d_to_compare["parameters"]["EVENT"]["DESCRIPTIONS"]["value"],
        ["Hey! This is new!", "", "Hey! This is new again!", ""],
    )
    np.testing.assert_equal(c3d_to_compare["parameters"]["EVENT"]["SUBJECTS"]["value"], ["Me", "", "You", ""])
    np.testing.assert_equal(c3d_to_compare["parameters"]["EVENT"]["ICON_IDS"]["value"], [2, 0, 3, 0])
    np.testing.assert_equal(c3d_to_compare["parameters"]["EVENT"]["GENERIC_FLAGS"]["value"], [1, 0, 2, 0])


def test_values():
    c3d = ezc3d.c3d("test/c3dTestFiles/Vicon.c3d")
    array = c3d["data"]["points"]
    decimal = 6

    np.testing.assert_array_equal(x=array.shape, y=(4, 51, 580), err_msg="Shape does not match")
    raveled = array.ravel()
    np.testing.assert_array_almost_equal(
        x=raveled[0],
        y=44.16278839111328,
        decimal=decimal,
    )
    np.testing.assert_array_almost_equal(
        x=raveled[-1],
        y=1.0,
        decimal=decimal,
    )
    np.testing.assert_array_almost_equal(x=np.nanmean(array), y=362.2979849093196, decimal=decimal)
    np.testing.assert_array_almost_equal(x=np.nanmedian(array), y=337.7519226074219, decimal=decimal)
    np.testing.assert_allclose(actual=np.nansum(array), desired=42535594.91827867, rtol=0.05)
    np.testing.assert_array_equal(x=np.isnan(array).sum(), y=915)


def test_force_platform_filter():
    c3d = ezc3d.c3d("test/c3dTestFiles/Qualisys.c3d", extract_forceplat_data=True)
    all_pf = c3d["data"]["platform"]
    np.testing.assert_equal(len(all_pf), 2)

    # Frames
    np.testing.assert_equal(all_pf[0]["force"].shape[1], 3400)
    np.testing.assert_equal(all_pf[0]["moment"].shape[1], 3400)
    np.testing.assert_equal(all_pf[0]["center_of_pressure"].shape[1], 3400)
    np.testing.assert_equal(all_pf[0]["Tz"].shape[1], 3400)

    np.testing.assert_equal(all_pf[1]["force"].shape[1], 3400)
    np.testing.assert_equal(all_pf[1]["moment"].shape[1], 3400)
    np.testing.assert_equal(all_pf[1]["center_of_pressure"].shape[1], 3400)
    np.testing.assert_equal(all_pf[1]["Tz"].shape[1], 3400)

    # Units
    np.testing.assert_string_equal(all_pf[0]["unit_force"], "N")
    np.testing.assert_string_equal(all_pf[0]["unit_moment"], "Nmm")
    np.testing.assert_string_equal(all_pf[0]["unit_position"], "mm")

    np.testing.assert_string_equal(all_pf[1]["unit_force"], "N")
    np.testing.assert_string_equal(all_pf[1]["unit_moment"], "Nmm")
    np.testing.assert_string_equal(all_pf[1]["unit_position"], "mm")

    # Position of pf
    np.testing.assert_array_almost_equal(all_pf[0]["origin"], [1.524, -0.762, -34.036])
    np.testing.assert_array_almost_equal(
        all_pf[0]["corners"], [[508, 508, 0, 0], [464, 0, 0, 464], [0, 0, 0, 0]], decimal=3
    )

    np.testing.assert_array_almost_equal(all_pf[1]["origin"], [1.016, 0, -36.322])
    np.testing.assert_array_almost_equal(
        all_pf[1]["corners"], [[1017, 1017, 509, 509], [464, 0, 0, 464], [0, 0, 0, 0]], decimal=3
    )

    # Calibration matrix
    np.testing.assert_array_almost_equal(all_pf[0]["cal_matrix"], np.zeros((6, 6)))

    np.testing.assert_array_almost_equal(all_pf[1]["cal_matrix"], np.zeros((6, 6)))

    # Data at 3 different time
    expected_force = [[0.140, 106.480, -0.140], [0.046, -66.407, -0.138], [-0.184, 763.647, 0.367]]
    expected_moment = [[20.868, 54768.655, 51.780], [-4.623, -24103.676, 4.483], [-29.393, -12229.124, -29.960]]
    expected_cop = [[228.813, 285.564, 241.787], [118.296, 303.720, 373.071], [0, 0, 0]]
    expected_Tz = [[0, 0, 0], [0, 0, 0], [-44.141, -2496.299, -51.390]]
    np.testing.assert_array_almost_equal(all_pf[0]["force"][:, [0, 1000, -1]], expected_force, decimal=3)
    np.testing.assert_array_almost_equal(all_pf[0]["moment"][:, [0, 1000, -1]], expected_moment, decimal=3)
    np.testing.assert_array_almost_equal(all_pf[0]["center_of_pressure"][:, [0, 1000, -1]], expected_cop, decimal=3)
    np.testing.assert_array_almost_equal(all_pf[0]["Tz"][:, [0, 1000, -1]], expected_Tz, decimal=3)

    expected_force = [[0.046, 0.232, 0.185], [-0.185, -0.184, -0.046], [0.723, 0.361, 0.542]]
    expected_moment = [[49.366, 68.671, 16.708], [-96.907, -46.501, 50.403], [0.047, -19.720, 30.122]]
    expected_cop = [[897.0422, 891.673, 670.044], [300.283, 422.019, 262.813], [0, 0, 0]]
    expected_Tz = [[0, 0, 0], [0, 0, 0], [27.944, 48.016, 31.545]]
    np.testing.assert_array_almost_equal(all_pf[1]["force"][:, [0, 1000, -1]], expected_force, decimal=3)
    np.testing.assert_array_almost_equal(all_pf[1]["moment"][:, [0, 1000, -1]], expected_moment, decimal=3)
    np.testing.assert_array_almost_equal(all_pf[1]["center_of_pressure"][:, [0, 1000, -1]], expected_cop, decimal=3)
    np.testing.assert_array_almost_equal(all_pf[1]["Tz"][:, [0, 1000, -1]], expected_Tz, decimal=3)


def test_rotations():
    c3d = ezc3d.c3d("test/c3dTestFiles/C3DRotationExample.c3d")
    array = c3d["data"]["rotations"]
    decimal = 6

    np.testing.assert_array_equal(x=array.shape, y=(4, 4, 21, 340), err_msg="Shape does not match")
    raveled = array.ravel()
    np.testing.assert_array_almost_equal(
        x=array[2, 3, 2, 5],
        y=931.6382446289062,
        decimal=decimal,
    )
    np.testing.assert_array_almost_equal(
        x=raveled[-1],
        y=1.0,
        decimal=decimal,
    )
    np.testing.assert_array_almost_equal(x=np.nansum(array), y=9367125.137371363, decimal=decimal)


@pytest.fixture(scope="module", params=["BTS", "Optotrak", "Qualisys", "Vicon", "Label2"])
def c3d_build_rebuild_all(request):
    base_folder = Path("test/c3dTestFiles")
    orig_file = Path(base_folder / (request.param + ".c3d"))
    rebuild_file = Path(base_folder / (request.param + "_after.c3d"))

    original = ezc3d.c3d(orig_file.as_posix())
    original.write(rebuild_file.as_posix())
    rebuilt = ezc3d.c3d(rebuild_file.as_posix())

    yield (original, rebuilt)

    Path.unlink(rebuild_file)


@pytest.fixture(scope="module", params=["BTS", "Optotrak", "Qualisys", "Vicon", "C3DRotationExample"])
def c3d_build_rebuild_reduced(request):
    base_folder = Path("test/c3dTestFiles")
    orig_file = Path(base_folder / (request.param + ".c3d"))
    rebuild_file = Path(base_folder / (request.param + "_after.c3d"))

    original = ezc3d.c3d(orig_file.as_posix())
    original.write(rebuild_file.as_posix())
    rebuilt = ezc3d.c3d(rebuild_file.as_posix())
    if request.param == "C3DRotationExample":
        rebuilt["parameters"]["ROTATION"]["DATA_START"]["value"][0] = 6

    yield (original, rebuilt)

    Path.unlink(rebuild_file)


def test_parse_and_rebuild(c3d_build_rebuild_all):
    for i in c3d_build_rebuild_all:
        assert isinstance(i, ezc3d.c3d)
    orig, rebuilt = c3d_build_rebuild_all
    assert orig == rebuilt


def test_parse_and_rebuild_header(c3d_build_rebuild_all):
    orig, rebuilt = c3d_build_rebuild_all
    assert orig["header"] == rebuilt["header"]


def test_parse_and_rebuild_parameters(c3d_build_rebuild_reduced):
    orig, rebuilt = c3d_build_rebuild_reduced
    for group_key in orig.parameters._storage:
        for param_key in orig.parameters[group_key]:
            if not isinstance(orig.parameters[group_key][param_key], dict):
                # Only test the values that are actual parameters
                continue
            if "type" not in orig.parameters[group_key][param_key]:
                # Only test the values that are actual parameters
                continue

            if param_key == "DATA_START":
                # Skip DATA_START as it is an internal value
                continue

            try:
                assert orig.parameters[group_key][param_key]['type'] == rebuilt.parameters[group_key][param_key]['type']
            except:
                # Type may differ for empty values
                if not orig.parameters[group_key][param_key]['value'] and not rebuilt.parameters[group_key][param_key]['value']:
                    pass
                else:
                    assert orig.parameters[group_key][param_key]['type'] == rebuilt.parameters[group_key][param_key]['type']
            assert orig.parameters[group_key][param_key]['description'] == rebuilt.parameters[group_key][param_key]['description']
            assert orig.parameters[group_key][param_key]['is_locked'] == rebuilt.parameters[group_key][param_key]['is_locked']
            assert np.all(orig.parameters[group_key][param_key]['value'] == rebuilt.parameters[group_key][param_key]['value'])
    


def test_parse_and_rebuild_data(c3d_build_rebuild_all):
    orig, rebuilt = c3d_build_rebuild_all
    assert orig["data"] == rebuilt["data"]

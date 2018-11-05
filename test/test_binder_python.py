"""
Test for file IO
"""
from pathlib import Path

import numpy as np
import pytest

import ezc3d

def test_create_c3d():
    c3d = ezc3d.c3d()
    
    # Test the header 
    assert c3d['header']['points']['size'] == 0
    assert c3d['header']['points']['frame_rate'] == 0.0
    assert c3d['header']['points']['first_frame'] == 0
    assert c3d['header']['points']['last_frame'] == 0
    
    assert c3d['header']['analogs']['size'] == 0
    assert c3d['header']['analogs']['frame_rate'] == 0.0
    assert c3d['header']['analogs']['first_frame'] == 0
    assert c3d['header']['analogs']['last_frame'] == 0
    
    assert c3d['header']['events']['size'] == 18
    assert c3d['header']['events']['events_time'] == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert c3d['header']['events']['events_label'] == ('', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '')
    
    # Test the parameters
    assert c3d['parameters']['POINT']['USED']['value'][0] == 0
    assert c3d['parameters']['POINT']['SCALE']['value'][0] == -1
    assert c3d['parameters']['POINT']['RATE']['value'][0] == 0.0
    assert c3d['parameters']['POINT']['FRAMES']['value'][0] == 0
    assert len(c3d['parameters']['POINT']['LABELS']['value']) == 0
    assert len(c3d['parameters']['POINT']['DESCRIPTIONS']['value']) == 0
    assert len(c3d['parameters']['POINT']['UNITS']['value']) == 0
    
    assert c3d['parameters']['ANALOG']['USED']['value'][0] == 0
    assert len(c3d['parameters']['ANALOG']['LABELS']['value']) == 0
    assert len(c3d['parameters']['ANALOG']['DESCRIPTIONS']['value']) == 0
    assert c3d['parameters']['ANALOG']['GEN_SCALE']['value'][0] == 1
    assert len(c3d['parameters']['ANALOG']['SCALE']['value']) == 0
    assert len(c3d['parameters']['ANALOG']['OFFSET']['value']) == 0
    assert len(c3d['parameters']['ANALOG']['UNITS']['value']) == 0
    assert c3d['parameters']['ANALOG']['RATE']['value'][0] == 0.0
    assert len(c3d['parameters']['ANALOG']['FORMAT']['value']) == 0
    assert len(c3d['parameters']['ANALOG']['BITS']['value']) == 0
    
    assert c3d['parameters']['FORCE_PLATFORM']['USED']['value'][0] == 0
    assert len(c3d['parameters']['FORCE_PLATFORM']['TYPE']['value']) == 0
    assert c3d['parameters']['FORCE_PLATFORM']['ZERO']['value'] == (1, 0)
    assert len(c3d['parameters']['FORCE_PLATFORM']['CORNERS']['value']) == 0
    assert len(c3d['parameters']['FORCE_PLATFORM']['ORIGIN']['value']) == 0
    assert len(c3d['parameters']['FORCE_PLATFORM']['CHANNEL']['value']) == 0
    assert len(c3d['parameters']['FORCE_PLATFORM']['CAL_MATRIX']['value']) == 0
    
    # Test the data
    assert c3d['data']['points'].shape == (4, 0, 0)
    assert c3d['data']['analogs'].shape == (1, 0, 0)
    
    
def test_create_and_read_c3d():
    import numpy as np

    import ezc3d

    # Load an empty c3d structure
    c3d = ezc3d.c3d()

    # Fill it with random data
    points = np.random.rand(3, 5, 100)
    analogs = np.random.rand(1, 6, 1000)

    c3d['parameters']['POINT']['RATE']['value'] = [100]
    c3d['parameters']['POINT']['LABELS']['value'] = ('point1', 'point2', 'point3', 'point4', 'point5')
    c3d['data']['points'] = points

    c3d['parameters']['ANALOG']['RATE']['value'] = [1000]
    c3d['parameters']['ANALOG']['LABELS']['value'] = ('analog1', 'analog2', 'analog3', 'analog4', 'analog5', 'analog6')
    c3d['data']['analogs'] = analogs

    # Write and read back the data
    c3d.write("temporary.c3d")
    c3d_to_compare = ezc3d.c3d("temporary.c3d")

    # Compare the read c3d
    np.testing.assert_almost_equal(c3d_to_compare['data']['points'][0:3, :, :], points)
    np.testing.assert_almost_equal(c3d_to_compare['data']['analogs'], analogs)


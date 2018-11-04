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
    
    
def test_load_c3d():
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


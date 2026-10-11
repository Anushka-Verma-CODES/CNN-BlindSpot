import sys
import os
import pytest

sys.path.append(os.path.abspath("src"))
from game import get_missions, check_mission_complete

def test_mission_1_first_contact():
    m = get_missions()[0]
    is_complete, _ = check_mission_complete(m, 30.0, 30.0, move_count=0)
    assert not is_complete
    is_complete, _ = check_mission_complete(m, 30.0, 25.0, move_count=1)
    assert is_complete
    is_complete, reason = check_mission_complete(m, 30.0, 25.0, move_count=4)
    assert not is_complete
    assert "exceeded" in reason.lower()

def test_mission_2_break_prediction():
    m = get_missions()[1]
    is_complete, _ = check_mission_complete(m, 30.0, 25.0, move_count=3)
    assert not is_complete
    is_complete, _ = check_mission_complete(m, 30.0, 10.0, move_count=3)
    assert is_complete
    is_complete, reason = check_mission_complete(m, 30.0, 10.0, move_count=4)
    assert not is_complete
    
def test_mission_4_increase_score():
    m = get_missions()[3] 
    is_complete, _ = check_mission_complete(m, 30.0, 28.0, move_count=1)
    assert not is_complete
    is_complete, _ = check_mission_complete(m, 30.0, 32.0, move_count=3)
    assert is_complete
    is_complete, reason = check_mission_complete(m, 30.0, 32.0, move_count=4)
    assert not is_complete

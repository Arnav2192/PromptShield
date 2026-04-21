"""Tests for StateManager."""
import threading
import pytest
from promptshield.state_manager import StateManager


@pytest.fixture
def sm():
    return StateManager(decay=0.6, threshold=1.5)


def test_initial_score_zero(sm):
    assert sm.get_score("sess-1") == 0.0


def test_update_returns_correct_score(sm):
    score = sm.update("sess-1", 1.0)
    assert score == pytest.approx(1.0)


def test_decay_applied(sm):
    sm.update("sess-1", 1.0)
    score = sm.update("sess-1", 0.5)
    # 1.0 * 0.6 + 0.5 = 1.1
    assert score == pytest.approx(1.1)


def test_threshold_flagging(sm):
    sm.update("sess-1", 1.0)
    sm.update("sess-1", 1.0)  # 1.0*0.6 + 1.0 = 1.6 >= 1.5
    assert sm.is_flagged("sess-1")


def test_not_flagged_below_threshold(sm):
    sm.update("sess-1", 0.5)
    assert not sm.is_flagged("sess-1")


def test_reset_clears_score(sm):
    sm.update("sess-1", 2.0)
    sm.reset("sess-1")
    assert sm.get_score("sess-1") == 0.0
    assert not sm.is_flagged("sess-1")


def test_separate_sessions_independent(sm):
    sm.update("sess-A", 2.0)
    assert sm.get_score("sess-B") == 0.0


def test_thread_safety():
    sm = StateManager(decay=1.0, threshold=9999999)
    results = []

    def worker():
        for _ in range(100):
            results.append(sm.update("shared", 1.0))

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 1000
    # With decay=1.0, each update adds 1.0; final score should equal 1000.0
    assert sm.get_score("shared") == pytest.approx(1000.0)

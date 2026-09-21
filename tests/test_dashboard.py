from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def run_app(policy: str, scenario: str = "Fixture laporan", seed: int = 0) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.selectbox[0].set_value(policy)
    at.sidebar.radio[0].set_value(scenario)
    at.sidebar.number_input[0].set_value(seed)
    at.sidebar.button[0].click().run()
    return at


def test_app_loads_without_episode():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("Jalankan episode" in i.value for i in at.info)


def test_fixture_rule_based_ends_without_consensus():
    at = run_app("Rule-based (meniru laporan)")
    assert not at.exception
    assert any("TANPA KONSENSUS setelah 6 putaran" in s.value for s in at.subheader)
    assert at.slider[0].max > 6


def test_stepping_through_the_episode():
    at = run_app("Rule-based (meniru laporan)")
    at.slider[0].set_value(5).run()             # round 1 finished: cash check step
    assert not at.exception
    at.button[1].click().run()                  # "Maju"
    assert at.session_state["step"] == 6 and not at.exception


def test_random_scenario_with_all_policies():
    for policy in ("Random", "IQL", "CTDE actor-critic", "Rencana optimal satu putaran"):
        at = run_app(policy, "Acak (pilih seed)", 3)
        assert not at.exception, policy

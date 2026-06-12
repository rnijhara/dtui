from dtui.providers.base import TrajectoryProvider
from dtui.providers.replay import ReplayProvider
from dtui.trajectory import StepRecord, Trajectory, write_jsonl


def _traj():
    return Trajectory(
        prompt="p",
        steps=[
            StepRecord(step=0, canvas=["?? ", "?? "]),
            StepRecord(step=1, canvas=["ok ", "?? "]),
            StepRecord(step=2, canvas=["ok ", "go "]),
        ],
    )


def test_replay_yields_all_steps():
    provider = ReplayProvider(_traj())
    assert isinstance(provider, TrajectoryProvider)
    assert provider.supports_trajectory is True
    steps = list(provider.stream_trajectory("ignored"))
    assert [s.step for s in steps] == [0, 1, 2]
    assert steps[-1].text() == "ok go "


def test_replay_from_file(tmp_path):
    path = tmp_path / "t.jsonl"
    write_jsonl(path, _traj().steps, prompt="p")
    provider = ReplayProvider.from_file(path)
    steps = list(provider.stream_trajectory("x"))
    assert len(steps) == 3

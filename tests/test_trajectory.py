from dtui.trajectory import StepRecord, Trajectory, read_jsonl, write_jsonl


def test_steprecord_roundtrip():
    rec = StepRecord(step=1, canvas=["a ", "b "], status=["masked", "revealed"], confidence=[0.0, 0.9])
    assert StepRecord.from_dict(rec.to_dict()) == rec
    assert rec.text() == "a b "


def test_steprecord_validates_lengths():
    import pytest

    with pytest.raises(ValueError):
        StepRecord(step=0, canvas=["a", "b"], status=["masked"])


def test_jsonl_roundtrip(tmp_path):
    steps = [
        StepRecord(step=0, canvas=["x ", "y "]),
        StepRecord(step=1, canvas=["hi ", "y "]),
    ]
    path = tmp_path / "t.jsonl"
    write_jsonl(path, steps, prompt="hello")
    traj = read_jsonl(path)
    assert isinstance(traj, Trajectory)
    assert traj.prompt == "hello"
    assert len(traj) == 2
    assert traj.final_text == "hi y "

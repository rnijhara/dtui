"""DiffusionGemma adapter capture logic, with a mocked transformers streamer.

No torch, no transformers, no 26B weights: we inject a stub ``transformers``
module so the capturing streamer subclasses a fake base, and drive its
``put``/``put_draft`` with fake token-id tensors.
"""

import queue
import sys
import types

from dtui.adapters.diffusion_gemma import decode_canvas, to_id_list


class _Tok:
    table = {10: "Diff", 11: "usion", 12: " models", 99: "<noise>"}

    def decode(self, ids):
        return self.table.get(ids[0], "?")


def test_to_id_list_flattens_2d():
    assert to_id_list([[1, 2, 3]]) == [1, 2, 3]
    assert to_id_list([4, 5]) == [4, 5]


def test_to_id_list_handles_tensor_like():
    class _Fake:
        def tolist(self):
            return [[7, 8]]

    assert to_id_list(_Fake()) == [7, 8]


def test_decode_canvas():
    assert decode_canvas([10, 11, 12], _Tok()) == ["Diff", "usion", " models"]


def test_capturing_streamer_records_each_step(monkeypatch):
    # Inject a stub transformers module before the factory imports it.
    fake = types.ModuleType("transformers")

    class _FakeBase:
        def __init__(self, tokenizer, **kw):
            self.tokenizer = tokenizer

        def put(self, value):  # pragma: no cover - overridden
            ...

        def end(self):  # pragma: no cover - overridden
            ...

    fake.TextDiffusionStreamer = _FakeBase
    monkeypatch.setitem(sys.modules, "transformers", fake)

    from dtui.adapters.diffusion_gemma import make_capturing_streamer_class

    cls = make_capturing_streamer_class()
    sink: "queue.Queue" = queue.Queue()
    streamer = cls(_Tok(), sink)

    # Draft canvases drive every per-step frame; the final draft is fully
    # resolved. put() receives a committed delta and must emit NO frame. end()
    # sends the sentinel.
    streamer.put_draft([[99, 99, 12]])   # step 0
    streamer.put_draft([[10, 99, 12]])   # step 1
    streamer.put_draft([[10, 11, 12]])   # step 2 (fully denoised)
    streamer.put([[10, 11, 12]])         # committed delta -> no frame
    streamer.end()

    records = []
    while True:
        r = sink.get()
        if r is None:
            break
        records.append(r)

    assert [r.step for r in records] == [0, 1, 2]
    assert records[0].canvas == ["<noise>", "<noise>", " models"]
    assert records[-1].text() == "Diffusion models"

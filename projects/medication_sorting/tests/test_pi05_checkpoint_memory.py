"""保存限流的生命周期测试；真实 Orbax 写读另用 CPU 验证。"""
from types import SimpleNamespace

import pytest

from tools.medication_sorting import pi05_adapter


@pytest.mark.parametrize("fail", [False, True])
def test_checkpoint_limits_are_local_and_restored_on_error(fail, monkeypatch):
    calls = []
    registry = object()
    monkeypatch.setattr(pi05_adapter, "make_streaming_registry", lambda *args: registry, raising=False)
    original_ocp = SimpleNamespace(PyTreeCheckpointHandler=lambda **kw: calls.append(kw))
    module = SimpleNamespace(ocp=original_ocp)

    def initialize(path, **kwargs):
        assert path == "run"
        module.ocp.PyTreeCheckpointHandler()
        module.ocp.PyTreeCheckpointHandler()
        if fail:
            raise RuntimeError("save setup failed")
        return "manager", kwargs["resume"]

    module.initialize_checkpoint_dir = initialize
    pi05_adapter.configure_checkpoint_memory(module)
    if fail:
        with pytest.raises(RuntimeError, match="save setup failed"):
            module.initialize_checkpoint_dir("run", resume=True)
    else:
        assert module.initialize_checkpoint_dir("run", resume=True) == ("manager", True)
    assert calls == [{"save_concurrent_gb": 3, "restore_concurrent_gb": 3,
                      "type_handler_registry": registry}] * 2
    assert module.ocp is original_ocp
    assert original_ocp.PyTreeCheckpointHandler is module.ocp.PyTreeCheckpointHandler


def test_device_arrays_are_committed_before_copying_the_next_leaf():
    import asyncio
    events = []

    def transfer(value, info, arg):
        events.append(("copy", value))
        return value * 2

    async def commit(host_value, info, arg):
        await asyncio.sleep(0)
        events.append(("commit", host_value))

    serialize = getattr(pi05_adapter, "serialize_arrays_sequentially", None)
    assert callable(serialize), "缺少逐数组复制并等待写盘的保存路径"
    asyncio.run(serialize([1, 2, 3], [None] * 3, [None] * 3,
                          transfer=transfer, commit=commit))
    assert events == [("copy", 1), ("commit", 2), ("copy", 2), ("commit", 4),
                      ("copy", 3), ("commit", 6)]


def test_write_failure_prevents_copying_more_arrays():
    import asyncio
    copied = []

    def transfer(value, info, arg):
        copied.append(value)
        return value

    async def fail(*args):
        raise OSError("disk write failed")

    serialize = getattr(pi05_adapter, "serialize_arrays_sequentially", None)
    assert callable(serialize), "缺少逐数组保存路径"
    with pytest.raises(OSError, match="disk write failed"):
        asyncio.run(serialize([1, 2], [None] * 2, [None] * 2,
                              transfer=transfer, commit=fail))
    assert copied == [1]

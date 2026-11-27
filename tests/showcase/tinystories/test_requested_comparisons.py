"""Explicit diagnostic requests fail when their prerequisites are unavailable."""
from types import SimpleNamespace
from opentsp_vllm.models.tinystories import driver, driver_gates, comparison

def test_missing_requested_checkpoint_is_not_a_success(monkeypatch,capsys):
    monkeypatch.setattr(comparison,"load_weights",lambda name:(None,"missing checkpoint"))
    assert driver.main(["--ids","0","--tokens","1","--weights","real","--checkpoint","/missing"])==1
    assert "missing checkpoint" in capsys.readouterr().out

def test_missing_requested_rounded_comparison_fails_the_gate(monkeypatch):
    monkeypatch.setattr(driver_gates,"check_rounded",lambda *a,**k:None)
    summary={"gates":{}}
    result=driver_gates.run(None,SimpleNamespace(weights="real"),[0],[0],{"rows":[{}],"hiddens":None,"kv":None},{"rounded"},summary,None,say=lambda *x:None)
    assert result==1
    assert summary["gates"]["rounded"]["ok"] is False

def test_fp32_request_with_synthetic_weights_fails_the_gate():
    summary={"gates":{}}
    result=driver_gates.run(None,SimpleNamespace(weights="synthetic"),[0],[0],{"rows":[{}],"hiddens":None},{"fp32"},summary,None,say=lambda *x:None)
    assert result==1
    assert summary["gates"]["fp32"]["ok"] is False

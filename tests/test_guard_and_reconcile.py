import subprocess
import sys
from pathlib import Path

import pytest

from llm_cost_meter import Ledger, MeterKind
from llm_cost_meter.guards.ast_guard import check_file, main
from llm_cost_meter.reconcile import Cause, reconcile

FIX = Path(__file__).parent / "fixtures"


def test_ast_guard_flags_direct_client():
    problems = check_file(FIX / "bad_module.py")
    assert any("provider client OpenAI" in p and ":5:" in p for p in problems)


def test_ast_guard_flags_unbucketed_label():
    problems = check_file(FIX / "bad_module.py")
    assert any("shiny_new_feature" in p and ":6:" in p for p in problems)


def test_ast_guard_passes_clean_module():
    assert check_file(FIX / "good_module.py") == []
    assert main([str(FIX / "good_module.py")]) == 0


def test_ast_guard_cli_exits_nonzero_on_bad_code():
    cmd = [sys.executable, "-m", "llm_cost_meter.guards.ast_guard", str(FIX / "bad_module.py")]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 1 and "bad_module.py:5" in proc.stdout


def _bill(ledger, service, micro, key):
    ledger.append(key, payer="p", tenant="t", request_id="r", service=service,
                  meter=MeterKind.TOKEN, units=1, micro=micro, kind="debit")


def test_reconcile_classifies_causes(prices):
    ledger = Ledger()
    # correctly billed, so it must not show up
    _bill(ledger, "chat", prices.cost_tokens("acme-chat", 1000, 100), "a")
    # billed at the input rate because the cached rate is missing; the provider charged less
    _bill(ledger, "planner", 3000, "b")
    # our price is wrong: provider cost differs from our table for the same usage
    _bill(ledger, "summarizer", prices.cost_tokens("acme-chat", 1000, 0), "c")
    # we billed the right unit price, but one call was never recorded
    _bill(ledger, "rerank", 100, "d")

    export = [
        dict(service="chat", model="acme-chat", input_tokens=1000, output_tokens=100, cached_tokens=0,
             cost_micro=prices.cost_tokens("acme-chat", 1000, 100)),
        dict(service="planner", model="no-cache-model", input_tokens=1000, output_tokens=0, cached_tokens=600,
             cost_micro=1380),
        dict(service="summarizer", model="acme-chat", input_tokens=1000, output_tokens=0, cached_tokens=0,
             cost_micro=3600),
        dict(service="voice", model="acme-chat", input_tokens=500, output_tokens=0, cached_tokens=0,
             cost_micro=1500),
        dict(service="rerank", model="reranker", meter="rerank_unit", input_tokens=0, output_tokens=0,
             cost_micro=300),
    ]
    causes = {m.service: m.cause for m in reconcile(ledger, prices, export)}
    assert causes == {
        "planner": Cause.MISSING_CACHED_RATE,
        "summarizer": Cause.PRICE_MISMATCH,
        "voice": Cause.UNMETERED_SERVICE,
        "rerank": Cause.UNKNOWN,
    }


BYPASSES = {
    "alias": "from langchain_openai import ChatOpenAI as C\nllm = C()\n",
    "partial": "import functools, openai\nmake = functools.partial(openai.OpenAI)\n",
    "reference": "from anthropic import AsyncAnthropic\nfactory = AsyncAnthropic\n",
    "getattr": "import openai\nllm = getattr(openai, 'OpenAI')()\n",
    "init_chat_model": "from langchain.chat_models import init_chat_model\nllm = init_chat_model('x')\n",
    "azure": "from langchain_openai import AzureChatOpenAI\nllm = AzureChatOpenAI()\n",
    "variable_label": "from llm_cost_meter import metered_chat_model\nlabel = 'chat'\nmetered_chat_model(label, 'm', llm_factory=f, meter=m)\n",
    "fstring_label": "from llm_cost_meter import metered_chat_model\nmetered_chat_model(f'chat{1}', 'm', llm_factory=f, meter=m)\n",
    "kwargs_label": "from llm_cost_meter import metered_chat_model\nmetered_chat_model(**opts)\n",
    "aliased_factory_bad_label": "from llm_cost_meter import metered_chat_model as g\ng('nope', 'm', llm_factory=f, meter=m)\n",
    "charge_units_label": "meter.charge_units(kind, 1, work_key='w', service='unregistered')\n",
    "charge_units_variable": "meter.charge_units(kind, 1, work_key='w', service=svc)\n",
    "syntax_error": "def broken(:\n",
}


@pytest.mark.parametrize("name", BYPASSES)
def test_ast_guard_catches_bypass(tmp_path, name):
    f = tmp_path / "feature.py"
    f.write_text(BYPASSES[name])
    assert check_file(f), f"{name} slipped through"
    assert main([str(f)]) == 1


def test_ast_guard_allow_list_is_by_package_path_not_basename(tmp_path):
    code = "from openai import OpenAI\nclient = OpenAI()\n"
    rogue = tmp_path / "providers.py"
    rogue.write_text(code)
    assert check_file(rogue)  # a providers.py anywhere else is not exempt
    ok = tmp_path / "llm_cost_meter"
    ok.mkdir()
    (ok / "__init__.py").write_text("")
    (ok / "providers.py").write_text(code)
    assert check_file(ok / "providers.py") == []


def test_ast_guard_accepts_registered_literal_labels(tmp_path):
    f = tmp_path / "ok.py"
    f.write_text("meter.charge_units(k, 1, work_key='w', service='rerank')\n"
                 "metered_chat_model('planner', 'm', llm_factory=f, meter=m)\n")
    assert check_file(f) == []


def test_reconcile_sums_rows_per_service_and_can_include_shadow(prices):
    ledger = Ledger()
    ledger.append("a", payer="p", tenant="t", request_id="r", service="chat", meter=MeterKind.TOKEN,
                  units=1, micro=300, kind="debit")
    ledger.append("b", payer="p", tenant="t", request_id="r", service="chat", meter=MeterKind.TOKEN,
                  units=1, micro=200, kind="debit", shadow=True)
    row = dict(model="acme-chat", input_tokens=0, output_tokens=0)
    two_calls = [dict(row, service="chat", cost_micro=100), dict(row, service="chat", cost_micro=200)]
    assert reconcile(ledger, prices, two_calls) == []  # 100 + 200 == 300 billed
    assert [m.service for m in reconcile(ledger, prices, two_calls, include_shadow=True)] == ["chat"]


def test_allow_list_needs_the_full_package_path(tmp_path):
    code = "from openai import OpenAI\nclient = OpenAI()\n"
    sub = tmp_path / "llm_cost_meter" / "sub"
    sub.mkdir(parents=True)
    (tmp_path / "llm_cost_meter" / "__init__.py").write_text("")
    (sub / "__init__.py").write_text("")
    (sub / "providers.py").write_text(code)
    assert check_file(sub / "providers.py")  # llm_cost_meter/sub/providers.py is not the allowed path


FALSE_POSITIVES = {
    "type_checking": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from openai import OpenAI\n"
                     "def f(c: 'OpenAI', d: OpenAI) -> OpenAI:\n    return d\nx: OpenAI | None = None\n",
    "isinstance": "from openai import OpenAI\ndef f(o):\n    return isinstance(o, OpenAI)\n",
    "enum_member": "import enum\nclass Vendor(enum.Enum):\n    OpenAI = 1\nv = Vendor.OpenAI\n",
    "forwarding_wrapper": "def build(service, meter):\n    return metered_chat_model(service, 'm', llm_factory=f, meter=meter)\n",
    "suppressed": "from openai import OpenAI\nclient = OpenAI()  # billing-guard: ok\n",
    "unrelated_module": "import mylib\nx = mylib.OpenAI\n",
}


@pytest.mark.parametrize("name", FALSE_POSITIVES)
def test_ast_guard_has_no_false_positive(tmp_path, name):
    f = tmp_path / "feature.py"
    f.write_text(FALSE_POSITIVES[name])
    assert check_file(f) == []


def test_suppression_is_per_line(tmp_path):
    f = tmp_path / "feature.py"
    f.write_text("from openai import OpenAI\na = OpenAI()  # billing-guard: ok\nb = OpenAI()\n")
    assert [p.split(":")[1] for p in check_file(f)] == ["3"]


def test_reconcile_covers_ledger_only_services_and_free_ones(prices):
    ledger = Ledger()
    _bill(ledger, "doc_parse", 500, "a")  # billed, provider shows nothing
    _bill(ledger, "chat", 100, "b")
    export = [{"service": "chat", "model": "acme-chat", "input_tokens": 0, "output_tokens": 0, "cost_micro": 100},
              {"service": "health_probe", "model": "acme-chat", "input_tokens": 9, "output_tokens": 0,
               "cost_micro": 40}]  # deliberately free to the user: not a mismatch
    assert {m.service: m.cause for m in reconcile(ledger, prices, export)} == {"doc_parse": Cause.NO_PROVIDER_SPEND}

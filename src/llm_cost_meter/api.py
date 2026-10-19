"""A minimal credits API: balance, usage, reconciliation, pricing history.

Identity comes from `payer_dependency`, never from a path or body parameter: a caller may only read its own
balance and usage. The default reads an `X-Payer` header as a stand-in for the payer your auth layer derives
from the token. Reconciliation and pricing history are operator views and need your own admin protection.
"""
from collections.abc import Callable
from dataclasses import asdict

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

from .meter import Meter
from .pricing_history import PriceHistory
from .reconcile import balance_drift, reconcile


def header_payer(x_payer: str | None = Header(default=None)) -> str:
    if not x_payer:
        raise HTTPException(status_code=401, detail="no payer identity on the request")
    return x_payer


class ReconcileRequest(BaseModel):
    export: list[dict]  # provider cost export rows, see reconcile()
    include_shadow: bool = False


def create_credits_router(meter: Meter, history: PriceHistory,
                          payer_dependency: Callable[..., str] = header_payer) -> APIRouter:
    router = APIRouter(prefix="/credits")

    @router.get("/balance")
    def balance(payer: str = Depends(payer_dependency)) -> dict:
        drift = balance_drift(meter.ledger, meter.balances, [payer]).get(payer, 0)
        return {"payer": payer, "balance_micro": meter.balances.get(payer),
                "ledger_micro": meter.ledger.balance_delta(payer), "drift_micro": drift}

    @router.get("/usage")
    def usage(request_id: str | None = None, payer: str = Depends(payer_dependency)) -> dict:
        rows = [dict(r) for r in meter.ledger.rows(request_id, payer=payer)]
        return {"payer": payer, "rows": rows, "total_micro": sum(r["micro_credits"] for r in rows)}

    @router.post("/reconciliation")
    def reconciliation(body: ReconcileRequest) -> dict:
        found = reconcile(meter.ledger, meter.prices, body.export, include_shadow=body.include_shadow)
        return {"mismatches": [{**asdict(m), "cause": m.cause.value} for m in found]}

    @router.get("/pricing-history")
    def pricing_history(model: str | None = None) -> dict:
        return {"changes": [{**asdict(c)} | {"rates": {k: str(v) for k, v in asdict(c.rates).items() if v is not None}}
                            for c in history.history(model)]}

    return router

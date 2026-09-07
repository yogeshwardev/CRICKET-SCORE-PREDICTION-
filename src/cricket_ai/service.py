"""Authenticated API, durable forecasts/outcomes, historical replay and monitoring."""
from __future__ import annotations
import hashlib
import json
import os
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, Header, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict, model_validator
from sqlalchemy import create_engine, String, Text, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from .features import build_features
from .models import predict
from .evaluation import regression, calibration_error


class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):
    __tablename__ = "predictions"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    match_id: Mapped[str] = mapped_column(String(100), index=True)
    version: Mapped[str] = mapped_column(String(100))
    timestamp: Mapped[str] = mapped_column(String(40))
    request_json: Mapped[str] = mapped_column(Text)
    prediction_json: Mapped[str] = mapped_column(Text)
    actual_json: Mapped[str | None] = mapped_column(Text, nullable=True)


class Delivery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    over_number: int = Field(ge=1, le=20)
    ball_number: int = Field(ge=1, le=100)
    batter: str = Field(min_length=1, max_length=100)
    non_striker: str = Field(min_length=1, max_length=100)
    bowler: str = Field(min_length=1, max_length=100)
    runs_batter: int = Field(ge=0, le=12)
    runs_extras: int = Field(ge=0, le=20)
    runs_total: int = Field(ge=0, le=32)
    wides: int = Field(default=0, ge=0, le=20)
    noballs: int = Field(default=0, ge=0, le=20)
    byes: int = Field(default=0, ge=0, le=20)
    legbyes: int = Field(default=0, ge=0, le=20)
    penalty: int = Field(default=0, ge=0, le=20)
    legal: Literal[0, 1]
    batter_ball: Literal[0, 1]
    wicket: int = Field(ge=0, le=2)
    bowler_wicket: int = Field(ge=0, le=1)
    boundary: Literal[0, 1]
    six: Literal[0, 1]
    bowler_conceded: int = Field(ge=0, le=32)

    @model_validator(mode="after")
    def accounting(self):
        if self.runs_total != self.runs_batter+self.runs_extras:
            raise ValueError("Inconsistent total")
        if self.runs_extras != self.wides+self.noballs+self.byes+self.legbyes+self.penalty:
            raise ValueError("Inconsistent extras")
        if self.legal != int(not(self.wides or self.noballs)) or self.batter_ball != int(not self.wides):
            raise ValueError("Invalid ball accounting")
        if self.bowler_conceded != self.runs_batter+self.wides+self.noballs:
            raise ValueError("Invalid bowler-conceded total")
        if self.six > self.boundary or (self.six and self.runs_batter != 6) or (self.boundary and self.runs_batter not in (4, 6)):
            raise ValueError("Invalid boundary flag")
        if self.bowler_wicket > self.wicket:
            raise ValueError("Invalid wicket accounting")
        return self


class MatchState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    match_id: str = Field(min_length=1, max_length=100)
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    innings: Literal[1, 2]
    over: int = Field(ge=1, le=20)
    score: int = Field(ge=0, le=500)
    wickets: int = Field(ge=0, le=9)
    striker: str = Field(min_length=1, max_length=100)
    non_striker: str = Field(min_length=1, max_length=100)
    bowler: str = Field(min_length=1, max_length=100)
    venue: str = Field(min_length=1, max_length=200)
    team_batting: str = Field(min_length=1, max_length=100)
    team_bowling: str = Field(min_length=1, max_length=100)
    competition: Literal["IPL"] = "IPL"
    target: int = Field(default=0, ge=0, le=501)
    scheduled_overs: Literal[20] = 20
    revised_target: Literal[False] = False
    bowler_confirmed: Literal[True]
    previous_deliveries: list[Delivery] = Field(max_length=400)

    @model_validator(mode="after")
    def boundary_state(self):
        datetime.strptime(self.date, "%Y-%m-%d")
        if self.striker == self.non_striker or self.team_batting == self.team_bowling:
            raise ValueError("Distinct batters and teams required")
        if (self.innings == 1 and self.target != 0) or (self.innings == 2 and self.target <= self.score):
            raise ValueError("Invalid chase target")
        indices = [(d.over_number, d.ball_number) for d in self.previous_deliveries]
        if indices != sorted(set(indices)):
            raise ValueError("Deliveries must be unique and in order")
        for o in range(1, self.over):
            deliveries = [d for d in self.previous_deliveries if d.over_number == o]
            if sum(d.legal for d in deliveries) != 6 or [d.ball_number for d in deliveries] != list(range(1, len(deliveries)+1)):
                raise ValueError("Each preceding over must contain six legal balls and contiguous events")
        if any(d.over_number >= self.over for d in self.previous_deliveries):
            raise ValueError("Target-over delivery leakage")
        return self

    def features(self, history):
        s = self.model_dump()
        s.update(batter=self.striker, over_number=self.over, current_score=self.score,
                 current_wickets=self.wickets, season=int(self.date[:4]))
        return build_features(s, [d.model_dump() for d in self.previous_deliveries], history)


class Outcome(BaseModel):
    model_config = ConfigDict(extra="forbid")
    runs: int = Field(ge=0, le=100)
    wicket: Literal[0, 1]
    boundary: Literal[0, 1]
    six: Literal[0, 1]


def create_app(root: Path | None = None):
    root = root or Path(os.getenv("CREASE_ROOT", ".")).resolve()
    database = os.getenv("DATABASE_URL", "sqlite:///"+str(root / "crease.db"))
    engine = create_engine(database, pool_pre_ping=True)
    runtime = {"bundle": None, "history": None, "report": None}
    @asynccontextmanager
    async def lifespan(app):
        Base.metadata.create_all(engine)
        pointer = root / "models/active.json"
        if pointer.exists():
            version = json.loads(pointer.read_text())["version"]
            folder = (root / "models" / version).resolve()
            if folder.parent != (root / "models").resolve():
                raise RuntimeError("Invalid model pointer")
            runtime["bundle"] = joblib.load(folder / "bundle.joblib")
            runtime["report"] = json.loads((folder / "report.json").read_text())
            runtime["history"] = joblib.load(root / "data/processed/history.joblib")
        yield
        engine.dispose()
    app = FastAPI(title="Crease AI", version="0.1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(","),
                       allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"])

    def authorized(authorization: str | None = Header(default=None)):
        token = os.getenv("CREASE_API_KEY")
        if not token:
            raise HTTPException(503, "Set CREASE_API_KEY to enable protected endpoints")
        if not authorization or not secrets.compare_digest(authorization, "Bearer "+token):
            raise HTTPException(401, "Invalid bearer token")

    def ready():
        if runtime["bundle"] is None:
            raise HTTPException(503, "No promoted model. Train, inspect report, then promote a qualifying version.")
        return runtime["bundle"]

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": runtime["bundle"] is not None,
                "live_provider_configured": bool(os.getenv("LIVE_PROVIDER_URL"))}

    @app.get("/model", dependencies=[Depends(authorized)])
    def model_report():
        ready()
        return runtime["report"]

    @app.post("/predict/next-over", dependencies=[Depends(authorized)])
    def next_over(state: MatchState):
        bundle = ready()
        # Loaded historical snapshot is only valid for later match dates.
        try:
            f = state.features(runtime["history"])
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        payload = state.model_dump_json()
        identity = hashlib.sha256((bundle["version"]+payload).encode()).hexdigest()
        with Session(engine) as session:
            existing = session.get(PredictionRecord, identity)
            if existing:
                return json.loads(existing.prediction_json)
            result = predict(bundle, f)
            result.update(prediction_id=identity, mode="live_state", match_id=state.match_id)
            session.add(PredictionRecord(id=identity, match_id=state.match_id, version=bundle["version"],
                                         timestamp=datetime.now(timezone.utc).isoformat(), request_json=payload,
                                         prediction_json=json.dumps(result)))
            try:
                session.commit()
            except Exception:
                session.rollback()
                existing = session.get(PredictionRecord, identity)
                if existing:
                    return json.loads(existing.prediction_json)
                raise
        return result

    @app.post("/predictions/{prediction_id}/actual", dependencies=[Depends(authorized)])
    def outcome(prediction_id: str, actual: Outcome):
        with Session(engine) as session:
            record = session.get(PredictionRecord, prediction_id)
            if record is None:
                raise HTTPException(404, "Unknown prediction")
            value = actual.model_dump_json()
            if record.actual_json is not None and record.actual_json != value:
                raise HTTPException(409, "Outcome already recorded; corrections need an audited reconciliation")
            record.actual_json = value
            session.commit()
        return {"recorded": True}

    @app.get("/monitoring", dependencies=[Depends(authorized)])
    def monitoring():
        with Session(engine) as session:
            records = session.scalars(select(PredictionRecord).order_by(PredictionRecord.timestamp.desc()).limit(1000)).all()
            completed = [r for r in records if r.actual_json is not None]
            output = {"stored": len(records), "scored": len(completed), "status": "awaiting_outcomes", "metrics": None}
            if completed:
                # Never pool different model versions into one metric.
                output["by_version"] = {}
                for version in sorted(set(r.version for r in completed)):
                    group = [r for r in completed if r.version == version]
                    p = [json.loads(r.prediction_json) for r in group]
                    a = [json.loads(r.actual_json) for r in group]
                    measured = regression([r["runs"] for r in a], np.array([r["expected_runs"] for r in p])) if len(a) > 1 else {"mae": abs(a[0]["runs"]-p[0]["expected_runs"])}
                    measured.update(n=len(a), interval_coverage=float(np.mean([r["lower_80"] <= v["runs"] <= r["upper_80"] for r, v in zip(p, a)])),
                                    wicket_brier=float(np.mean([(r["wicket_probability"]-v["wicket"])**2 for r, v in zip(p, a)])))
                    output["by_version"][version] = measured
                output["status"] = "measured"
            return output

    @app.get("/replay/states", dependencies=[Depends(authorized)])
    def replay_states():
        bundle = ready()
        frame = pd.read_parquet(root / "data/processed/overs.parquet")
        year = int(runtime["report"]["partitions"]["test"]["start"][:4])
        frame = frame[frame.season == year].tail(80)
        return [{"id": int(i), "label": f"{r.date} · {r.team_batting} · innings {r.innings}, over {r.over_number}"} for i, r in frame.iterrows()]

    @app.get("/replay/{row_id}", dependencies=[Depends(authorized)])
    def replay(row_id: int):
        bundle = ready()
        frame = pd.read_parquet(root / "data/processed/overs.parquet")
        if row_id not in frame.index or str(frame.loc[row_id, "date"]) < runtime["report"]["partitions"]["test"]["start"]:
            raise HTTPException(404, "Unknown test replay state")
        row = frame.loc[row_id].to_dict()
        result = predict(bundle, row)
        result.update(mode="historical_replay", state={k: row[k] for k in ["match_id", "date", "innings", "over_number", "current_score", "current_wickets", "batter", "non_striker", "bowler", "venue", "team_batting", "team_bowling"]},
                      actual_runs=int(row["target_next_over_runs"]))
        return result

    @app.websocket("/ws/status")
    async def websocket(ws: WebSocket):
        import asyncio
        await ws.accept()
        try:
            token = await asyncio.wait_for(ws.receive_text(), timeout=5)
            if not os.getenv("CREASE_API_KEY") or not secrets.compare_digest(token, os.environ["CREASE_API_KEY"]):
                await ws.close(code=1008)
                return
            while True:
                await ws.send_json(health())
                await asyncio.sleep(10)
        except (WebSocketDisconnect, TimeoutError):
            pass
    return app


app = create_app()

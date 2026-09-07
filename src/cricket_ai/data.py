"""Immutable Cricsheet acquisition and lossless delivery normalization."""
from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path
import pandas as pd

URL = "https://cricsheet.org/downloads/ipl_json.zip"
NOT_WICKETS = {"retired hurt", "retired not out"}
NOT_BOWLER = NOT_WICKETS | {"run out", "retired out", "obstructing the field"}
TEAM_ALIASES = {"Royal Challengers Bangalore": "Royal Challengers Bengaluru",
                "Delhi Daredevils": "Delhi Capitals", "Kings XI Punjab": "Punjab Kings"}
VENUE_ALIASES_FILE = Path(__file__).resolve().parents[2] / "configs/venue_aliases.csv"


def venue_aliases() -> dict[str, str]:
    """Curated, auditable ground renames that plain normalization cannot infer."""
    if not VENUE_ALIASES_FILE.exists():
        return {}
    table = pd.read_csv(VENUE_ALIASES_FILE)
    return dict(zip(table.alias, table.canonical))


def canonical_venue(name: str, aliases: dict[str, str]) -> str:
    """Collapse locality suffixes and punctuation variants onto one ground identity.

    Cricsheet records the same ground as "Wankhede Stadium", "Wankhede Stadium, Mumbai"
    and "M.Chinnaswamy Stadium" / "M Chinnaswamy Stadium". Splitting history across those
    keys weakens venue features and invents cold starts for grounds with decades of play.
    Only the leading ground name is kept; renames come from the reviewed alias file.
    """
    head = " ".join(name.split(",")[0].replace(".", " ").split())
    return aliases.get(head, head)


def acquire(root: Path) -> Path:
    directory = root / "data/raw"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "ipl_json.zip"
    if not target.exists():
        temporary = target.with_suffix(".download")
        urllib.request.urlretrieve(URL, temporary)
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip():
                raise ValueError("Corrupt archive")
        temporary.replace(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest = directory / f"manifest-{digest[:12]}.json"
    if not manifest.exists():
        manifest.write_text(json.dumps({"source": URL, "sha256": digest,
                                       "bytes": target.stat().st_size}, indent=2))
    return target


def normalize_match(match_id: str, document: dict, aliases: dict[str, str] | None = None) -> tuple[list[dict], str | None, dict]:
    aliases = venue_aliases() if aliases is None else aliases
    info = document["info"]
    people = info.get("registry", {}).get("people", {})
    # No post-match outcome is ever a feature. These exclusions define the population.
    reason = None
    if info.get("match_type") != "T20" or info.get("balls_per_over", 6) != 6:
        reason = "unsupported_format"
    elif info.get("outcome", {}).get("method"):
        reason = "revised_target_or_rain_method"
    elif info.get("outcome", {}).get("result") == "no result":
        reason = "no_result"
    elif info.get("missing"):
        reason = "missing_data"
    elif info.get("overs", 20) != 20:
        reason = "shortened_match"
    elif any(i.get("penalty_runs") or i.get("miscounted_overs") or
             i.get("target", {}).get("overs", 20) != 20 for i in document["innings"] if not i.get("super_over")):
        reason = "shortened_or_ambiguous_innings"
    if reason:
        return [], reason, people
    def identity(name):
        if name not in people:
            raise ValueError(f"Unregistered player {name} in {match_id}")
        return people[name]
    rows = []
    first_total = 0
    for inning_no, inning in enumerate(document["innings"], 1):
        if inning.get("super_over"):
            continue
        team = inning["team"]
        opponent = next(t for t in info["teams"] if t != team)
        score = wickets = legal_count = 0
        target = first_total + 1 if inning_no == 2 else 0
        if inning.get("target", {}).get("runs", target) != target:
            return [], "target_discrepancy", people
        for expected, over in enumerate(inning["overs"]):
            if over["over"] != expected:
                return [], "non_contiguous_overs", people
            for ball_index, delivery in enumerate(over["deliveries"], 1):
                extras = delivery.get("extras", {})
                dismissals = delivery.get("wickets", [])
                legal = not (extras.get("wides", 0) or extras.get("noballs", 0))
                count_wickets = sum(w["kind"] not in NOT_WICKETS for w in dismissals)
                r = delivery["runs"]
                if r["total"] != r["batter"] + r["extras"] or r["extras"] != sum(extras.values()):
                    raise ValueError(f"Inconsistent run accounting: {match_id}")
                rows.append(dict(match_id=match_id, date=str(info["dates"][0]),
                    season=int(str(info["dates"][0])[:4]), competition="IPL", venue=canonical_venue(info["venue"], aliases),
                    city=info.get("city", "unknown"), team_batting=TEAM_ALIASES.get(team, team),
                    team_bowling=TEAM_ALIASES.get(opponent, opponent), innings=inning_no,
                    over_number=over["over"] + 1, ball_number=ball_index,
                    batter=identity(delivery["batter"]), non_striker=identity(delivery["non_striker"]),
                    bowler=identity(delivery["bowler"]), runs_batter=r["batter"], runs_extras=r["extras"],
                    runs_total=r["total"], extra_type="|".join(sorted(extras)),
                    wides=extras.get("wides", 0), noballs=extras.get("noballs", 0),
                    byes=extras.get("byes", 0), legbyes=extras.get("legbyes", 0),
                    penalty=extras.get("penalty", 0), legal=int(legal),
                    batter_ball=int(not extras.get("wides", 0)), wicket=count_wickets,
                    bowler_wicket=sum(w["kind"] not in NOT_BOWLER for w in dismissals),
                    dismissed_player="|".join(identity(w["player_out"]) for w in dismissals if w["kind"] not in NOT_WICKETS),
                    dismissal_type="|".join(w["kind"] for w in dismissals),
                    boundary=int(r["batter"] in (4, 6) and not r.get("non_boundary", False)),
                    six=int(r["batter"] == 6 and not r.get("non_boundary", False)),
                    bowler_conceded=r["batter"] + extras.get("wides", 0) + extras.get("noballs", 0),
                    target=target, current_score=score, current_wickets=wickets,
                    legal_balls_before=legal_count))
                score += r["total"]
                wickets += count_wickets
                legal_count += legal
        if inning_no == 1:
            first_total = score
    return rows, None, people


def prepare(root: Path) -> pd.DataFrame:
    archive = acquire(root)
    rows, exclusions, mapping, hashes = [], [], set(), set()
    aliases = venue_aliases()
    with zipfile.ZipFile(archive) as z:
        for name in sorted(z.namelist()):
            if not name.endswith(".json"):
                continue
            payload = z.read(name)
            document = json.loads(payload)
            # Ignore metadata differences when detecting duplicate match bodies.
            fingerprint = hashlib.sha256(json.dumps({"info": document["info"], "innings": document["innings"]}, sort_keys=True).encode()).hexdigest()
            if fingerprint in hashes:
                exclusions.append({"match_id": Path(name).stem, "reason": "duplicate"})
                continue
            hashes.add(fingerprint)
            parsed, reason, people = normalize_match(Path(name).stem, document, aliases)
            for alias, canonical in people.items():
                # Different people can share a name (e.g. Harmeet Singh).
                # Identity always comes from the match registry, never alias alone.
                mapping.add((alias, canonical))
            if reason:
                exclusions.append({"match_id": Path(name).stem, "reason": reason})
            rows.extend(parsed)
    frame = pd.DataFrame(rows).sort_values(["date", "match_id", "innings", "over_number", "ball_number"])
    destination = root / "data/interim"
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination / "deliveries.parquet", index=False)
    pd.DataFrame(exclusions).to_csv(destination / "exclusions.csv", index=False)
    pd.DataFrame([{"alias": k, "player_id": v} for k, v in sorted(mapping)]).to_csv(root / "data/player_mapping.csv", index=False)
    return frame

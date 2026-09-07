import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="crease")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    training = sub.add_parser("train")
    training.add_argument("--trials", type=int, default=6)
    training.add_argument("--iterations", type=int, default=400)
    promotion = sub.add_parser("promote")
    promotion.add_argument("version")
    sub.add_parser("serve")
    sub.add_parser("report")
    live = sub.add_parser("ingest")
    live.add_argument("match_id")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "prepare":
        from .data import prepare
        from .features import make_samples
        deliveries = prepare(root)
        print(f"Normalized {len(deliveries):,} deliveries from {deliveries.match_id.nunique()} matches", flush=True)
        frame = make_samples(deliveries, root)
        print(f"Built {len(frame):,} over samples, {frame.date.min()} to {frame.date.max()}", flush=True)
    elif args.command == "train":
        from .models import train
        report = train(root, args.trials, args.iterations)
        print(json.dumps({k: report[k] for k in ["version", "test_regression", "uncertainty", "promotion_eligible"]}, indent=2))
    elif args.command == "promote":
        folder = (root / "models" / args.version).resolve()
        if folder.parent != (root / "models").resolve():
            raise ValueError("Invalid model version")
        report = json.loads((folder / "report.json").read_text())
        if not report["promotion_eligible"]:
            raise ValueError("Model did not beat every baseline on validation; promotion refused")
        temporary = root / "models/active.tmp"
        temporary.write_text(json.dumps({"version": args.version}))
        temporary.replace(root / "models/active.json")
        print("Promoted", args.version, "— restart API to load")
    elif args.command == "serve":
        import uvicorn
        os.environ["CREASE_ROOT"] = str(root)
        uvicorn.run("cricket_ai.service:app", host="127.0.0.1", port=8000)
    elif args.command == "report":
        print((root / "reports/latest.json").read_text())
    elif args.command == "ingest":
        import httpx
        from .live import LiveAdapter
        state = LiveAdapter().fetch(args.match_id)
        with httpx.Client(timeout=30) as client:
            response = client.post(os.getenv("CREASE_API_URL", "http://127.0.0.1:8000")+"/predict/next-over",
                                   json=state.model_dump(), headers={"Authorization": "Bearer "+os.environ["CREASE_API_KEY"]})
            response.raise_for_status()
            print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()

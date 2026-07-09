"""CLI: generate HTML report or serve it on localhost."""

from __future__ import annotations

import argparse
import socket
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dashboard.report import ARTIFACT_ROOT, build_report, discover_experiments

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "dashboard" / "index.html"


def pick_port(preferred: int = 0) -> int:
    if preferred > 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", preferred)) != 0:
                return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def resolve_experiment(name: str | None) -> Path:
    experiments = discover_experiments(PROJECT_ROOT / ARTIFACT_ROOT)
    if not experiments:
        raise SystemExit(
            "No experiment artifacts found. Run:\n"
            "  pip install -e .\n"
            "  python -m alpha_pipeline.cli --output artifacts/demo"
        )
    if name:
        matches = [path for path in experiments if path.name == name]
        if not matches:
            available = ", ".join(path.name for path in experiments)
            raise SystemExit(f"Experiment '{name}' not found. Available: {available}")
        return matches[0]
    demo = [path for path in experiments if path.name == "demo"]
    return demo[0] if demo else experiments[0]


def cmd_build(args: argparse.Namespace) -> None:
    experiment = resolve_experiment(args.experiment)
    output = Path(args.output) if args.output else DEFAULT_OUTPUT
    build_report(experiment, output)
    print(f"Wrote {output.resolve()}")


def cmd_serve(args: argparse.Namespace) -> None:
    experiment = resolve_experiment(args.experiment)
    output = Path(args.output) if args.output else DEFAULT_OUTPUT
    build_report(experiment, output)
    port = pick_port(args.port)
    serve_dir = output.parent.resolve()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *handler_args, **kwargs):
            super().__init__(*handler_args, directory=str(serve_dir), **kwargs)

    url = f"http://127.0.0.1:{port}/{output.name}"
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Serving {output.name} from {serve_dir}")
    print(f"Open {url}")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpha research HTML dashboard")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Generate a standalone HTML report")
    build.add_argument("--experiment", default=None, help="Experiment folder name under artifacts/")
    build.add_argument("--output", default=None, help="Output HTML path")
    build.set_defaults(func=cmd_build)

    serve = sub.add_parser("serve", help="Generate HTML and serve on localhost")
    serve.add_argument("--experiment", default=None, help="Experiment folder name under artifacts/")
    serve.add_argument("--output", default=None, help="HTML path to generate and serve")
    serve.add_argument("--port", type=int, default=8765, help="Preferred port (falls back if busy)")
    serve.add_argument("--no-open", dest="open", action="store_false", help="Do not open a browser tab")
    serve.set_defaults(open=True, func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
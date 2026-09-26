"""The ``subbridge`` command: doctor, serve, and run."""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager, suppress

from .doctor import print_report
from .gateway import Gateway, serve

TAGLINE = (
    "Prototype on the subscription your team already has, ship with a real API key."
)

_POSIX = os.name == "posix"

_RUN_EXIT_CODES = (
    "Exit codes: the command's own exit code; 128+N if it dies from signal N; "
    "126 if it cannot be executed; 127 if it is not found; 1 if the port is "
    "already in use; 2 for a usage error."
)

_PROXY_VARIABLES = ("http_proxy", "https_proxy", "all_proxy")
_PROXY_NOTE = (
    "If HTTP_PROXY, HTTPS_PROXY, or ALL_PROXY is set, add 127.0.0.1 to "
    "NO_PROXY too, or the SDKs may send the gateway key and your prompts to "
    "that proxy instead of the gateway."
)


def _proxy_capture_warning() -> str | None:
    """A one-line warning when a proxy could intercept the gateway's own traffic.

    The official SDKs treat the gateway's `127.0.0.1` base URL like any other
    HTTP endpoint, so `HTTP_PROXY`, `HTTPS_PROXY`, or `ALL_PROXY` (either
    case) still applies to it unless `NO_PROXY`/`no_proxy` exempts
    `127.0.0.1`.
    """
    active = any(
        os.environ.get(name.upper()) or os.environ.get(name)
        for name in _PROXY_VARIABLES
    )
    if not active:
        return None
    no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    exempted = {entry.strip() for entry in no_proxy.split(",")}
    if "127.0.0.1" in exempted:
        return None
    return f"subbridge: {_PROXY_NOTE}"


def _warn_about_proxy_capture() -> None:
    warning = _proxy_capture_warning()
    if warning:
        print(warning, file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    return args.handler(args)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="subbridge", description=TAGLINE)
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    doctor = commands.add_parser(
        "doctor",
        help="Check the local CLIs and sign-in without sending a prompt.",
        description="Check local provider CLI, login, model catalog, and plan/usage visibility.",
    )
    doctor.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON."
    )
    doctor.set_defaults(handler=lambda args: print_report(args.json))
    serve_cmd = commands.add_parser(
        "serve",
        help="Run the localhost gateway for the official Anthropic and OpenAI SDKs.",
        description=TAGLINE,
    )
    _add_port(serve_cmd)
    serve_cmd.set_defaults(handler=_serve)
    run = commands.add_parser(
        "run",
        help="Run a command with the gateway's base URLs and key in its environment.",
        description=f"{TAGLINE} Example: subbridge run -- python app.py",
        epilog=f"{_RUN_EXIT_CODES}\n\n{_PROXY_NOTE}",
    )
    _add_port(run)
    run.add_argument("argv", nargs=argparse.REMAINDER, help="The command, after --.")
    run.set_defaults(handler=_run)
    return parser


def _add_port(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--port", type=_port, default=0, help="Port on 127.0.0.1 (default: a free one)."
    )


def _port(value: str) -> int:
    """An argparse ``type=`` that turns an out-of-range port into a usage error."""
    try:
        port = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid port: {value!r}") from None
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError(
            f"port must be between 0 and 65535, got {port}"
        )
    return port


def client_environment(gateway: Gateway) -> dict[str, str]:
    """The four variables the official SDKs read for their endpoint and key."""
    return {
        "ANTHROPIC_BASE_URL": gateway.anthropic_base_url,
        "ANTHROPIC_API_KEY": gateway.api_key,
        "OPENAI_BASE_URL": gateway.openai_base_url,
        "OPENAI_API_KEY": gateway.api_key,
    }


def _start(port: int) -> Gateway | None:
    try:
        return serve(port=port)
    except OSError as exc:
        reason = exc.strerror or exc
        print(
            f"subbridge: cannot listen on 127.0.0.1:{port}: {reason}", file=sys.stderr
        )
        return None


def _serve(args: argparse.Namespace) -> int:
    _warn_about_proxy_capture()
    gateway = _start(args.port)
    if gateway is None:
        return 1
    with gateway:
        print(_banner(gateway), flush=True)
        _wait_for_interrupt()
    return 0


def _banner(gateway: Gateway) -> str:
    exports = "\n".join(
        f"export {name}={shlex.quote(value)}"
        for name, value in client_environment(gateway).items()
    )
    return (
        f"SubBridge gateway on {gateway.anthropic_base_url} (this machine only)\n"
        f"  Anthropic base URL: {gateway.anthropic_base_url}\n"
        f"  OpenAI base URL:    {gateway.openai_base_url}\n"
        f"  Key:                {gateway.api_key}\n\n"
        f"{exports}\n\n"
        "Each request runs your signed-in CLI, within what your plan and your "
        "organization's policy allow.\nPress Ctrl+C to stop."
    )


def _wait_for_interrupt() -> None:
    with suppress(KeyboardInterrupt):
        while True:
            time.sleep(3600)


def _run(args: argparse.Namespace) -> int:
    command = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
    if not command:
        print("subbridge run: give a command after --", file=sys.stderr)
        return 2
    _warn_about_proxy_capture()
    gateway = _start(args.port)
    if gateway is None:
        return 1
    try:
        with gateway:
            env = {**os.environ, **client_environment(gateway)}
            try:
                proc = subprocess.Popen(command, env=env)
            except FileNotFoundError:
                print(
                    f"subbridge run: command not found: {command[0]}", file=sys.stderr
                )
                return 127
            except PermissionError:
                print(
                    f"subbridge run: permission denied: {command[0]}", file=sys.stderr
                )
                return 126
            with _yield_interrupts_to_child(proc):
                code = proc.wait()
    except KeyboardInterrupt:
        return 130
    return code if code >= 0 else 128 - code


@contextmanager
def _yield_interrupts_to_child(proc: subprocess.Popen) -> Generator[None, None, None]:
    """While `proc` runs, let it handle Ctrl+C itself instead of us killing it.

    A terminal's Ctrl+C reaches `proc` directly too, since it shares our
    process group by default, so we ignore SIGINT ourselves here and just
    wait for `proc` to act on its own copy and exit, then return its real
    exit code. On POSIX, also forward SIGTERM to `proc`, so `kill` on this
    process (from another terminal, say) doesn't orphan `proc` or skip the
    gateway's own cleanup in `with gateway:`.
    """
    previous_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
    previous_term = None
    if _POSIX:
        previous_term = signal.signal(
            signal.SIGTERM, lambda signum, frame: proc.send_signal(signal.SIGTERM)
        )
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous_int)
        if _POSIX:
            signal.signal(signal.SIGTERM, previous_term)


if __name__ == "__main__":
    raise SystemExit(main())

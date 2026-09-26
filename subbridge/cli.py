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
from .gateway import Gateway, client_environment, proxy_capture_warning, serve
from .gateway._process_env import PROXY_NOTE

TAGLINE = (
    "Prototype on the subscription your team already has, ship with a real API key."
)

_POSIX = os.name == "posix"

_RUN_EXIT_CODES = (
    "Exit codes: the command's own exit code; 128+N if it dies from signal N; "
    "126 if it cannot be executed; 127 if it is not found; 1 if the port is "
    "already in use; 2 for a usage error."
)


def _warn_about_proxy_capture() -> None:
    warning = proxy_capture_warning()
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
        epilog=f"{_RUN_EXIT_CODES}\n\n{PROXY_NOTE}",
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
            with _yield_interrupts_to_child() as child:
                try:
                    child.proc = subprocess.Popen(command, env=env)
                except FileNotFoundError:
                    print(
                        f"subbridge run: command not found: {command[0]}",
                        file=sys.stderr,
                    )
                    return 127
                except PermissionError:
                    print(
                        f"subbridge run: permission denied: {command[0]}",
                        file=sys.stderr,
                    )
                    return 126
                if child.pending_term:
                    # A SIGTERM arrived before the child existed to forward
                    # it to; catch up on it now instead of leaving it
                    # dropped and `wait()` below stuck until Ctrl+C.
                    child.proc.send_signal(signal.SIGTERM)
                code = child.proc.wait()
    except KeyboardInterrupt:
        return 130
    return code if code >= 0 else 128 - code


class _ChildSlot:
    """Holds the child `Popen` once `Popen()` returns it.

    Starts empty (`proc` is `None`) so `_yield_interrupts_to_child()` can
    install its SIGTERM handler *before* the child exists, and the handler
    still has something to check. `pending_term` lets a SIGTERM that
    arrives in that gap be forwarded once the child does exist, instead of
    being silently dropped.
    """

    proc: subprocess.Popen | None = None
    pending_term: bool = False


@contextmanager
def _yield_interrupts_to_child() -> Generator[_ChildSlot, None, None]:
    """Install signal handling before spawning the child, not just around its wait.

    Installed before `subprocess.Popen()` even runs -- not only around
    `proc.wait()` -- so a SIGTERM or SIGINT arriving in the gap between
    deciding to run the child and `Popen()` actually returning cannot kill
    this process outright: that would orphan the child once it exists and
    skip the gateway's own cleanup in `with gateway:`. The caller sets
    `.proc` on the yielded slot as soon as `Popen()` succeeds (and, if
    `.pending_term` was set in the meantime, forwards the signal itself
    then); the SIGTERM handler below forwards it directly once `.proc` is
    already there, and just records it as pending otherwise.

    A terminal's Ctrl+C reaches the child directly too, since it shares our
    process group by default, so we install a no-op SIGINT handler of our
    own (not `signal.SIG_IGN`: that disposition, unlike a Python-level
    handler, survives `exec()` -- so a child spawned while it's in place
    would inherit SIGINT *ignored* outright, rather than the default
    behavior a child with no handler of its own is expected to have) and
    just wait for the child to act on its own copy and exit, then return
    its real exit code. On POSIX, also forward SIGTERM to the child once it
    exists, so `kill` on this process (from another terminal, say) doesn't
    orphan it or skip the gateway's own cleanup.
    """
    slot = _ChildSlot()
    previous_int = signal.signal(signal.SIGINT, lambda signum, frame: None)
    previous_term = None
    if _POSIX:

        def _forward_sigterm(signum, frame):
            if slot.proc is not None:
                slot.proc.send_signal(signal.SIGTERM)
            else:
                slot.pending_term = True

        previous_term = signal.signal(signal.SIGTERM, _forward_sigterm)
    try:
        yield slot
    finally:
        signal.signal(signal.SIGINT, previous_int)
        if _POSIX:
            signal.signal(signal.SIGTERM, previous_term)


if __name__ == "__main__":
    raise SystemExit(main())

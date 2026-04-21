#!/usr/bin/env python3
"""
PromptShield unified launcher.

Starts all necessary components (API server + frontend dev server) and
reports their live status in the terminal.

Usage:
    python start.py [--no-frontend] [--port-api PORT] [--port-ui PORT]

Flags:
    --no-frontend       Skip the Vite frontend (useful in headless/CI envs)
    --port-api PORT     API server port (default: 8001)
    --port-ui  PORT     Frontend dev-server port (default: 8080)
"""
from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import signal
import subprocess
import sys
import textwrap
import threading
import time
import urllib.request

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Start all PromptShield components.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=textwrap.dedent(__doc__ or ""),
)
parser.add_argument("--no-frontend", action="store_true", help="Skip the Vite frontend")
parser.add_argument("--port-api", type=int, default=8001, metavar="PORT", help="API port (default: 8001)")
parser.add_argument("--port-ui", type=int, default=8080, metavar="PORT", help="Frontend port (default: 8080)")
args = parser.parse_args()

PORT_API: int = args.port_api
PORT_UI: int = args.port_ui
RUN_FRONTEND: bool = not args.no_frontend

ROOT = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(ROOT, "frontend")

# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
BOLD = "\033[1m"
RESET = "\033[0m"

_ANSI_OK = f"{GREEN}✅ UP  {RESET}"
_ANSI_DOWN = f"{RED}❌ DOWN{RESET}"
_ANSI_LOCAL = f"{GREEN}✅ OK  {RESET}"
_ANSI_STARTING = f"{YELLOW}⏳ ...  {RESET}"


def _color_status(up: bool | None) -> str:
    if up is None:
        return _ANSI_STARTING
    return _ANSI_OK if up else _ANSI_DOWN


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Step 0 — ensure the library is importable
# ---------------------------------------------------------------------------
def _ensure_library_installed() -> bool:
    """Return True if promptshield is importable (install if not)."""
    try:
        importlib.import_module("promptshield")
        return True
    except ImportError:
        pass

    log(f"{YELLOW}Installing promptshield library…{RESET}")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", "."],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log(f"{RED}Failed to install promptshield:{RESET}\n{result.stdout}\n{result.stderr}")
        return False
    return True


def _ensure_api_deps_installed() -> bool:
    """Install API dependencies if required packages are missing."""
    required_modules = ["fastapi", "uvicorn"]

    missing = []
    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(module_name)

    if not missing:
        return True

    req_file = os.path.join(ROOT, "api", "requirements.txt")
    log(f"{YELLOW}Installing API dependencies…{RESET}")

    if os.path.isfile(req_file):
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", req_file],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
    else:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", *missing],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

    if result.returncode != 0:
        log(f"{RED}Failed to install API dependencies:{RESET}\n{result.stdout}\n{result.stderr}")
        return False

    return True


# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------
_procs: list[subprocess.Popen] = []
_procs_lock = threading.Lock()


def _spawn(cmd: list[str], cwd: str, prefix: str, env: dict | None = None) -> subprocess.Popen:
    """Spawn a subprocess and stream its stdout/stderr with a coloured prefix."""
    merged_env = {**os.environ, **(env or {})}

    creationflags = 0
    start_new_session = False
    if platform.system() == "Windows":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        start_new_session = True

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=merged_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        creationflags=creationflags,
        start_new_session=start_new_session,
    )

    def _stream():
        assert proc.stdout is not None
        for line in proc.stdout:
            print(f"{CYAN}[{prefix}]{RESET} {line}", end="", flush=True)

    t = threading.Thread(target=_stream, daemon=True)
    t.start()

    with _procs_lock:
        _procs.append(proc)

    return proc


# ---------------------------------------------------------------------------
# Health polling
# ---------------------------------------------------------------------------
def _is_up(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status < 500
    except Exception:
        return False


def _wait_for_up(url: str, max_wait: float = 30.0, interval: float = 1.0) -> bool:
    deadline = time.time() + max_wait
    while time.time() < deadline:
        if _is_up(url):
            return True
        time.sleep(interval)
    return False


# ---------------------------------------------------------------------------
# Status table
# ---------------------------------------------------------------------------
def _print_status(api_up: bool | None, ui_up: bool | None) -> None:
    """Print a live status table."""
    lines = [
        "",
        f"  {BOLD}PromptShield Stack Status{RESET}",
        f"  {'─' * 54}",
        f"  {'Component':<20} {'Status':<14} {'URL':<30}",
        f"  {'─' * 54}",
        f"  {'Python Library':<20} {_ANSI_LOCAL:<14} {'(local)':<30}",
        f"  {'API Server':<20} {_color_status(api_up):<14} {f'http://localhost:{PORT_API}':<30}",
    ]
    if RUN_FRONTEND:
        lines.append(
            f"  {'Frontend (Vite)':<20} {_color_status(ui_up):<14} {f'http://localhost:{PORT_UI}':<30}"
        )
    lines += [
        f"  {'─' * 54}",
        f"  Press {BOLD}Ctrl+C{RESET} to stop all services.",
        "",
    ]
    print("\n".join(lines), flush=True)


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
def _shutdown(signum=None, frame=None):
    log(f"\n{YELLOW}Shutting down all services…{RESET}")
    with _procs_lock:
        for proc in _procs:
            try:
                if platform.system() == "Windows":
                    proc.terminate()
                else:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass

    for proc in _procs:
        try:
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    log(f"{GREEN}All services stopped. Goodbye!{RESET}")
    sys.exit(0)


signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    log(f"\n{BOLD}🛡  Starting PromptShield…{RESET}\n")
    log(f"{CYAN}Python:{RESET} {sys.version}")
    log(f"{CYAN}Working dir:{RESET} {ROOT}")

    if not _ensure_library_installed():
        log(f"{RED}Cannot import promptshield. Aborting.{RESET}")
        sys.exit(1)

    if not _ensure_api_deps_installed():
        log(f"{RED}Cannot install API dependencies. Aborting.{RESET}")
        sys.exit(1)

    # Start API server
    api_cmd = [
        sys.executable, "-m", "uvicorn",
        "api.main:app",
        "--port", str(PORT_API),
        "--reload",
    ]
    log(f"{CYAN}Starting API server on port {PORT_API}…{RESET}")
    _spawn(api_cmd, cwd=ROOT, prefix="API")

    # Start frontend dev server (optional, UI left untouched)
    if RUN_FRONTEND:
        if not os.path.isdir(FRONTEND_DIR):
            log(f"{RED}frontend/ directory not found at {FRONTEND_DIR}.{RESET}")
            log("Run the command from the repository root or use --no-frontend.")
            sys.exit(1)

        npm_exe = shutil.which("npm")
        if not npm_exe:
            log(f"{RED}npm was not found on PATH.{RESET}")
            log("Install Node.js from https://nodejs.org/ and reopen your terminal.")
            log("Or run the script with --no-frontend.")
            sys.exit(1)

        node_modules_dir = os.path.join(FRONTEND_DIR, "node_modules")
        if not os.path.isdir(node_modules_dir):
            log(f"{CYAN}Installing frontend npm packages…{RESET}")
            result = subprocess.run(
                [npm_exe, "install"],
                cwd=FRONTEND_DIR,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                log(f"{RED}npm install failed:{RESET}\n{result.stdout}\n{result.stderr}")
                sys.exit(1)

        npm_cmd = [npm_exe, "run", "dev", "--", "--port", str(PORT_UI)]
        log(f"{CYAN}Starting frontend dev server on port {PORT_UI}…{RESET}")
        _spawn(npm_cmd, cwd=FRONTEND_DIR, prefix="UI ")

    # Wait for services
    log(f"\n{YELLOW}Waiting for services to start (up to 30 s)…{RESET}")

    api_health_url = f"http://localhost:{PORT_API}/health"
    ui_url = f"http://localhost:{PORT_UI}"

    api_up = _wait_for_up(api_health_url, max_wait=30)
    ui_up = _wait_for_up(ui_url, max_wait=30) if RUN_FRONTEND else None

    if not api_up:
        log(f"{RED}API server did not start within 30 seconds. Check [API] logs above.{RESET}")
        _shutdown()
        sys.exit(1)

    if RUN_FRONTEND and not ui_up:
        log(f"{RED}Frontend did not start within 30 seconds. Check [UI ] logs above.{RESET}")
        _shutdown()
        sys.exit(1)

    log(
        f"\n{BOLD}{GREEN}🛡  PromptShield is ready → "
        f"http://localhost:{PORT_UI if RUN_FRONTEND else PORT_API}{RESET}{RESET}\n"
    )

    while True:
        api_status = _is_up(api_health_url)
        ui_status = _is_up(ui_url) if RUN_FRONTEND else None
        _print_status(api_up=api_status, ui_up=ui_status)
        time.sleep(5)


if __name__ == "__main__":
    main()
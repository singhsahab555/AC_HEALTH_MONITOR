"""
setup_scheduler.py — Register the pipeline with Windows Task Scheduler
========================================================================
FIX: The previous version failed with:
     ERROR: Value for '/TR' option cannot be more than 261 character(s).

This version writes a small launcher.bat file first, then points
Task Scheduler at that .bat file — solving the path-length limit.

Usage:
    python setup_scheduler.py              # register / update the task
    python setup_scheduler.py --remove     # delete the task
    python setup_scheduler.py --test       # run the task right now
    python setup_scheduler.py --status     # check task health
"""

import argparse
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

TASK_NAME    = "AC_Health_Pipeline"
TRIGGER_TIME = "11:00"          # 11:00 AM IST

# ─────────────────────────────────────────────────────────────────────────────
def is_admin() -> bool:
    try:
        import ctypes
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def get_python_exe() -> str:
    """Return pythonw.exe (no console popup) or python.exe as fallback."""
    py = Path(sys.executable)
    pythonw = py.parent / "pythonw.exe"
    return str(pythonw) if pythonw.exists() else str(py)


def pipeline_root() -> Path:
    return Path(__file__).parent.resolve()


def run_schtasks(args: list[str]) -> tuple[int, str, str]:
    result = subprocess.run(
        ["schtasks"] + args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.returncode, result.stdout, result.stderr


# ─────────────────────────────────────────────────────────────────────────────
# KEY FIX: write a launcher .bat that Task Scheduler calls
# The .bat file sits in the pipeline root and has a short enough path
# to stay well under the 261-char limit regardless of where the project lives.
# ─────────────────────────────────────────────────────────────────────────────
def write_launcher_bat(root: Path, py_exe: str, log_file: Path) -> Path:
    """
    Creates run_pipeline_launcher.bat in the pipeline root.
    Returns the path to the .bat file.
    """
    bat_path = root / "run_pipeline_launcher.bat"
    script   = root / "run_pipeline.py"

    bat_content = textwrap.dedent(f"""\
        @echo off
        cd /d "{root}"
        "{py_exe}" "{script}" >> "{log_file}" 2>&1
    """)

    bat_path.write_text(bat_content, encoding="utf-8")
    print(f"  Launcher .bat: {bat_path}")
    return bat_path


def create_task():
    root    = pipeline_root()
    py_exe  = get_python_exe()
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "scheduler_output.log"

    # Write the launcher bat
    bat_path = write_launcher_bat(root, py_exe, log_file)

    print(f"\n  Task name   : {TASK_NAME}")
    print(f"  Trigger     : Daily at {TRIGGER_TIME} IST")
    print(f"  Interpreter : {py_exe}")
    print(f"  Launcher    : {bat_path}")
    print(f"  TR length   : {len(str(bat_path))} chars (limit=261)")
    print(f"  Log append  : {log_file}")
    print()

    if len(str(bat_path)) > 260:
        print("  ⚠  WARNING: launcher .bat path is still > 260 chars.")
        print("     Move the ac_pipeline folder to a shorter path, e.g.:")
        print("     C:\\ac_pipeline\\")
        print("     Then re-run this script.")
        sys.exit(1)

    # Delete existing task silently
    run_schtasks(["/Delete", "/TN", TASK_NAME, "/F"])

    # Task Scheduler /TR points at the .bat (well under 261 chars)
    rc, out, err = run_schtasks([
        "/Create",
        "/TN",   TASK_NAME,
        "/TR",   f'"{bat_path}"',
        "/SC",   "DAILY",
        "/ST",   TRIGGER_TIME,
        "/F",
        "/RL",   "HIGHEST",
        "/RU",   os.environ.get("USERNAME", ""),
        "/IT",
    ])

    if rc != 0:
        print(f"  ❌ schtasks /Create failed (exit {rc})")
        print(f"  STDOUT: {out}")
        print(f"  STDERR: {err}")
        print()
        print("  Possible fixes:")
        print("  1. Run this script as Administrator")
        print("     Right-click → 'Run as administrator'")
        print("  2. If the path is still too long, move the project to:")
        print("     C:\\ac_pipeline\\")
        sys.exit(1)

    print(f"  ✅ Task '{TASK_NAME}' created successfully.")
    print()

    # Apply retry + run-if-missed via PowerShell
    ps_script = textwrap.dedent(f"""
        $task = Get-ScheduledTask -TaskName "{TASK_NAME}"
        $task.Settings.RestartCount = 1
        $task.Settings.RestartInterval = "PT5M"
        $task.Settings.StartWhenAvailable = $true
        $task.Settings.WakeToRun = $false
        Set-ScheduledTask -TaskName "{TASK_NAME}" -Settings $task.Settings
        Write-Host "Task settings updated."
    """).strip()

    ps = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
        capture_output=True, text=True
    )
    if ps.returncode == 0:
        print("  ✅ Retry settings applied (retry once after 5 min; run if missed).")
    else:
        print(f"  ⚠  PowerShell settings update: {ps.stderr.strip()}")
        print("     The task still runs — retry settings not applied.")


def remove_task():
    rc, out, err = run_schtasks(["/Delete", "/TN", TASK_NAME, "/F"])
    if rc == 0:
        print(f"  ✅ Task '{TASK_NAME}' removed.")
        # Also remove the launcher bat
        bat = pipeline_root() / "run_pipeline_launcher.bat"
        if bat.exists():
            bat.unlink()
            print(f"  ✅ Launcher .bat removed: {bat}")
    else:
        print(f"  ⚠  Could not remove task: {err.strip()}")


def check_status():
    rc, out, err = run_schtasks(["/Query", "/TN", TASK_NAME, "/FO", "LIST", "/V"])
    if rc != 0:
        print(f"  ❌ Task '{TASK_NAME}' not found.")
        print("     Run: python setup_scheduler.py")
        return
    for line in out.splitlines():
        for key in ("Task To Run", "Status", "Next Run Time",
                    "Last Run Time", "Last Result", "Run As User"):
            if line.strip().startswith(key + ":"):
                print(f"  {line.strip()}")


def test_run():
    print(f"\n  Triggering '{TASK_NAME}' right now …")
    rc, out, err = run_schtasks(["/Run", "/TN", TASK_NAME])
    if rc == 0:
        print("  ✅ Task triggered! Watch logs/scheduler_output.log for output.")
    else:
        print(f"  ❌ Failed: {err.strip()}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Manage AC_Health_Pipeline Windows Scheduled Task")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--remove", action="store_true")
    group.add_argument("--test",   action="store_true")
    group.add_argument("--status", action="store_true")
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║   AC Health Pipeline — Task Scheduler Setup     ║")
    print("╚══════════════════════════════════════════════════╝")

    if args.remove:
        remove_task(); return
    if args.status:
        check_status(); return
    if args.test:
        test_run(); return

    if not is_admin():
        print()
        print("  ⚠  Not running as Administrator.")
        print("     Right-click setup_scheduler.py → 'Run as administrator'")
        print()

    create_task()

    print()
    print("  ┌────────────────────────────────────────────────────┐")
    print(f"  │  ✅ Pipeline scheduled every morning at {TRIGGER_TIME} IST   │")
    print("  │                                                    │")
    print("  │  Verify:  python setup_scheduler.py --status      │")
    print("  │  Test now: python setup_scheduler.py --test       │")
    print("  │  Remove:  python setup_scheduler.py --remove      │")
    print("  └────────────────────────────────────────────────────┘")
    print()


if __name__ == "__main__":
    main()

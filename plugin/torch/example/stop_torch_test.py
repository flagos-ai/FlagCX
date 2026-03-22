#!/usr/bin/env python3
"""
Stop torchrun processes across multiple machines in batch.

Usage:
    python stop_torch_test.py --hostfile hosts.txt [--dry-run]

This script reads a hostfile and SSHs into each machine to kill torchrun
processes and their children. Useful when distributed training hangs due
to failures on some nodes.
"""

import argparse
import subprocess
import sys
from typing import List, Tuple


def parse_hostfile(path: str) -> List[str]:
    """
    Parse hostfile and return list of host IPs.
    
    Hostfile format (same as run_torch_test.py):
        <ip> slots=<n> type=<device_type>
    
    Only the IP is needed for stopping processes.
    """
    hosts = []
    with open(path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if not parts:
                continue
            ip = parts[0]
            hosts.append(ip)
    return hosts


def ssh_exec(host: str, command: str, dry_run: bool = False) -> Tuple[int, str, str]:
    """
    Execute a command on a remote host via SSH.
    
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    ssh_cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        host,
        command
    ]
    
    if dry_run:
        print(f"  [DRY-RUN] Would execute: ssh {host} '{command}'")
        return (0, "", "")
    
    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        return (result.returncode, result.stdout, result.stderr)
    except subprocess.TimeoutExpired:
        return (-1, "", "SSH command timed out")
    except Exception as e:
        return (-1, "", str(e))


def stop_processes_on_host(host: str, dry_run: bool = False) -> bool:
    """
    Stop torchrun processes on a remote host.

    Returns:
        True if successful (or no processes to kill), False on error.
    """
    print(f"\n[{host}] Stopping torchrun processes...")

    pattern = "torchrun"

    # First, list matching processes
    list_cmd = f"pgrep -af '{pattern}'"
    ret, stdout, stderr = ssh_exec(host, list_cmd, dry_run=False)

    if stdout.strip():
        print(f"  Found torchrun processes:")
        for line in stdout.strip().split('\n'):
            print(f"    {line}")
    else:
        print(f"  No torchrun processes found")
        return True

    # Kill the processes
    # Use pkill with SIGTERM first, then SIGKILL if needed
    kill_cmd = f"pkill -f '{pattern}' 2>/dev/null; sleep 1; pkill -9 -f '{pattern}' 2>/dev/null; echo 'done'"
    ret, stdout, stderr = ssh_exec(host, kill_cmd, dry_run=dry_run)

    if dry_run:
        return True

    if ret != 0 and "done" not in stdout:
        print(f"  Warning: pkill returned {ret}")
        if stderr:
            print(f"  stderr: {stderr}")
        return False
    else:
        print(f"  Killed torchrun processes")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Stop torchrun processes across multiple machines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Stop all torchrun processes
    python stop_torch_test.py --hostfile hosts.txt

    # Dry run to see what would be killed
    python stop_torch_test.py --hostfile hosts.txt --dry-run
        """
    )

    parser.add_argument(
        "--hostfile",
        required=True,
        help="Path to hostfile (same format as run_torch_test.py)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be killed without actually killing"
    )

    args = parser.parse_args()

    # Parse hostfile
    try:
        hosts = parse_hostfile(args.hostfile)
    except FileNotFoundError:
        print(f"Error: Hostfile not found: {args.hostfile}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing hostfile: {e}", file=sys.stderr)
        sys.exit(1)

    if not hosts:
        print("Error: No hosts found in hostfile", file=sys.stderr)
        sys.exit(1)

    print(f"Hosts to process: {', '.join(hosts)}")
    print("Will kill: torchrun processes")

    if args.dry_run:
        print("\n=== DRY RUN MODE - No processes will be killed ===")

    # Stop processes on each host
    all_success = True
    for host in hosts:
        success = stop_processes_on_host(host, dry_run=args.dry_run)
        if not success:
            all_success = False

    print("\n" + "=" * 50)
    if args.dry_run:
        print("Dry run complete. No processes were killed.")
    elif all_success:
        print("All processes stopped successfully.")
    else:
        print("Some operations failed. Check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Automate launching the torch API test across multiple nodes.
- Parse hostfile lines like: 192.168.1.1 slots=8 type=A800
- Parse YAML env config with common and device-type-specific envs.
- Validate device types and slot counts.
- Generate per-host run.sh with env exports + torchrun command.
- Copy run.sh to each host (scp) and execute via ssh (passwordless assumed).

Requires PyYAML.
"""
import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from typing import Any, Dict, List

try:
    import yaml
except ImportError:  # pragma: no cover
    print("PyYAML is required. Install with: python -m pip install pyyaml", file=sys.stderr)
    sys.exit(1)


class ConfigError(Exception):
    pass


def parse_hostfile(path: str) -> List[Dict[str, str]]:
    hosts: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ConfigError(f"Invalid hostfile line: '{line}'")
            host = parts[0]
            slots = None
            dtype = None
            for p in parts[1:]:
                if p.startswith("slots="):
                    slots = int(p.split("=", 1)[1])
                elif p.startswith("type="):
                    dtype = p.split("=", 1)[1]
            if slots is None or dtype is None:
                raise ConfigError(f"Missing slots/type in line: '{line}'")
            hosts.append({"host": host, "slots": slots, "type": dtype})
    if not hosts:
        raise ConfigError("Hostfile is empty")
    return hosts


def load_env_config(path: str) -> Dict[str, Any]:
    """
    Reads from yaml config file designated by the user, see example_torch_env.yaml for expected format.

    Returns a dict with keys:
        - common_env: Dict[str, str]
        - device_specific_env: Dict[str, Dict[str, str]]
        - before_start: str
        - test_dir: str
        - master_port: int
        - master_addr: str or None
        - testfile: str
        - log_dir: str
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cmds = data.get("cmds", {})
    if cmds and not isinstance(cmds, dict):
        raise ConfigError("cmds must be a mapping")
    before_start = ""
    if isinstance(cmds, dict):
        before_start = cmds.get("before_start", "") or ""

    test_dir = data.get("test_dir", None)
    if not test_dir:
        raise ConfigError("'test_dir' must be provided in the config")
    test_dir_abs = os.path.abspath(os.path.expanduser(test_dir))

    log_dir = data.get("log_dir", None)
    if not log_dir:
        raise ConfigError("'log_dir' must be provided in the config")
    log_dir_abs = os.path.abspath(os.path.expanduser(log_dir))

    testfile = data.get("testfile", None)
    if not testfile:
        raise ConfigError("'testfile' must be provided in the config")
    testfile_abs = os.path.abspath(os.path.expanduser(testfile))

    master_port = data.get("master_port", 12345)
    try:
        master_port = int(master_port)
    except Exception as exc:
        raise ConfigError("'master_port' must be an integer") from exc
    if not (10000 <= master_port <= 19999):
        raise ConfigError("'master_port' must be in range 10000-19999")

    master_addr = data.get("master_addr", None)

    envs = data.get("envs", {})
    if not isinstance(envs, dict):
        raise ConfigError("envs must be a mapping")
    device_specific = envs.get("device_type_specific", {})
    if not isinstance(device_specific, dict):
        raise ConfigError("envs.device_type_specific must be a mapping")
    common = {k: v for k, v in envs.items() if k != "device_type_specific"}

    return {
        "common_env": common,
        "device_specific_env": device_specific,
        "before_start": before_start,
        "test_dir": test_dir_abs,
        "master_port": master_port,
        "master_addr": master_addr,
        "testfile": testfile_abs,
        "log_dir": log_dir_abs,
    }


def validate_hosts(hosts: List[Dict[str, str]], device_specific: Dict[str, Dict[str, str]]):
    slot_counts = {h["slots"] for h in hosts}
    if len(slot_counts) != 1:
        raise ConfigError(f"Inconsistent slots per node: {sorted(slot_counts)}. torchrun requires a consistent nproc_per_node.")
    host_types = {h["type"] for h in hosts}
    extra_types = set(device_specific.keys()) - host_types
    if extra_types:
        print(
            f"Warning: device_type_specific has entries not present in hostfile: {sorted(extra_types)}",
            file=sys.stderr,
        )


def merge_envs(common: Dict[str, str], specific: Dict[str, str]) -> Dict[str, str]:
    env = {}
    env.update(common)
    env.update(specific)
    return env


def format_env_exports(env: Dict[str, str]) -> str:
    lines = []
    for k, v in env.items():
        if v is None:
            continue
        lines.append(f"export {k}={shlex.quote(str(v))}")
    return "\n".join(lines)


def build_run_script(env_exports: str, command: str, pre_command: str) -> str:
    pre = pre_command.strip()
    pre_block = f"{pre}\n\n" if pre else ""
    return """#!/bin/bash
set -euo pipefail
{pre_block}{env_exports}

{command}
""".format(env_exports=env_exports, command=command, pre_block=pre_block)


def run_ssh_command(host: str, command: str) -> None:
    """Execute a command on a remote host via SSH.

    Args:
        host: The remote host to connect to.
        command: The command to execute on the remote host.

    Exits with the subprocess return code if the command fails.
    """
    ssh_cmd = f"ssh {host} {shlex.quote(command)}"
    result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"SSH command failed on {host}: {command}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        sys.exit(result.returncode)


def run_scp_command(host: str, src: str, dst: str) -> None:
    """Copy a file to a remote host via SCP.

    Args:
        host: The remote host to copy to.
        src: The local source file path.
        dst: The remote destination file path.

    Exits with the subprocess return code if the command fails.
    """
    scp_cmd = f"scp {shlex.quote(src)} {host}:{shlex.quote(dst)}"
    result = subprocess.run(scp_cmd, shell=True)
    if result.returncode != 0:
        print(f"SCP command failed on {host}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Auto-generate and run torchrun scripts across nodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run torch test across multiple nodes
    python run_torch_test.py --hostfile hosts.txt --config env_config.yaml

    # Run with extra arguments passed to the test script
    python run_torch_test.py --hostfile hosts.txt --config env_config.yaml --extra-args "--batch-size 32"

    # Dry run to see generated scripts without executing
    python run_torch_test.py --hostfile hosts.txt --config env_config.yaml --dry-run
        """
    )
    parser.add_argument("--hostfile", required=True, help="Path to hostfile")
    parser.add_argument("--config", required=True, help="Path to YAML env config")
    parser.add_argument("--extra-args", default="", help="Extra args appended to the command")
    parser.add_argument("--dry-run", action="store_true", help="Generate scripts but do not execute remotely")
    args = parser.parse_args()

    try:
        hosts = parse_hostfile(args.hostfile)
        config = load_env_config(args.config)
        validate_hosts(hosts, config["device_specific_env"])
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        sys.exit(1)

    nnodes = len(hosts)
    nproc_per_node = hosts[0]["slots"]
    master_addr = config["master_addr"] or hosts[0]["host"]
    master_port = config["master_port"]

    for node_rank, h in enumerate(hosts):
        env = merge_envs(config["common_env"], config["device_specific_env"].get(h["type"], {}))
        env_exports = format_env_exports(env)

        cmd = (
            f"torchrun --nnodes {nnodes} --nproc_per_node {nproc_per_node} --node_rank {node_rank} "
            f"--master_addr {master_addr} --master_port {master_port} {config['testfile']}"
        )
        if args.extra_args:
            cmd = f"{cmd} {args.extra_args}"

        run_sh = build_run_script(env_exports, cmd, config["before_start"])

        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp:
            tmp.write(run_sh)
            tmp_path = tmp.name

        host_run_name = f"run_torch_test_{h['host']}.sh"
        remote_path = os.path.join(config["test_dir"], host_run_name)
        log_name = f"run_torch_test_{h['host']}.log"
        log_path = os.path.join(config["log_dir"], log_name)

        try:
            run_ssh_command(h["host"], f"mkdir -p {os.path.dirname(remote_path)}")
            run_scp_command(h["host"], tmp_path, remote_path)
            run_ssh_command(h["host"], f"chmod +x {remote_path}")
            run_ssh_command(h["host"], f"mkdir -p {config['log_dir']}")
            if not args.dry_run:
                run_ssh_command(h["host"], f"nohup {remote_path} > {log_path} 2>&1 &")
                print(f"Started test on {h['host']}, output will be logged to {log_path}")
            else:
                print(f"[dry-run] Generated {remote_path} on {h['host']}, would log to {log_path}")
        finally:
            os.remove(tmp_path)

    print(f"All {nnodes} nodes launched. Check log files in {config['log_dir']} for output.")


if __name__ == "__main__":
    main()

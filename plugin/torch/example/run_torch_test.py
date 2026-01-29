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
from typing import Dict, List, Tuple

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


def load_env_config(path: str) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]], str, str, int, str, str]:
    """
    Expect structure:
    cmds:
      before_start: some command
    envs:
      VAR1: value
      VAR2: value
      device_type_specific:
        TYPEA: { VARX: val }
        TYPEB: { VARY: val }
    Common vars are the direct children of envs, excluding device_type_specific.
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

    testfile = data.get("testfile", None)
    if not testfile:
        raise ConfigError("'testfile' must be provided in the config")
    testfile_abs = os.path.abspath(os.path.expanduser(testfile))

    master_port = data.get("master_port", 8281)
    try:
        master_port = int(master_port)
    except Exception as exc:
        raise ConfigError("'master_port' must be an integer") from exc

    master_addr = data.get("master_addr", None)

    envs = data.get("envs", {})
    if not isinstance(envs, dict):
        raise ConfigError("envs must be a mapping")
    device_specific = envs.get("device_type_specific", {})
    if not isinstance(device_specific, dict):
        raise ConfigError("envs.device_type_specific must be a mapping")
    common = {k: v for k, v in envs.items() if k != "device_type_specific"}
    return common, device_specific, before_start, test_dir_abs, master_port, master_addr, testfile_abs


def validate_hosts(hosts: List[Dict[str, str]], device_specific: Dict[str, Dict[str, str]]):
    slot_counts = {h["slots"] for h in hosts}
    if len(slot_counts) != 1:
        raise ConfigError(f"Inconsistent slots per node: {sorted(slot_counts)}. torchrun requires a consistent nproc_per_node.")
    for h in hosts:
        if h["type"] not in device_specific:
            raise ConfigError(f"Device type '{h['type']}' not found in config device_type_specific")


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


def scp_push(host: str, local_path: str, remote_path: str):
    subprocess.run(["ssh", host, "mkdir", "-p", os.path.dirname(remote_path)], check=True)
    subprocess.run(["scp", local_path, f"{host}:{remote_path}"], check=True)


def ssh_exec(host: str, remote_path: str):
    subprocess.run(["ssh", host, remote_path], check=True)


def main():
    parser = argparse.ArgumentParser(description="Auto-generate and run torchrun scripts across nodes")
    parser.add_argument("--hostfile", required=True, help="Path to hostfile")
    parser.add_argument("--config", required=True, help="Path to YAML env config")
    parser.add_argument("--extra-args", default="", help="Extra args appended to the command")
    parser.add_argument("--dry-run", action="store_true", help="Generate scripts but do not execute remotely")
    args = parser.parse_args()

    try:
        hosts = parse_hostfile(args.hostfile)
        common_env, device_specific_env, before_start_cmd, test_dir_abs, master_port_cfg, master_addr_cfg, testfile_abs = load_env_config(args.config)
        validate_hosts(hosts, device_specific_env)
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        sys.exit(1)

    remote_path = os.path.join(test_dir_abs, "run_torch_test.sh")

    nnodes = len(hosts)
    nproc_per_node = hosts[0]["slots"]
    master_addr = master_addr_cfg or hosts[0]["host"]
    master_port = master_port_cfg

    for node_rank, h in enumerate(hosts):
        env = merge_envs(common_env, device_specific_env.get(h["type"], {}))
        env_exports = format_env_exports(env)

        cmd = (
            f"torchrun --nnodes {nnodes} --nproc_per_node {nproc_per_node} --node_rank {node_rank} "
            f"--master_addr {master_addr} --master_port {master_port} {testfile_abs}"
        )
        if args.extra_args:
            cmd = f"{cmd} {args.extra_args}"

        run_sh = build_run_script(env_exports, cmd, before_start_cmd)

        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp:
            tmp.write(run_sh)
            tmp_path = tmp.name

        try:
            scp_push(h["host"], tmp_path, remote_path)
            subprocess.run(["ssh", h["host"], "chmod", "+x", remote_path], check=True)
            if not args.dry_run:
                ssh_exec(h["host"], remote_path)
            else:
                print(f"[dry-run] Generated {remote_path} on {h['host']} but did not execute")
        finally:
            os.remove(tmp_path)

    print("All nodes processed successfully.")


if __name__ == "__main__":
    main()

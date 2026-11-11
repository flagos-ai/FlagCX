#!/usr/bin/env bash
# Share only networking: binaries, SDK libraries, mounts and PIDs stay in the
# job container. Do not launch a second container or enter the host filesystem.
set -euo pipefail

if [[ "${1:-}" == --in-host-network ]]; then
  expected_namespace=$2
  shift 2
  actual_namespace=$(readlink /proc/self/ns/net)
  if [[ "$actual_namespace" != "$expected_namespace" ]]; then
    echo "Host network namespace mismatch: expected=$expected_namespace actual=$actual_namespace" >&2
    exit 1
  fi
  echo "Hardware workload network namespace: $actual_namespace (host)"
  python3 - <<'PY'
import socket
import subprocess
import tempfile
from pathlib import Path

# Docker's embedded resolver only serves the original bridge namespace. Give
# this workload the host resolver in its private mount namespace; sibling
# checkout/artifact processes keep the original container DNS configuration.
resolver = Path("/etc/resolv.conf")
if any(line.split()[:2] == ["nameserver", "127.0.0.11"]
       for line in resolver.read_text().splitlines()):
    host_resolver = Path("/run/flagcx-host-proc/1/root/etc/resolv.conf")
    if not host_resolver.is_file():
        raise SystemExit("Host resolv.conf is unavailable for the hardware workload")
    # Copy before mounting: canonicalizing /proc/1/root would resolve paths
    # against the container root. The mounted inode survives tempfile cleanup.
    with tempfile.NamedTemporaryFile(prefix="flagcx-host-resolv-") as source:
        source.write(host_resolver.read_bytes())
        source.flush()
        Path(source.name).chmod(0o644)
        subprocess.run(["mount", "--bind", source.name, str(resolver)], check=True)
    print("Hardware workload DNS: host resolv.conf (private mount)", flush=True)

interfaces = {name for _, name in socket.if_nameindex()}
print("Network interfaces: " + ", ".join(sorted(interfaces)), flush=True)
usable_ports = 0
for port in sorted(Path("/sys/class/infiniband").glob("mlx5_*/ports/*")):
    if not (port / "state").read_text().startswith("4:"):
        continue
    if (port / "link_layer").read_text().strip() != "Ethernet":
        continue
    gid_interfaces = set()
    for entry in (port / "gid_attrs/ndevs").glob("*"):
        try:
            name = entry.read_text().strip()
        except OSError:
            # Unpopulated GID table slots can return EINVAL/ENODEV.
            continue
        if name:
            gid_interfaces.add(name)
    visible = gid_interfaces & interfaces
    print(f"RDMA {port.parent.parent.name} port {port.name}: "
          f"GID interfaces={sorted(gid_interfaces)}, visible={sorted(visible)}",
          flush=True)
    if not visible:
        raise SystemExit("RDMA GID interfaces are absent from the selected host "
                         "network namespace; check the runner/DinD host network")
    usable_ports += 1
if not usable_ports:
    raise SystemExit("No ACTIVE RoCE port is visible in the host network namespace")
PY
  exec "$@"
fi

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <command> [arguments...]" >&2
  exit 2
fi

namespace_file=${FLAGCX_CI_HOST_NETNS:-/run/flagcx-host-proc/1/ns/net}
if [[ ! -e "$namespace_file" ]]; then
  echo "Missing $namespace_file; mount /proc:/run/flagcx-host-proc:ro in the job container" >&2
  exit 1
fi
for tool in nsenter unshare mount; do
  if ! command -v "$tool" >/dev/null; then
    echo "$tool is required in the CI image (util-linux)" >&2
    exit 1
  fi
done
expected_namespace=$(readlink "$namespace_file")
echo "Entering host network: current=$(readlink /proc/self/ns/net) target=$expected_namespace"
script_path=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")
exec unshare --mount --propagation private -- \
  nsenter --net="$namespace_file" -- \
  bash "$script_path" --in-host-network "$expected_namespace" "$@"

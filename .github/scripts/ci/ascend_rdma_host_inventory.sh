#!/usr/bin/env bash
# Read-only host inventory for Ascend NPU Direct RDMA diagnostics.
set -euo pipefail

echo "Ascend RDMA host: kernel=$(uname -r)"
echo 'Ascend RDMA host: device nodes and sysfs:'
ls -l /dev/infiniband /sys/class/infiniband /sys/class/infiniband_verbs 2>&1 || true
if command -v ibv_devices >/dev/null 2>&1; then
  echo 'Ascend RDMA host: ibv_devices:'
  timeout 10s env IBV_SHOW_WARNINGS=1 ibv_devices 2>&1 || true
fi
if command -v rdma >/dev/null 2>&1; then
  echo 'Ascend RDMA host: rdma link show:'
  timeout 5s rdma link show 2>&1 || true
fi
if [[ -r /proc/modules ]]; then
  for module in npu_peermem mlx5_ib ib_uverbs ib_core; do
    if grep -q "^${module} " /proc/modules; then
      echo "Ascend RDMA host: module ${module}=loaded"
    else
      echo "Ascend RDMA host: module ${module}=not-loaded"
    fi
  done
fi

if command -v modinfo >/dev/null 2>&1; then
  for module in npu_peermem mlx5_ib; do
    if filename=$(modinfo -F filename "$module" 2>/dev/null); then
      echo "Ascend RDMA host: ${module} file=$filename"
      echo "Ascend RDMA host: ${module} vermagic=$(modinfo -F vermagic "$module" 2>/dev/null || true)"
      echo "Ascend RDMA host: ${module} depends=$(modinfo -F depends "$module" 2>/dev/null || true)"
    else
      echo "Ascend RDMA host: ${module} modinfo=unavailable"
    fi
  done
fi

if [[ -r /proc/kallsyms ]]; then
  for symbol in ib_register_peer_memory_client ib_unregister_peer_memory_client; do
    if grep -q "[[:space:]]${symbol}$" /proc/kallsyms; then
      echo "Ascend RDMA host: ${symbol}=present"
    else
      echo "Ascend RDMA host: ${symbol}=not-visible"
    fi
  done
fi

if command -v ofed_info >/dev/null 2>&1; then
  echo "Ascend RDMA host: OFED=$(ofed_info -s 2>/dev/null || echo unavailable)"
else
  echo "Ascend RDMA host: ofed_info=unavailable"
fi

if command -v journalctl >/dev/null 2>&1 && command -v timeout >/dev/null 2>&1; then
  echo 'Ascend RDMA host: recent peer-memory kernel messages:'
  timeout 5s journalctl -k --no-pager -n 300 2>/dev/null | \
    grep -Ei 'mlx5|rdma|uverbs|pci.*reset|npu.peermem|ib_register_peer_memory_client|hal_kernel_p2p_get_pages|dma_map_resource' || true
fi

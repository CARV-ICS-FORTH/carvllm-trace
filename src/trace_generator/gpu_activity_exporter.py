#   Copyright 2025 - 2026 Polidoros Dafnomilis, FORTH, Greece
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
# gpu_activity_exporter.py

import os
import time
import math
import threading

from prometheus_client import start_http_server, Gauge
import pynvml as nv

import torch
from torch.profiler import profile, ProfilerActivity

# ------------------- Prometheus Gauges -------------------
G_COMM = Gauge("gpu_comm_active_pct", "Pct of CUDA time in NCCL comm kernels (process-level)", ["proc"])

G_MEM_USED = Gauge("gpu_mem_used_bytes", "GPU memory used (bytes)", ["gpu"])
G_TEMP     = Gauge("gpu_temperature_c",  "GPU temperature (C)",    ["gpu"])
G_POWER    = Gauge("gpu_power_watts",    "GPU board power (W)",    ["gpu"])

# NVML PCIe instantaneous throughput (bytes/s)
G_PCIE_TX  = Gauge("gpu_pcie_tx_bytes_s", "NVML PCIe TX throughput (bytes/s)", ["gpu"])
G_PCIE_RX  = Gauge("gpu_pcie_rx_bytes_s", "NVML PCIe RX throughput (bytes/s)", ["gpu"])

# NVML utilization
G_UTIL     = Gauge("gpu_utilization_pct",     "NVML GPU Utilization (%)",     ["gpu"])
G_UTIL_MEM = Gauge("gpu_mem_utilization_pct", "NVML Memory Utilization (%)",  ["gpu"])

# Process-level compute vs copy (from bracketed profiler window)
G_COMP = Gauge("gpu_compute_active_pct",
               "Pct of CUDA device time spent in kernels (process-level)", ["proc"])
G_COPY = Gauge("gpu_copy_active_pct",
               "Pct of CUDA device time spent in memcpys/memsets (process-level)", ["proc"])

# Absolute times in the last window (ms)
G_COMP_MS = Gauge("gpu_compute_time_ms",
                  "CUDA kernel time in last profiling window (ms)", ["proc"])
G_COPY_MS = Gauge("gpu_copy_time_ms",
                  "CUDA memcpy/memset time in last profiling window (ms)", ["proc"])


# ------------------- NVML poller -------------------

def _nvml_poller(period_s: float):
    """Poll NVML counters periodically and expose them as Prometheus gauges."""
    try:
        nv.nvmlInit()
        n = nv.nvmlDeviceGetCount()
    except Exception as e:
        print("[gpu-activity-exporter] NVML init failed:", e, flush=True)
        return

    def pcie_bytes(handle, which):
        try:
            # returns KB/s; convert to bytes/s
            kb = nv.nvmlDeviceGetPcieThroughput(handle, which)
            return float(kb) * 1024.0
        except Exception:
            return 0.0

    while True:
        try:
            for i in range(n):
                h = nv.nvmlDeviceGetHandleByIndex(i)

                # memory
                mem = nv.nvmlDeviceGetMemoryInfo(h)
                G_MEM_USED.labels(gpu=str(i)).set(mem.used)

                # temperature
                try:
                    t = nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)
                except Exception:
                    t = float("nan")
                G_TEMP.labels(gpu=str(i)).set(0.0 if math.isnan(t) else t)

                # power (mW -> W)
                try:
                    mw = nv.nvmlDeviceGetPowerUsage(h)
                    G_POWER.labels(gpu=str(i)).set(mw / 1000.0)
                except Exception:
                    G_POWER.labels(gpu=str(i)).set(0.0)

                # utilization
                try:
                    util = nv.nvmlDeviceGetUtilizationRates(h)
                    G_UTIL.labels(gpu=str(i)).set(util.gpu)
                    G_UTIL_MEM.labels(gpu=str(i)).set(util.memory)
                except Exception:
                    G_UTIL.labels(gpu=str(i)).set(0.0)
                    G_UTIL_MEM.labels(gpu=str(i)).set(0.0)

                # PCIe throughput
                G_PCIE_TX.labels(gpu=str(i)).set(pcie_bytes(h, nv.NVML_PCIE_UTIL_TX_BYTES))
                G_PCIE_RX.labels(gpu=str(i)).set(pcie_bytes(h, nv.NVML_PCIE_UTIL_RX_BYTES))

            time.sleep(period_s)
        except Exception as e:
            print("[gpu-activity-exporter] NVML loop error:", e, flush=True)
            time.sleep(period_s)

# ------------------- Profiler helpers -------------------

_prof_ctx = None  # active torch.profiler context (only in the training thread)

def _device_us(ev) -> float:
    v = getattr(ev, "device_time_total", None)
    if v is None:
        v = getattr(ev, "cuda_time_total", 0.0)  # backward compatibility
    return float(v or 0.0)

def _cpu_us(ev) -> float:
    v = getattr(ev, "cpu_time_total", 0.0)
    return float(v or 0.0)

def _is_gpu_event(ev) -> bool:
    return _device_us(ev) > 0.0

def _classify(name: str) -> str:
    n = name.lower()
    if ("nccl" in n
        or "allreduce" in n
        or "all_gather" in n or "allgather" in n
        or "reduce_scatter" in n or "reducescatter" in n
        or "broadcast" in n or "alltoall" in n):
        return "comm"
    if "memcpy" in n or "memset" in n or "mem transfer" in n:
        return "copy"
    return "compute"


# -------------------  begin / end -------------------

_prof_ctx = None  # active torch.profiler context (only in the training thread)

def profiler_begin():
    global _prof_ctx
    if _prof_ctx is not None:
        return  # already open

    want_stack = os.environ.get("GPU_EXPORTER_WITH_STACK", "0") not in ("0", "", "false", "False")
    _prof_ctx = profile(
        activities=[ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=want_stack,   # <- if you want stack traces: export GPU_EXPORTER_WITH_STACK=1
    )
    _prof_ctx.__enter__()  # open profiler window

def profiler_end():
    """
    Close the profiler, compute percentages and (optionally) dump ALL events to CSV.
    Controlled by:
      export GPU_EXPORTER_DUMP_ALL=1
      export GPU_EXPORTER_DUMP_DIR=/path (optional)
    """
    import socket
    import os
    import csv

    global _prof_ctx
    if _prof_ctx is None:
        return

    # env toggles
    dump_all = os.environ.get("GPU_EXPORTER_DUMP_ALL", "0") == "1"
    dump_dir = os.environ.get("GPU_EXPORTER_DUMP_DIR", ".")
    host = socket.gethostname()
    try:
        rank = int(os.environ.get("RANK", "0"))
    except Exception:
        rank = 0

    try:
        # Close profiler and fetch events
        _prof_ctx.__exit__(None, None, None)
        events = _prof_ctx.events()

        # ------------------------------------------------------------------
        # 1) Compute global compute / copy / comm percentages
        # ------------------------------------------------------------------
        compute_us = 0.0
        copy_us    = 0.0
        comm_us    = 0.0

        def classify(name: str) -> str:
            n = name.lower()
            if "nccl" in n:
                return "comm"
            if "memcpy" in n or "memset" in n or "mem transfer" in n:
                return "copy"
            return "compute"

        # Only events with some device time
        evs = [e for e in events if (getattr(e, "device_time_total", 0.0) or 0.0) > 0.0]
        for ev in evs:
            dur = float(getattr(ev, "device_time_total", 0.0) or 0.0)
            kind = classify(getattr(ev, "name", ""))
            if kind == "copy":
                copy_us += dur
            elif kind == "comm":
                comm_us += dur
            else:
                compute_us += dur

        total = compute_us + copy_us + comm_us
        comp = ((compute_us / total) * 100.0) if total > 0 else 0.0
        cpy  = ((copy_us   / total) * 100.0) if total > 0 else 0.0
        com  = ((comm_us   / total) * 100.0) if total > 0 else 0.0

        G_COMP.labels(proc="global").set(comp)
        G_COPY.labels(proc="global").set(cpy)
        try:
            G_COMM.labels(proc="global").set(com)
        except NameError:
            # G_COMM might not be defined in some builds
            pass

        # Epoch tag is provided externally, e.g.:
        #   os.environ["GPU_EXPORTER_EPOCH"] = str(epoch+1)
        epoch_tag = os.environ.get("GPU_EXPORTER_EPOCH", "")

        # ------------------------------------------------------------------
        # 2) CSV dump of all events (if requested)
        # ------------------------------------------------------------------
        if dump_all:
            os.makedirs(dump_dir, exist_ok=True)
            out_path = os.path.join(
                dump_dir, f"profiler_events_{host}_rank{rank}.csv"
            )

            evs_sorted = evs  # keep profiler order

            def _get(e, attr, default=None):
                try:
                    return getattr(e, attr, default)
                except Exception:
                    return default

            def fmt2(x):
                """Format numeric values with 2 decimals, keep non-numerics as-is."""
                try:
                    return f"{float(x):.2f}"
                except (TypeError, ValueError):
                    return x

            # Simple "window"/iteration counter across profiler_end calls
            if not hasattr(profiler_end, "_window_id"):
                profiler_end._window_id = 0
            profiler_end._window_id += 1
            iteration_no = profiler_end._window_id  # renamed from window_id

            file_exists = os.path.exists(out_path)
            with open(out_path, "a" if file_exists else "w",
                      newline="", encoding="utf-8") as f:

                # ----------------------------------------------------------
                # Header line (only once) - with '#' at the beginning
                # ----------------------------------------------------------
                if not file_exists:
                    header_cols = [
                        "rank",
                        "epoch",
                        "iteration_no",     # was: window_id
                        "event_id",         # was: event_index
                        "event_operation",  # was: name

                        "device_us",
                        "cpu_us",
                        "time_range_start_us",
                        "time_range_end_us",
                        "time_range_duration_us",

                        "self_cpu_memory_usage",
                        "self_cpu_percent",
                        "self_cpu_time_total",
                        "self_cpu_time_total_str",
                        "self_cuda_memory_usage",
                        "self_cuda_time_total",
                        "self_device_memory_usage",
                        "self_device_time_total",
                        "self_device_time_total_str",
                        "sequence_nr",
                        "stack",
                        "thread",
                        "total_cpu_percent",
                        "total_device_percent",
                        "use_device",
                    ]

                    # Example: #rank|epoch|iteration_no|...
                    header_line = "#" + "|".join(header_cols)
                    f.write(header_line + "\n")

                # Writer for data rows (pipe-delimited)
                writer = csv.writer(f, delimiter="|", lineterminator="\n")

                for i, e in enumerate(evs_sorted, 1):
                    name   = _get(e, "name", "")
                    dev_us = float(_get(e, "device_time_total", 0.0) or 0.0)
                    cpu_us = float(_get(e, "cpu_time_total", 0.0) or 0.0)

                    rng = _get(e, "time_range", None)
                    if rng and hasattr(rng, "start") and hasattr(rng, "end"):
                        start_us = rng.start
                        end_us   = rng.end
                        dur_us   = end_us - start_us
                    else:
                        start_us = None
                        end_us   = None
                        dur_us   = None

                    stk = _get(e, "stack", [])
                    if isinstance(stk, (list, tuple)):
                        stack_str = " | ".join(str(s) for s in stk)
                    else:
                        stack_str = str(stk) if stk is not None else ""

                    writer.writerow([
                        # rank / epoch / iteration / event_id / operation
                        rank,
                        epoch_tag,
                        iteration_no,
                        i,
                        name,

                        # timing and range (2 decimals)
                        fmt2(dev_us),
                        fmt2(cpu_us),
                        fmt2(start_us),
                        fmt2(end_us),
                        fmt2(dur_us),

                        # self / total metrics
                        fmt2(_get(e, "self_cpu_memory_usage", 0)),
                        fmt2(_get(e, "self_cpu_percent", -1)),
                        fmt2(_get(e, "self_cpu_time_total", 0)),
                        _get(e, "self_cpu_time_total_str", "0.000us"),
                        fmt2(_get(e, "self_cuda_memory_usage", 0)),
                        fmt2(_get(
                            e, "self_cuda_time_total",
                            _get(e, "self_device_time_total", 0)
                        )),
                        fmt2(_get(e, "self_device_memory_usage", 0)),
                        fmt2(_get(e, "self_device_time_total", 0)),
                        _get(
                            e,
                            "self_device_time_total_str",
                            str(_get(e, "self_device_time_total", "0")) + "us"
                        ),
                        _get(e, "sequence_nr", -1),
                        stack_str,
                        _get(e, "thread", None),
                        fmt2(_get(e, "total_cpu_percent", -1)),
                        fmt2(_get(e, "total_device_percent", -1)),
                        _get(e, "use_device", "cuda"),
                    ])
            '''
            print(
                f"[gpu-activity-exporter] appended {len(evs_sorted)} events "
                f"to {out_path}",
                flush=True,
            )
            '''
    finally:
        _prof_ctx = None




# ------------------- Exporter bootstrap -------------------

def start_exporter(port: int = 9108, nvml_period_s: float = 1.0):
    """
    Start the HTTP server for Prometheus and the NVML polling thread.
    Compute/copy metrics come from your profiler_begin()/profiler_end() brackets.
    """
    start_http_server(port)
    threading.Thread(target=_nvml_poller, args=(nvml_period_s,), daemon=True).start()
    print(f"[gpu-activity-exporter] up on :{port} | NVML every {nvml_period_s}s | profiler=bracketed",
          flush=True)

if __name__ == "__main__":
    port = int(os.environ.get("GPU_EXPORTER_PORT", "9108"))
    period = float(os.environ.get("GPU_EXPORTER_NVML_PERIOD_S", "1.0"))
    start_exporter(port=port, nvml_period_s=period)
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        pass

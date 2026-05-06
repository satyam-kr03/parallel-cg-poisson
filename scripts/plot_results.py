import glob
import os
import re

import matplotlib.pyplot as plt

time_pattern = re.compile(r"CG done in (\d+) iterations, rel_res=([0-9.eE+-]+), time=([0-9.]+) s")
profile_pattern = re.compile(
    r"Profile breakdown \(s\): comm=([0-9.]+), spmv=([0-9.]+), dot=([0-9.]+), axpy=([0-9.]+),?\s*allreduce=([0-9.]+)"
)
name_pattern = re.compile(r"N(\d+)_T(\d+)_P(\d+)")

def parse_logs(paths):
    parsed = []
    for path in paths:
        name = os.path.basename(path).replace(".log", "")
        name_match = name_pattern.match(name)
        if not name_match:
            continue
        N = int(name_match.group(1))
        T = int(name_match.group(2))
        P = int(name_match.group(3))

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        time_match = time_pattern.search(content)
        time_s = float(time_match.group(3)) if time_match else None
        iters = int(time_match.group(1)) if time_match else None

        profile_match = profile_pattern.search(content)
        profile = None
        if profile_match:
            profile = {
                "comm": float(profile_match.group(1)),
                "spmv": float(profile_match.group(2)),
                "dot": float(profile_match.group(3)),
                "axpy": float(profile_match.group(4)),
                "allreduce": float(profile_match.group(5)),
            }

        parsed.append(
            {
                "name": name,
                "N": N,
                "T": T,
                "P": P,
                "time_s": time_s,
                "iters": iters,
                "profile": profile,
            }
        )
    return parsed

scaling_logs = sorted(glob.glob("results/scaling/*.log"))
omp_logs = sorted(glob.glob("results/omp_threads/*.log"))

scaling_records = parse_logs(scaling_logs)
omp_records = parse_logs(omp_logs) if omp_logs else []

if not scaling_records:
    print("No scaling logs found.")
    raise SystemExit(0)

os.makedirs("results/scaling", exist_ok=True)
os.makedirs("report/figures", exist_ok=True)

# Scaling summary: grouped bars by N with MPI ranks as groups (T=1).
summary_records = [r for r in scaling_records if r["T"] == 1 and r["time_s"] is not None]
if summary_records:
    Ns = sorted({r["N"] for r in summary_records})
    Ps = sorted({r["P"] for r in summary_records})
    data = {(r["N"], r["P"]): r["time_s"] for r in summary_records}

    width = 0.18 if len(Ps) <= 4 else 0.8 / max(1, len(Ps))
    x_base = list(range(len(Ns)))

    plt.figure(figsize=(7, 4))
    for i, P in enumerate(Ps):
        offset = (i - (len(Ps) - 1) / 2) * width
        x_pos = [x + offset for x in x_base]
        values = [data.get((N, P), 0.0) for N in Ns]
        plt.bar(x_pos, values, width=width, label=f"P={P}")

    plt.xticks(x_base, [str(N) for N in Ns])
    plt.xlabel("Grid size N")
    plt.ylabel("Time (s)")
    plt.legend(title="MPI ranks", ncol=len(Ps), fontsize=8, title_fontsize=9)
    plt.tight_layout()
    plt.savefig("results/scaling/summary.png", dpi=200)
    plt.savefig("report/figures/scaling_summary.png", dpi=200)
    print("Wrote results/scaling/summary.png and report/figures/scaling_summary.png")
else:
    print("No T=1 scaling records found; skipping scaling summary plot.")

# OpenMP thread scaling for kernel timings (P=1, fixed N).
omp_source = omp_records if omp_records else scaling_records
omp_records = [r for r in omp_source if r["P"] == 1 and r["profile"] is not None]
if omp_records:
    target_N = 1024
    filtered = [r for r in omp_records if r["N"] == target_N and r["iters"]]
    filtered.sort(key=lambda r: r["T"])
    if filtered:
        threads = [r["T"] for r in filtered]
        spmv = [r["profile"]["spmv"] / r["iters"] for r in filtered]
        axpy = [r["profile"]["axpy"] / r["iters"] for r in filtered]

        plt.figure(figsize=(6, 4))
        plt.plot(threads, spmv, marker="o", label="SpMV")
        plt.plot(threads, axpy, marker="s", label="AXPY")
        plt.xticks(threads, [str(t) for t in threads])
        plt.xlabel("OpenMP threads")
        plt.ylabel("Kernel time per iteration (s)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("results/scaling/omp_kernels.png", dpi=200)
        plt.savefig("report/figures/omp_kernels.png", dpi=200)
        print("Wrote results/scaling/omp_kernels.png and report/figures/omp_kernels.png")
    else:
        print("No OpenMP kernel records found for N=1024; skipping kernel plot.")
else:
    print("No OpenMP kernel breakdowns found; skipping kernel plot.")

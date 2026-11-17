import argparse
import csv
import hashlib
import math
import os
import statistics
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from mnl_bwk import simulate


def configurations(horizons):
    cases = []
    for horizon in horizons:
        for epsilon in (0.01, 0.25):
            for algorithm in ("MNLwK-UCB", "AAGZ"):
                cases.append(("infinite", algorithm, horizon, epsilon, 1.0))
            for alpha in (0.0, -0.5, -0.75):
                cases.append(("sublinear", "MNLwK-UCB", horizon, epsilon, alpha))
        for q0 in (0.75, 0.5, 0.25, 0.1):
            cases.append(("scaling", "MNLwK-UCB", horizon, 0.25, q0))
    return cases


def run_replication(case, replication, master_seed):
    token = repr((master_seed, case, replication)).encode()
    seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
    result = simulate(*case, seed)
    result["replication"] = replication
    return result


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit_curve(x, y, basis):
    if len(x) < len(basis):
        return []
    design = [[function(value) for function in basis] for value in x]
    m = len(basis)
    matrix = [[sum(row[i] * row[j] for row in design) for j in range(m)]
              + [sum(row[i] * value for row, value in zip(design, y))] for i in range(m)]
    for i in range(m):
        pivot = max(range(i, m), key=lambda j: abs(matrix[j][i]))
        matrix[i], matrix[pivot] = matrix[pivot], matrix[i]
        if abs(matrix[i][i]) < 1e-12:
            return []
        scale = matrix[i][i]
        matrix[i] = [value / scale for value in matrix[i]]
        for j in range(m):
            if j != i:
                scale = matrix[j][i]
                matrix[j] = [a - scale * b for a, b in zip(matrix[j], matrix[i])]
    points = [min(x) + (max(x) - min(x)) * i / 100 for i in range(101)]
    return [(value, sum(matrix[j][-1] * basis[j](value) for j in range(m))) for value in points]


def make_plots(rows, directory):
    colors = ("blue!75!black", "orange!85!black", "teal!75!black", "violet!80!black", "red!75!black")
    markers = ("*", "square*", "triangle*", "diamond*", "pentagon*")
    header = r"""\begin{tikzpicture}
\begin{groupplot}[
group style={group size=2 by 1,horizontal sep=1.6cm},
width=.44\textwidth,height=6cm,
xlabel={Time horizon $T/10^5$},ylabel={Mean regret},
grid=major,grid style={gray!20},
tick label style={font=\small},label style={font=\small},
legend style={font=\scriptsize,draw=none,at={(.5,-.25)},anchor=north,legend columns=2}]
"""
    footer = "\\end{groupplot}\n\\end{tikzpicture}\n"

    def select(regime, algorithm, epsilon, parameter):
        return sorted((row for row in rows if (row["regime"], row["algorithm"],
                       row["epsilon"], row["parameter"]) == (regime, algorithm, epsilon, parameter)),
                      key=lambda row: row["T"])

    def plot(points, color, marker, label=None, basis=None):
        coordinates = " ".join(f"({x:.10g},{y:.10g}) +- (0,{se:.10g})" for x, y, se in points)
        text = (r"\addplot+[" + color + ",mark=" + marker
                + r",error bars/.cd,y dir=both,y explicit] coordinates {" + coordinates + "};\n")
        if label:
            text += r"\addlegendentry{" + label + "}\n"
        if basis:
            fit = fit_curve([p[0] for p in points], [p[1] for p in points], basis)
            if fit:
                coordinates = " ".join(f"({x:.10g},{y:.10g})" for x, y in fit)
                text += r"\addplot[" + color + r",dashed,no marks,forget plot] coordinates {" + coordinates + "};\n"
        return text

    def points(selected):
        return [(row["T"] / 1e5, row["mean_regret"], row["se_regret"]) for row in selected]

    for regime, filename in (("infinite", "infinite_inventory"), ("sublinear", "sublinear_inventory")):
        text = header
        for epsilon in (0.01, 0.25):
            text += r"\nextgroupplot[title={$\epsilon=" + str(epsilon) + "$}]\n"
            series = [("MNLwK-UCB", 1.0, "MNLwK-UCB"), ("AAGZ", 1.0, "AAGZ")] if regime == "infinite" else [
                ("MNLwK-UCB", alpha, r"$\alpha=" + f"{alpha:g}" + "$") for alpha in (0.0, -0.5, -0.75)]
            basis = (lambda x: 1.0, math.sqrt if epsilon == 0.01 else math.log)
            for j, (algorithm, parameter, label) in enumerate(series):
                text += plot(points(select(regime, algorithm, epsilon, parameter)), colors[j], markers[j], label, basis)
        (directory / (filename + ".tex")).write_text(text + footer)

    text = header + r"\nextgroupplot[title={Regret over the horizon}]" + "\n"
    terminal = []
    for j, q0 in enumerate((1.0, 0.75, 0.5, 0.25, 0.1)):
        selected = select("sublinear", "MNLwK-UCB", 0.25, 0.0) if q0 == 1 else select("scaling", "MNLwK-UCB", 0.25, q0)
        text += plot(points(selected), colors[j], markers[j], r"$q_0=" + f"{q0:g}" + "$")
        terminal.append((q0, selected[-1]["mean_regret"], selected[-1]["se_regret"]))
    terminal.sort()
    horizon = max(row["T"] for row in rows)
    text += r"\nextgroupplot[title={Regret at $T=" + str(horizon) + r"$},xlabel={Inventory scaling $q_0$}]" + "\n"
    text += plot(terminal, "black", "*", basis=(lambda x: 1.0, math.sqrt, lambda x: x))
    (directory / "uniform_scaling.tex").write_text(text + footer)
    document = r"""\documentclass{article}
\usepackage[a4paper,landscape,margin=2cm]{geometry}
\usepackage{pgfplots}
\pgfplotsset{compat=1.17}
\usepgfplotslibrary{groupplots}
\pagestyle{empty}
\begin{document}
"""
    for name, title in (("infinite_inventory", "Recovering the infinite inventory"),
                        ("sublinear_inventory", "Sublinear inventory growth"),
                        ("uniform_scaling", "Uniform inventory scaling")):
        document += r"\begin{center}\textbf{" + title + r"}\par\medskip" + "\n"
        document += r"\input{" + name + r".tex}\par\medskip" + "\n"
        document += f"Mean regret over {rows[0]['replications']} independent replications. "
        document += r"Error bars show one standard error.\end{center}\clearpage" + "\n"
    (directory / "plots.tex").write_text(document + "\\end{document}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(range(100000, 1000001, 100000)))
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    if args.replications < 2 or args.workers < 1 or min(args.horizons) < 2:
        parser.error("Use at least two replications, one worker, and horizons of at least two")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = configurations(sorted(set(args.horizons)))
    results = defaultdict(list)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        pending = {executor.submit(run_replication, case, replication, args.seed): case
                   for case in cases for replication in range(1, args.replications + 1)}
        with (args.output_dir / "replications.csv").open("w", newline="") as stream:
            writer = None
            for completed, future in enumerate(as_completed(pending), 1):
                row = future.result()
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
                results[pending[future]].append(row)
                if completed % max(1, len(pending) // 20) == 0 or completed == len(pending):
                    print(f"{completed}/{len(pending)} replications complete", flush=True)
    raw, summaries = [], []
    for case in cases:
        rows = sorted(results[case], key=lambda row: row["replication"])
        raw.extend(rows)
        regrets = [row["regret"] for row in rows]
        sd = statistics.stdev(regrets)
        summary = dict(zip(("regime", "algorithm", "T", "epsilon", "parameter"), case))
        summary.update(replications=len(rows), mean_regret=statistics.mean(regrets),
                       sd_regret=sd, se_regret=sd / math.sqrt(len(rows)),
                       mean_realized_regret=statistics.mean(row["realized_regret"] for row in rows),
                       benchmark=rows[0]["benchmark"], active_products=rows[0]["active_products"],
                       stockouts=sum(row["stop_time"] > 0 for row in rows),
                       mean_nonempty_offers=statistics.mean(row["nonempty_offers"] for row in rows),
                       mean_lp_solves=statistics.mean(row["lp_solves"] for row in rows))
        summaries.append(summary)
    write_csv(args.output_dir / "replications.csv", raw)
    write_csv(args.output_dir / "summary.csv", summaries)
    make_plots(summaries, args.output_dir)
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()

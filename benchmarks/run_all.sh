#!/bin/bash

# Create results directory if it doesn't exist
mkdir -p results

echo "Running KEEL FARC-HD Materialization Wall Benchmark..."
uv run run_keel_farchd.py > results/keel_farchd_oom_crash.log 2>&1
echo "Saved to results/keel_farchd_oom_crash.log"

echo "Running Synthetic Scaling Benchmark..."
uv run synthetic_scaling_benchmark.py > results/synthetic_scaling.log 2>&1
echo "Saved to results/synthetic_scaling.log"

echo "Running Extensive Bio-informatics Benchmark..."
uv run run_extensive_benchmark.py > results/extensive_benchmark.log 2>&1
echo "Saved to results/extensive_benchmark.log"

echo "Generating Materialization Wall Plot..."
uv run plot_materialization_wall.py > results/plot_materialization.log 2>&1
echo "Plot saved to ../../../../../articles/2027/bioinformatics-transparent-ml/materialization_wall.pdf"

echo "Generating Pareto Analysis Plot..."
uv run plot_pareto_only.py > results/plot_pareto.log 2>&1
echo "Plot saved to ../../../../../articles/2027/bioinformatics-transparent-ml/pareto_tradeoff.pdf"

echo "All benchmarks executed and persisted in results/ directory."

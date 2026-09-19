# Benchmarks

This directory contains benchmarking scripts for evaluating the performance, scaling limits, and pareto optimal properties of `transparent-ml` algorithms compared to their legacy Java counterparts.

## Reproducing the Materialization Wall (FARC-HD JVM OutOfMemory Proof)

The paper claims that Evolutionary Fuzzy Systems like FARC-HD natively crash when attempting to process 5,000 features due to the "Materialization Wall" of Apriori candidate generation. We prove this empirically using the original KEEL Java implementation.

1. Ensure the Java KEEL code is compiled in `../../references/KEEL/build`. (You can compile it via `javac -sourcepath src -d build src/keel/Algorithms/Associative_Classification/ClassifierFuzzyFARCHD/*.java` within the KEEL directory).
2. Run the `run_keel_farchd.py` script:
   ```bash
   uv run run_keel_farchd.py
   ```
3. This script will:
   - Generate a synthetic dataset with 5000 features.
   - Export it into the KEEL `.dat` format.
   - Invoke the compiled KEEL JVM natively with a strict 2GB heap limit.
   - Capture the resulting `java.lang.OutOfMemoryError` and save it to `results/keel_farchd_oom_crash.log`.

## Scaling Limits and Pareto Analysis

- `synthetic_scaling_benchmark.py`: Runs a systemic scale-up from 1,000 to 500,000 features on our native Python implementations (FURIA, PART, RuleFit, etc.) to empirically pinpoint their time complexity breaking points.
- `plot_materialization_wall.py`: Generates the scaling complexity figure used in the paper.
- `plot_pareto_only.py`: Generates the Pareto trade-off matrix between Predictive Quality (F1) and Model Complexity (Rule Count).
- `run_extensive_benchmark.py`: Contains the overarching benchmark harness for executing experiments across different biomedical datasets.

## Running all benchmarks

For convenience and guaranteed persistence of all logs, we provide a unified executable script:

```bash
./run_all.sh
```

This will automatically execute all scaling and empirical benchmarks, generate the `.pdf` plots (saved to the article's directory), and redirect all console outputs (including metrics and OOM crashes) directly into the `results/` folder.

## Results
All output logs and figures are exported into the `results/` directory, making the empirical tests fully reproducible for other agents and developers.

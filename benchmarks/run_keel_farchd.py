import os
import subprocess
from sklearn.datasets import make_classification

def generate_keel_dat(X, y, filepath, relation_name="synthetic"):
    with open(filepath, 'w') as f:
        f.write(f"@relation {relation_name}\n")
        n_features = X.shape[1]
        
        for i in range(n_features):
            f.write(f"@attribute f{i} real[0.0, 1.0]\n")
        f.write("@attribute class {0,1}\n")
        
        inputs = ", ".join([f"f{i}" for i in range(n_features)])
        f.write(f"@inputs {inputs}\n")
        f.write("@outputs class\n")
        f.write("@data\n")
        
        for i in range(X.shape[0]):
            row = ", ".join([f"{val:.6f}" for val in X[i]])
            f.write(f"{row}, {y[i]}\n")

def generate_config(config_path, data_path):
    with open(config_path, 'w') as f:
        f.write("algorithm = FARC-HD\n")
        f.write(f'inputData = "{data_path}" "{data_path}" "{data_path}"\n')
        f.write('outputData = "./train_out.dat" "./test_out.dat" "./db.txt" "./rb.txt"\n')
        f.write("seed = 1\n")
        f.write("Num_of_Labels = 3\n")
        f.write("Support = 0.05\n")
        f.write("Confidence = 0.8\n")
        f.write("Depth_Max = 3\n")
        f.write("K_parameter = 2\n")
        f.write("MAX_Evals = 1000\n")
        f.write("Pop_Size = 10\n")
        f.write("alpha = 0.02\n")
        f.write("Bits_Gen = 30\n")
        f.write("typeInference = 0\n")

def run():
    print("Generating synthetic data (D=5000)...")
    X, y = make_classification(n_samples=200, n_features=5000, n_informative=5, random_state=42)
    
    # Normalize features to [0, 1] as expected by KEEL Fuzzy partitioning
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
    
    dat_path = "synthetic_5000.dat"
    config_path = "farchd_config.txt"
    
    generate_keel_dat(X, y, dat_path)
    generate_config(config_path, dat_path)
    
    keel_build_dir = os.path.abspath("../../../../references/KEEL/build")
    
    cmd = [
        "java",
        # Set max heap size to 2GB to demonstrate the limit. 
        # FARC-HD should blow past this due to Apriori on 5000 features.
        "-Xmx2G",
        "-cp",
        keel_build_dir,
        "keel.Algorithms.Associative_Classification.ClassifierFuzzyFARCHD.Main",
        config_path
    ]
    
    print("Running KEEL Java FARC-HD...")
    print("Command:", " ".join(cmd))
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        print("\n--- STDOUT ---")
        print(result.stdout)
        print("\n--- STDERR ---")
        print(result.stderr)
        
        with open("results/keel_farchd_oom_crash.log", "w") as f:
            f.write("--- STDOUT ---\n")
            f.write(result.stdout)
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)
            
        print("\nResults saved to results/keel_farchd_oom_crash.log")
        
    except subprocess.TimeoutExpired as e:
        print(f"\nCommand timed out after {e.timeout} seconds.")
        if e.stdout: print("STDOUT:", e.stdout.decode('utf-8', errors='ignore'))
        if e.stderr: print("STDERR:", e.stderr.decode('utf-8', errors='ignore'))

if __name__ == "__main__":
    run()

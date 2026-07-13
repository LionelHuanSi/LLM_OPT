import os
import subprocess
import sys
import time

def run_command(cmd, shell=True):
    print(f"[EXEC] {cmd}")
    res = subprocess.run(cmd, shell=shell)
    return res.returncode

def main():
    print("=== VIETTEL AI RACE LOCAL BENCHMARK SIMULATOR ===")
    print("Step 1: Checking Docker container with resource constraints (--cpus=3 --memory=8g)...")
    
    # Check if vLLM server is responding on port 8000
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:8000/health")
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                print("[INFO] vLLM server is running and healthy on http://localhost:8000.")
            else:
                print(f"[WARNING] Healthcheck returned status {resp.status}.")
    except Exception as e:
        print("[WARNING] Could not reach http://localhost:8000/health directly.")
        print("Make sure your Docker container is up via: docker compose up -d")

    print("\nStep 2: Launching benchmark_ers.py...")
    ret = run_command(f"{sys.executable} benchmark_ers.py")
    if ret == 0:
        print("\n[SUCCESS] Local benchmark completed successfully!")
    else:
        print(f"\n[ERROR] Benchmark process returned exit code {ret}.")

if __name__ == "__main__":
    main()

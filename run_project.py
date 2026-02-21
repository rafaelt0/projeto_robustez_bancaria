import subprocess
import os
import sys

def run_script(script_path):
    print(f"\n>>> Running: {script_path}")
    try:
        subprocess.run([sys.executable, script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"!!! Error running {script_path}: {e}")
        return False
    return True

def main():
    print("="*50)
    print("BANKING ROBUSTNESS PROJECT ORCHESTRATOR")
    print("="*50)

    scripts = [
        "scripts/modelos/modelo_final_recomendado.py",
        "scripts/analise/econometrica_table.py",
        "scripts/analise/gerar_tabelas_latex.py"
    ]

    for script in scripts:
        if not os.path.exists(script):
            print(f"!!! Script not found: {script}")
            continue
        
        success = run_script(script)
        if not success:
            print("Stopping pipeline due to error.")
            break
    
    print("\n" + "="*50)
    print("PIPELINE EXECUTION FINISHED")
    print("="*50)

if __name__ == "__main__":
    main()

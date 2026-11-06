#!/usr/bin/env python3
"""Test v13_core (4 rotors + Stability) on 10 physics laws."""
import numpy as np
import time
import sys
import multiprocessing
sys.path.insert(0, '/data/data/com.termux/files/home/sadegh')
from v13_core import gp_core_v23, evaluate_tree_safe, tree_nodes

# ═══════════════════════════════════════════════════════════
# Physics benchmarks (extracted from discover_all_physics_v13)
# ═══════════════════════════════════════════════════════════
PHYSICS_LAWS = [
    {"name": "1. Schrödinger", "n_vars": 3, "gen": lambda X: (X[:,0]**2) / (2 * X[:,1] + 1e-9) + X[:,2], "vars": ["p", "m", "V"]},
    {"name": "2. de Broglie", "n_vars": 1, "gen": lambda X: 6.626 / (X[:,0] + 1e-9), "vars": ["p"]},
    {"name": "4. E=mc²", "n_vars": 1, "gen": lambda X: X[:,0] * 299792458**2, "vars": ["m"]},
    {"name": "5. Relativistic", "n_vars": 2, "gen": lambda X: np.sqrt((X[:,0]*3e8)**2 + (X[:,1]*3e8)**2), "vars": ["p", "m"]},
    {"name": "7. Harmonic", "n_vars": 2, "gen": lambda X: (X[:,0] + 0.5) * X[:,1], "vars": ["n", "hw"]},
    {"name": "8. Boltzmann", "n_vars": 2, "gen": lambda X: np.exp(-X[:,0] / (X[:,1] + 1e-9)), "vars": ["dE", "kT"]},
    {"name": "9. Planck", "n_vars": 1, "gen": lambda X: X[:,0]**4, "vars": ["T"]},
    {"name": "10. Condensation", "n_vars": 2, "gen": lambda X: (X[:,0] - X[:,1])**2, "vars": ["T", "Tc"]},
]

# ═══════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0: return 0.0
    return 1.0 - (ss_res / ss_tot)

def check_ram():
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) // 1024
    except:
        return 0

def stability_check(tree, X, y, features):
    n = len(X)
    split = int(n * 0.8)
    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:], y[split:]
    
    try:
        pred_tr = np.nan_to_num(evaluate_tree_safe(tree, X_tr, features), nan=0.0, posinf=1e6, neginf=-1e6)
        pred_te = np.nan_to_num(evaluate_tree_safe(tree, X_te, features), nan=0.0, posinf=1e6, neginf=-1e6)
        
        var_tr = np.sum((y_tr - np.mean(y_tr))**2)
        var_te = np.sum((y_te - np.mean(y_te))**2)
        
        r2_tr = 1 - np.sum((y_tr - pred_tr)**2) / (var_tr + 1e-9) if var_tr > 0 else 0
        r2_te = 1 - np.sum((y_te - pred_te)**2) / (var_te + 1e-9) if var_te > 0 else 0
        gap = abs(r2_tr - r2_te)
        
        return {"r2_train": r2_tr, "r2_test": r2_te, "gap": gap, "stable": gap < 0.15}
    except:
        return {"r2_train": 0, "r2_test": 0, "gap": 1.0, "stable": False}

# ═══════════════════════════════════════════════════════════
# Worker
# ═══════════════════════════════════════════════════════════
def worker(seed, X, y, features, generations, pop_size, max_nodes, queue):
    np.random.seed(seed)
    import random; random.seed(seed)
    try:
        tree, mse = gp_core_v23(X, y, features,
                                 generations=generations,
                                 pop_size=pop_size,
                                 max_nodes=max_nodes)
        queue.put({"seed": seed, "tree": tree, "mse": mse})
    except Exception as e:
        queue.put({"seed": seed, "error": str(e)})

# ═══════════════════════════════════════════════════════════
# Run one law
# ═══════════════════════════════════════════════════════════
def run_one_law(law, num_rotors=4, generations=30, pop_size=100, max_nodes=30):
    n_samples = 300
    X = np.random.uniform(0.5, 2.0, (n_samples, law['n_vars']))
    y = law['gen'](X)
    y_m, y_s = y.mean(), y.std()
    if y_s < 1e-9: y_s = 1.0
    y_norm = (y - y_m) / y_s
    features = law['vars']
    
    queue = multiprocessing.Queue()
    procs = []
    for i in range(num_rotors):
        seed = 42 + i * 13
        p = multiprocessing.Process(target=worker, args=(seed, X, y_norm, features, generations, pop_size, max_nodes, queue))
        procs.append(p)
        p.start()
    
    results = []
    for _ in range(num_rotors):
        try:
            results.append(queue.get(timeout=120))
        except:
            pass
    
    for p in procs:
        if p.is_alive(): p.terminate()
        p.join(timeout=3)
    
    # MCDM+Stability
    scored = []
    for res in results:
        if res.get("tree") is None: continue
        tree = res["tree"]
        try:
            pred = np.nan_to_num(evaluate_tree_safe(tree, X, y_norm, features) if False else evaluate_tree_safe(tree, X, features), nan=0.0, posinf=1e6, neginf=-1e6)
            var_y = np.sum((y_norm - np.mean(y_norm))**2)
            r2 = 1 - np.sum((y_norm - pred)**2) / (var_y + 1e-9) if var_y > 0 else 0
            
            stab = stability_check(tree, X, y_norm, features)
            nodes = tree_nodes(tree)
            simplicity = 1.0 / (1.0 + nodes / 20.0)
            stability_score = 1.0 - min(stab["gap"], 1.0)
            
            score = 0.50 * max(0, r2) + 0.25 * stability_score + 0.25 * simplicity
            scored.append({"tree": tree, "r2": r2, "nodes": nodes, "gap": stab["gap"], 
                           "stable": stab["stable"], "score": score, "seed": res.get("seed")})
        except:
            continue
    
    if not scored:
        return None
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[0]

# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
def main():
    print("="*65)
    print("v13 + MCDM+Stability on 10 PHYSICS LAWS")
    print("="*65)
    
    np.random.seed(42)
    ram = check_ram()
    print(f"💾 RAM: {ram} MB\n")
    
    results = []
    start = time.time()
    
    for law in PHYSICS_LAWS:
        print(f"🔬 {law['name']}...", end=" ")
        winner = run_one_law(law, num_rotors=4, generations=30, pop_size=100, max_nodes=30)
        if winner is None:
            print("❌ no tree")
            continue
        stable_flag = "✅" if winner["stable"] else "⚠️"
        print(f"{stable_flag} R²={winner['r2']:.4f} (nodes={winner['nodes']}, gap={winner['gap']:.3f})")
        results.append({"name": law['name'], "winner": winner})
    
    elapsed = time.time() - start
    
    print("\n" + "="*65)
    print("📊 SUMMARY")
    print("="*65)
    r2_values = []
    for r in results:
        w = r['winner']
        stable_flag = "✅" if w['stable'] else "⚠️"
        print(f"  {stable_flag} {r['name']:25s} | R²={w['r2']:.4f} | nodes={w['nodes']:2d} | gap={w['gap']:.3f}")
        r2_values.append(w['r2'])
    
    if r2_values:
        print(f"\n  Average R² = {np.mean(r2_values):.4f}")
        print(f"  Passed (R² > 0.7) = {sum(1 for v in r2_values if v > 0.7)}/{len(r2_values)}")
        print(f"  Total time = {elapsed:.1f}s")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()

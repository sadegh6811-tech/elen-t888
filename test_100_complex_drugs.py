import numpy as np
import random
import sys
import io
import os
import time
from contextlib import redirect_stdout

sys.path.insert(0, os.path.expanduser('~/Nazi'))

# ساکت کردن خروجی‌های اضافه موتور
f = io.StringIO()
with redirect_stdout(f):
    from lab_1024d_log_space import gp_core_single, evaluate_tree_safe


def generate_complex_drug(n_samples=150, D=128):
    """
    شبیه‌سازی یک داروی پیچیده با روابط غیرخطی تودرتو (شاخه‌های زیاد)
    """
    np.random.seed(None)  # هر بار دانه جدید
    X = np.random.uniform(0.1, 2.0, (n_samples, D))
    
    # رابطه پیچیده: ترکیب توان‌ها، لگاریتم‌ها، و تعاملات
    y = (
        np.sum(X[:, :32]**2, axis=1) + 
        0.5 * np.sum(X[:, 32:64]**3, axis=1) +
        np.log1p(np.abs(np.sum(X[:, 64:96], axis=1))) +
        0.3 * np.prod(X[:, 96:100], axis=1)
    )
    y += np.random.normal(0, 0.1, n_samples)
    return X, y


def safe_predict(tree, X, features, scaling, y_true):
    if tree is None:
        return -999.0
    try:
        pred = evaluate_tree_safe(tree, X, features)
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if isinstance(scaling, (tuple, list)) and len(scaling) == 2:
            a, b = scaling
        elif isinstance(scaling, (int, float)):
            a, b = scaling, 0.0
        else:
            a, b = 1.0, 0.0
            
        y_pred = a * pred + b
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - y_true.mean())**2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    except:
        return -999.0


def parse_output(result):
    if result is None:
        return None, (1.0, 0.0)
    if isinstance(result, tuple):
        if len(result) >= 3:
            return result[0], result[2]
        elif len(result) >= 1:
            return result[0], (1.0, 0.0)
    return result, (1.0, 0.0)


def run_100_drugs():
    print("="*65)
    print("🧬 100 COMPLEX DRUGS BENCHMARK (Hybrid Architecture)")
    print("="*65)
    print("Simulating 100 drugs with 128 descriptors and nested branches...")
    
    results = {
        'chem_r2': [], 'hybrid_r2': [],
        'chem_fails': 0, 'hybrid_rescues': 0, 'total_improvement': 0
    }
    
    start_total = time.time()
    
    for i in range(1, 101):
        X, y = generate_complex_drug()
        split = int(len(X) * 0.8)
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]
        
        features = [f"x{j}" for j in range(X.shape[1])]
        
        # --- فاز ۱: موتور شیمی (داده خام) ---
        r2_chem = -999.0
        try:
            res = gp_core_single(X_tr, y_tr, features, generations=10, pop_size=80, max_nodes=20)
            tree, scale = parse_output(res)
            r2_chem = safe_predict(tree, X_te, features, scale, y_te)
        except:
            r2_chem = -999.0
            
        # --- فاز ۲: مسیریابی به موتور ریاضی ---
        r2_hybrid = r2_chem
        
        if r2_chem < 0.85:
            results['chem_fails'] += 1
            try:
                # فشرده‌سازی با لگاریتم
                chunk = X.shape[1] // 8
                log_feats = []
                for k in range(8):
                    group_sum = np.sum(X[:, k*chunk:(k+1)*chunk], axis=1) + 1.0
                    log_feats.append(np.log(group_sum))
                X_c = np.column_stack(log_feats)
                f_c = [f"log_g{k}" for k in range(8)]
                
                X_tr_c, X_te_c = X_c[:split], X_c[split:]
                res_math = gp_core_single(X_tr_c, y_tr, f_c, generations=15, pop_size=100, max_nodes=30)
                tree_m, scale_m = parse_output(res_math)
                r2_math = safe_predict(tree_m, X_te_c, f_c, scale_m, y_te)
                
                if r2_math > r2_chem:
                    r2_hybrid = r2_math
                    results['hybrid_rescues'] += 1
                    results['total_improvement'] += (r2_math - r2_chem)
            except:
                pass
                
        results['chem_r2'].append(r2_chem)
        results['hybrid_r2'].append(r2_hybrid)
        
        if i % 10 == 0:
            print(f"   [{i}/100] Processed... (Chem fails: {results['chem_fails']}, Rescues: {results['hybrid_rescues']})")
            
    elapsed = time.time() - start_total
    
    # --- آمار نهایی ---
    chem_valid = [r for r in results['chem_r2'] if r > -900]
    hybrid_valid = [r for r in results['hybrid_r2'] if r > -900]
    
    print("\n" + "="*65)
    print("🏆 FINAL RESULTS (100 Complex Drugs)")
    print("="*65)
    print(f"⏱️ Total Time: {elapsed:.1f}s (Avg: {elapsed/100:.1f}s per drug)")
    print(f"\n🧪 Chemistry Engine Alone:")
    print(f"   Mean R² (on valid): {np.mean(chem_valid) if chem_valid else 0:.4f}")
    print(f"   Total Failures: {results['chem_fails']}/100")
    print(f"\n🧬 Hybrid Architecture:")
    print(f"   Mean R²: {np.mean(hybrid_valid) if hybrid_valid else 0:.4f}")
    print(f"   Successful Rescues: {results['hybrid_rescues']}/{results['chem_fails']}")
    print(f"   Total Improvement: +{results['total_improvement']:.2f} R² points")
    
    if results['chem_fails'] > 0:
        avg_improvement = results['total_improvement'] / results['chem_fails']
        print(f"   Average Rescue Boost: +{avg_improvement:.4f} R² per rescued drug")
    
    print("="*65)


if __name__ == "__main__":
    run_100_drugs()

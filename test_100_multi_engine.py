import numpy as np
import random
import sys
import io
import os
import time
from contextlib import redirect_stdout

sys.path.insert(0, os.path.expanduser('~/Nazi'))

f = io.StringIO()
with redirect_stdout(f):
    from lab_1024d_log_space import gp_core_single, evaluate_tree_safe


def generate_complex_drug(n_samples=150, D=128):
    np.random.seed(None)
    X = np.random.uniform(0.1, 2.0, (n_samples, D))
    y = (
        np.sum(X[:, :32]**2, axis=1) + 
        0.5 * np.sum(X[:, 32:64]**3, axis=1) +
        np.log1p(np.abs(np.sum(X[:, 64:96], axis=1))) +
        0.3 * np.prod(X[:, 96:100], axis=1)
    )
    y += np.random.normal(0, 0.1, n_samples)
    return X, y


def safe_predict(tree, X, features, scaling, y_true):
    if tree is None: return -999.0
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
    if result is None: return None, (1.0, 0.0)
    if isinstance(result, tuple):
        if len(result) >= 3: return result[0], result[2]
        elif len(result) >= 1: return result[0], (1.0, 0.0)
    return result, (1.0, 0.0)


def pca_reduce(X, n_components=8):
    """کاهش ابعاد با PCA (بدون sklearn)"""
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1][:n_components]
    return X_centered @ eigvecs[:, idx]


def run_multi_engine_test():
    print("="*65)
    print("🧬⚛️🧮 MULTI-ENGINE HYBRID BENCHMARK (100 Complex Drugs)")
    print("="*65)
    
    chem_r2s, hybrid_r2s = [], []
    engine_wins = {'log': 0, 'pca': 0, 'v31_raw': 0}
    
    start_total = time.time()
    
    for i in range(1, 101):
        X, y = generate_complex_drug()
        split = int(len(X) * 0.8)
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]
        
        features = [f"x{j}" for j in range(X.shape[1])]
        D = X.shape[1]
        
        # --- موتور شیمی (بدون ترفند) ---
        r2_chem = -999.0
        try:
            res = gp_core_single(X_tr, y_tr, features, generations=10, pop_size=80, max_nodes=20)
            tree, scale = parse_output(res)
            r2_chem = safe_predict(tree, X_te, features, scale, y_te)
        except: pass
        
        chem_r2s.append(r2_chem)
        best_hybrid_r2 = r2_chem
        best_engine = "chem"
        
        # اگر شیمی شکست خورد، هر ۳ موتور ریاضی را اجرا کن
        if r2_chem < 0.85:
            # === موتور ۱: Log-Space ===
            try:
                chunk = D // 8
                log_feats = []
                for k in range(8):
                    group_sum = np.sum(X[:, k*chunk:(k+1)*chunk], axis=1) + 1.0
                    log_feats.append(np.log(group_sum))
                X_c = np.column_stack(log_feats)
                f_c = [f"log_g{k}" for k in range(8)]
                
                X_tr_c, X_te_c = X_c[:split], X_c[split:]
                res_log = gp_core_single(X_tr_c, y_tr, f_c, generations=15, pop_size=100, max_nodes=30)
                tree_l, scale_l = parse_output(res_log)
                r2_log = safe_predict(tree_l, X_te_c, f_c, scale_l, y_te)
                
                if r2_log > best_hybrid_r2:
                    best_hybrid_r2 = r2_log
                    best_engine = "log"
            except: pass
            
            # === موتور ۲: PCA ===
            try:
                X_pca = pca_reduce(X, n_components=8)
                f_p = [f"pca{k}" for k in range(8)]
                X_tr_p, X_te_p = X_pca[:split], X_pca[split:]
                
                res_pca = gp_core_single(X_tr_p, y_tr, f_p, generations=15, pop_size=100, max_nodes=30)
                tree_p, scale_p = parse_output(res_pca)
                r2_pca = safe_predict(tree_p, X_te_p, f_p, scale_p, y_te)
                
                if r2_pca > best_hybrid_r2:
                    best_hybrid_r2 = r2_pca
                    best_engine = "pca"
            except: pass
            
            # === موتور ۳: V31 Raw (روی داده خام با نسل بالا) ===
            try:
                res_raw = gp_core_single(X_tr, y_tr, features, generations=20, pop_size=100, max_nodes=35)
                tree_r, scale_r = parse_output(res_raw)
                r2_raw = safe_predict(tree_r, X_te, features, scale_r, y_te)
                
                if r2_raw > best_hybrid_r2:
                    best_hybrid_r2 = r2_raw
                    best_engine = "v31_raw"
            except: pass
            
            if best_engine != "chem":
                engine_wins[best_engine] += 1
        
        hybrid_r2s.append(best_hybrid_r2)
        
        if i % 10 == 0:
            print(f"   [{i}/100] Chem Fails: {sum(1 for r in chem_r2s if r < 0.85)}, Best Hybrid R²: {np.mean(hybrid_r2s):.3f}")
            
    elapsed = time.time() - start_total
    
    # --- آمار نهایی ---
    chem_valid = [r for r in chem_r2s if r > -900]
    hybrid_valid = [r for r in hybrid_r2s if r > -900]
    
    print("\n" + "="*65)
    print("🏆 FINAL RESULTS (Multi-Engine Hybrid)")
    print("="*65)
    print(f"⏱️ Total Time: {elapsed:.1f}s (Avg: {elapsed/100:.1f}s per drug)")
    print(f"\n🧪 Chemistry Engine Alone:")
    print(f"   Mean R² (on valid): {np.mean(chem_valid) if chem_valid else 0:.4f}")
    print(f"\n🧬 Multi-Engine Hybrid:")
    print(f"   Mean R²: {np.mean(hybrid_valid) if hybrid_valid else 0:.4f}")
    print(f"\n🏅 Engine Win Distribution (Rescues):")
    total_rescues = sum(engine_wins.values())
    if total_rescues > 0:
        for k, v in engine_wins.items():
            print(f"   {k}: {v}/{total_rescues} ({v/total_rescues*100:.1f}%)")
    print("="*65)


if __name__ == "__main__":
    run_multi_engine_test()

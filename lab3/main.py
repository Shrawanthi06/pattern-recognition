import os
import sys
import math

SCRIPT_DIR=os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0,SCRIPT_DIR)

from src.preprocessing import train_val_test_split
from src.regression import PolynomialRegression, RidgePolynomialRegression
from src.metrics import mean_squared_error,root_mean_squared_error,mean_absolute_error,r2_score
from src.visualization import (plot_fitted_curves,plot_degree_vs_error,plot_regularization_comparison,plot_residuals,plot_learning_curve)

INPUT_DIR=os.path.join(SCRIPT_DIR,"input")
OUTPUT_DIR=os.path.join(SCRIPT_DIR,"output")
os.makedirs(OUTPUT_DIR,exist_ok=True)

def load_data(filename):
    x = []
    y = []
    with open(filename,"r") as file:
        for line in file:
            line=line.strip()
            if not line:
                continue
            parts=line.split()
            if len(parts) != 2:
                continue
            x.append(float(parts[0]))
            y.append(float(parts[1]))
    return x, y

def print_separator(title):
    print("\n"+"="*78)
    print(f" {title}")
    print("=" * 78)

def main():
    print_separator("POLYNOMIAL REGRESSION & MODEL SELECTION")   
    #load dataset
    data_path=os.path.join(INPUT_DIR,"noisy_3.txt")
    print(f"Loading dataset from: {data_path}")
    x_all,y_all=load_data(data_path)
    N=len(x_all)
    print(f"Total samples loaded: {N}")
    print(f"x range:[{min(x_all):.4f}, {max(x_all):.4f}]")
    print(f"y range:[{min(y_all):.4f}, {max(y_all):.4f}]")

    #Train / Validation / Test Split (70% / 15% / 15%)
    SEED = 42
    (x_train,y_train), (x_val,y_val), (x_test,y_test) = train_val_test_split(x_all, y_all, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=SEED)
    print_separator(f"DATASET SPLITTING (Seed={SEED})")
    print(f"Training Set (70%): {len(x_train)} samples")
    print(f"Validation Set (15%): {len(x_val)} samples")
    print(f"Test Set (15%): {len(x_test)} samples")

    #polynomial degree search (d = 1 to 10)
    print_separator("EFFECT OF POLYNOMIAL DEGREE")
    degrees=list(range(1, 11))
    results=[]
    models={}

    print(f"{'Deg':<4} | {'Train MSE':<10} | {'Val MSE':<10} | {'Test MSE':<10} | {'Train R2':<9} | {'Val R2':<9} | {'Test R2':<9}")
    print("-" * 78)

    for d in degrees:
        model=PolynomialRegression(degree=d,use_scaling=True)
        model.fit(x_train,y_train)
        models[d]=model

        #evaluate on all splits
        pred_train=model.predict(x_train)
        pred_val=model.predict(x_val)
        pred_test=model.predict(x_test)
        train_mse=mean_squared_error(y_train,pred_train)
        val_mse=mean_squared_error(y_val,pred_val)
        test_mse=mean_squared_error(y_test,pred_test)
        train_r2=r2_score(y_train,pred_train)
        val_r2=r2_score(y_val,pred_val)
        test_r2=r2_score(y_test,pred_test)
        results.append({
            "degree":d,
            "train_mse":train_mse,
            "val_mse":val_mse,
            "test_mse":test_mse,
            "train_r2":train_r2,
            "val_r2":val_r2,
            "test_r2":test_r2,
        })

        print(f"{d:<4} | {train_mse:<10.4f} | {val_mse:<10.4f} | {test_mse:<10.4f} | {train_r2:<9.4f} | {val_r2:<9.4f} | {test_r2:<9.4f}")

    #select optimal degree based on validation MSE
    best_res=min(results,key=lambda item: item["val_mse"])
    best_deg=best_res["degree"]
    best_model=models[best_deg]

    print_separator("OPTIMAL MODEL SELECTION")
    print(f"Optimal Polynomial Degree chosen by Validation Set: d*={best_deg}")
    print(f"Validation Performance -> MSE: {best_res['val_mse']:.4f} | RMSE: {math.sqrt(best_res['val_mse']):.4f} | R2: {best_res['val_r2']:.4f}")
    print(f"Final Test Performance -> MSE: {best_res['test_mse']:.4f} | RMSE: {math.sqrt(best_res['test_mse']):.4f} | R2: {best_res['test_r2']:.4f}")

    #Generate Core Visualizations
    print_separator("GENERATING PLOTS")   
    #fitted curves plot for degrees 1,2,3,5,8
    key_degrees = [d for d in [1,2,3,5,8] if d in models]
    plot_models = {f"Degree {d}": models[d] for d in key_degrees}
    fitted_plot_path = os.path.join(OUTPUT_DIR,"fitted_polynomial_curves.png")
    plot_fitted_curves(x_all, y_all,plot_models,fitted_plot_path,title="Polynomial Regression Fits on Noisy Observations")
    print(f"[Saved] {fitted_plot_path}")

    #degree vs error curve
    train_mses=[r["train_mse"] for r in results]
    val_mses=[r["val_mse"] for r in results]
    err_plot_path=os.path.join(OUTPUT_DIR, "degree_vs_error.png")
    plot_degree_vs_error(degrees,train_mses,val_mses,err_plot_path)
    print(f"[Saved] {err_plot_path}")

    #residual distribution plot
    test_preds=best_model.predict(x_test)
    res_plot_path=os.path.join(OUTPUT_DIR,"residual_distribution.png")
    plot_residuals(y_test,test_preds,res_plot_path)
    print(f"[Saved] {res_plot_path}")

    #ridge regularization
    print_separator("RIDGE (L2) REGULARIZATION STUDY (d=10)")
    high_deg=10
    unreg_model=models[high_deg]
    alphas=[0.1,10.0,1000.0]
    ridge_models={}
    print(f"{'Model':<25} | {'Val MSE':<10} | {'Test MSE':<10}")
    print("-" * 50)
    print(f"{'Unregularized (deg=10)':<25} | {results[high_deg-1]['val_mse']:<10.4f} | {results[high_deg-1]['test_mse']:<10.4f}")  
    for a in alphas:
        ridge=RidgePolynomialRegression(degree=high_deg,alpha=a,use_scaling=True)
        ridge.fit(x_train,y_train)
        r_val_mse=mean_squared_error(y_val,ridge.predict(x_val))
        r_test_mse=mean_squared_error(y_test,ridge.predict(x_test))
        label=f"Ridge (alpha={a})"
        ridge_models[label]=ridge
        print(f"{label:<25} | {r_val_mse:<10.4f} | {r_test_mse:<10.4f}")
    ridge_plot_path=os.path.join(OUTPUT_DIR,"regularization_comparison.png")
    plot_regularization_comparison(x_all,y_all,unreg_model,ridge_models,ridge_plot_path)
    print(f"[Saved] {ridge_plot_path}")

    #seed sensitivity & robustness analysis
    print_separator("SEED ROBUSTNESS ANALYSIS (5 Seeds)")
    test_seeds=[42, 100, 7, 2024, 999]
    seed_test_mses=[]
    print(f"{'Random Seed':<12} | {'Optimal Degree':<15} | {'Test MSE':<10} | {'Test R2':<10}")
    print("-" * 55)
    for s in test_seeds:
        (x_tr, y_tr),(x_v, y_v),(x_te, y_te)=train_val_test_split(x_all,y_all,0.70,0.15,0.15,random_seed=s)
        s_model=PolynomialRegression(degree=best_deg,use_scaling=True)
        s_model.fit(x_tr, y_tr)
        s_test_preds=s_model.predict(x_te)
        s_mse=mean_squared_error(y_te,s_test_preds)
        s_r2=r2_score(y_te,s_test_preds)
        seed_test_mses.append(s_mse)
        print(f"{s:<12} | {best_deg:<15} | {s_mse:<10.4f} | {s_r2:<10.4f}")
    mean_seed_mse=sum(seed_test_mses)/len(seed_test_mses)
    std_seed_mse=math.sqrt(sum((val-mean_seed_mse)**2 for val in seed_test_mses)/len(seed_test_mses))
    print(f"\nRobustness across 5 seeds: Test MSE = {mean_seed_mse:.4f} +/- {std_seed_mse:.4f}")

    #learning Curves (sample size vs error)
    print_separator("LEARNING CURVE ANALYSIS")
    sample_fractions=[0.01,0.05,0.10,0.25,0.50,0.75,1.0]
    lc_sizes=[]
    lc_train_mses=[]
    lc_val_mses=[]    
    for frac in sample_fractions:
        sub_size=max(10,int(len(x_train)*frac))
        sub_x_tr=x_train[:sub_size]
        sub_y_tr=y_train[:sub_size]      
        lc_model=PolynomialRegression(degree=best_deg,use_scaling=True)
        lc_model.fit(sub_x_tr,sub_y_tr)      
        tr_err=mean_squared_error(sub_y_tr,lc_model.predict(sub_x_tr))
        v_err=mean_squared_error(y_val,lc_model.predict(x_val))       
        lc_sizes.append(sub_size)
        lc_train_mses.append(tr_err)
        lc_val_mses.append(v_err)
    lc_plot_path=os.path.join(OUTPUT_DIR,"learning_curves.png")
    plot_learning_curve(lc_sizes,lc_train_mses,lc_val_mses,lc_plot_path)
    print(f"[Saved] {lc_plot_path}")

    #noise Level Estimation
    residuals=[yt-yp for yt, yp in zip(y_test, test_preds)]
    mean_noise=sum(residuals)/len(residuals)
    noise_std=math.sqrt(sum((r-mean_noise)**2 for r in residuals)/len(residuals))
    print_separator("ESTIMATED OBSERVATION NOISE")
    print(f"Noise Mean (epsilon): {mean_noise:.4f} (Ideal: 0.0)")
    print(f"Estimated Noise Standard Deviation (sigma_noise): {noise_std:.4f}")
    print(f"Theoretical Minimum Attainable MSE (sigma^2): {noise_std**2:.4f}")
    print_separator("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()

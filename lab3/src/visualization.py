import matplotlib.pyplot as plt

def plot_fitted_curves(x_data,y_data,models_dict,output_path,title="Polynomial Regression Fits"):
    plt.figure(figsize=(12,7))  
    # Subsample data points for scatter plot so it renders cleanly and fast
    step=max(1,len(x_data)//1000)
    plt.scatter(x_data[::step], y_data[::step],color="gray",alpha=0.35,s=12,label="Noisy Observations") 
    #generates smooth x line spanning the full range
    min_x, max_x=min(x_data), max(x_data)
    smooth_x=[min_x+i*(max_x-min_x)/500 for i in range(501)]
    
    colors=["red","blue","green","pink","orange","brown"]
    for idx,(label,model) in enumerate(models_dict.items()):
        preds=model.predict(smooth_x)
        color=colors[idx % len(colors)]
        plt.plot(smooth_x,preds,label=label,linewidth=2.2,color=color)    
    plt.title(title,fontsize=14,fontweight="bold")
    plt.xlabel("Input Variable (x)",fontsize=12)
    plt.ylabel("Target Output (y)",fontsize=12)
    plt.grid(True,linestyle="--",alpha=0.6)
    plt.legend(loc="best",fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path,dpi=300)
    plt.close()


def plot_degree_vs_error(degrees, train_mses, val_mses, output_path):
    #train MSE and validation MSE vs polynomial degree to identify the optimal model
    plt.figure(figsize=(10, 6))
    plt.plot(degrees,train_mses,marker="o",linewidth=2.0,color="blue",label="Training MSE")
    plt.plot(degrees,val_mses,marker="s",linewidth=2.0,color="red",label="Validation MSE")
    # Highlight best validation degree
    best_deg_idx=val_mses.index(min(val_mses))
    best_deg=degrees[best_deg_idx]
    best_val_mse=val_mses[best_deg_idx] 
    plt.scatter([best_deg], [best_val_mse],color="gold",edgecolor="black",s=180,zorder=5,label=f"Optimal Degree (d={best_deg}, MSE={best_val_mse:.4f})")  
    plt.title("Bias-Variance Tradeoff: Polynomial Degree vs MSE",fontsize=14,fontweight="bold")
    plt.xlabel("Polynomial Degree (d)", fontsize=12)
    plt.ylabel("Mean Squared Error (MSE)", fontsize=12)
    plt.xticks(degrees)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_regularization_comparison(x_data, y_data,unregularized_model,ridge_models_dict,output_path):
    #compares high-degree unregularized polynomial with ridge regularized polynomials
    plt.figure(figsize=(12,7))
    
    step =max(1,len(x_data)//1000)
    plt.scatter(x_data[::step],y_data[::step],color="lightgray",alpha=0.4,s=12,label="Observations")
    
    min_x,max_x= min(x_data),max(x_data)
    smooth_x= [min_x+i*(max_x-min_x)/500 for i in range(501)]
    
    plt.plot(smooth_x,unregularized_model.predict(smooth_x),label=f"Unregularized (deg={unregularized_model.degree})",color="red",linestyle="--", linewidth=2)            
    colors=["green","blue","pink"]
    for idx,(label,model) in enumerate(ridge_models_dict.items()):
        preds=model.predict(smooth_x)
        plt.plot(smooth_x, preds,label=label,linewidth=2.2,color=colors[idx % len(colors)])    
    plt.title("Effect of Ridge (L2) Regularization on High-Degree Polynomial",fontsize=14,fontweight="bold")
    plt.xlabel("Input (x)",fontsize=12)
    plt.ylabel("Target (y)",fontsize=12)
    plt.grid(True,linestyle="--",alpha=0.6)
    plt.legend(loc="best",fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path,dpi=300)
    plt.close()


def plot_residuals(y_true, y_pred, output_path):
    residuals=[yt - yp for yt, yp in zip(y_true, y_pred)]   
    fig,(ax1, ax2) = plt.subplots(1,2,figsize=(14, 5))
    
    #residuals vs predicted
    ax1.scatter(y_pred[::5],residuals[::5],alpha=0.3,color="blue",s=10)
    ax1.axhline(0,color="red",linestyle="--",linewidth=1.5)
    ax1.set_title("Residuals vs Predicted Values",fontsize=12,fontweight="bold")
    ax1.set_xlabel("Predicted Target",fontsize=10)
    ax1.set_ylabel("Residual (Actual - Predicted)",fontsize=10)
    ax1.grid(True, linestyle="--",alpha=0.6)
    
    #residuals
    ax2.hist(residuals,bins=50,color="green",edgecolor="black",alpha=0.7,density=True)
    ax2.set_title("Residual Error Distribution (Noise Analysis)",fontsize=12,fontweight="bold")
    ax2.set_xlabel("Residual Value",fontsize=10)
    ax2.set_ylabel("Density",fontsize=10)
    ax2.grid(True,linestyle="--",alpha=0.6)  
    plt.tight_layout()
    plt.savefig(output_path,dpi=300)
    plt.close()

def plot_learning_curve(train_sizes,train_mses,val_mses,output_path):
    #error vs training sample size
    plt.figure(figsize=(10,6))
    plt.plot(train_sizes,train_mses,marker="o",color="blue",label="Training MSE",linewidth=2)
    plt.plot(train_sizes,val_mses,marker="s",color="red",label="Validation MSE",linewidth=2) 
    plt.yscale("log")
    plt.title("Learning Curves (MSE vs Training Set Size)",fontsize=14,fontweight="bold")
    plt.xlabel("Number of Training Samples",fontsize=12)
    plt.ylabel("Mean Squared Error (MSE, log scale)",fontsize=12)
    plt.grid(True, which="both", linestyle="--",alpha=0.6)
    plt.legend(loc="best",fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path,dpi=300)
    plt.close()

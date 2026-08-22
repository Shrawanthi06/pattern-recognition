import math

def mean_squared_error(y_true, y_pred):
    #Compute Mean Squared Error (MSE): (1 / N) * sum((y_true - y_pred)^2)
    n=len(y_true)
    if n!=len(y_pred) or n==0:
        raise ValueError("y_true and y_pred must have the same non-zero length.")
    squared_errors=[(yt-yp)**2 for yt, yp in zip(y_true, y_pred)]
    return sum(squared_errors) / n

def root_mean_squared_error(y_true, y_pred):
    #Compute Root Mean Squared Error (RMSE): sqrt(MSE)
    return math.sqrt(mean_squared_error(y_true, y_pred))

def mean_absolute_error(y_true, y_pred):
    #Compute Mean Absolute Error (MAE): (1 / N) * sum(|y_true - y_pred|)
    n=len(y_true)
    if n!=len(y_pred) or n==0:
        raise ValueError("y_true and y_pred must have the same non-zero length.")
    abs_errors=[abs(yt-yp) for yt, yp in zip(y_true, y_pred)]
    return sum(abs_errors)/n

def r2_score(y_true, y_pred):
    #Compute Coefficient of Determination (R^2 Score): 1 - (SS_res / SS_tot)
    #SS_res = sum((y_true - y_pred)^2)
    #SS_tot = sum((y_true - mean(y_true))^2)
    n=len(y_true)
    if n!=len(y_pred) or n==0:
        raise ValueError("y_true and y_pred must have the same non-zero length.")
    mean_y=sum(y_true)/n
    ss_res=sum((yt-yp)**2 for yt, yp in zip(y_true, y_pred))
    ss_tot=sum((yt-mean_y)**2 for yt in y_true)
    if ss_tot < 1e-14:
        return 1.0 if ss_res < 1e-14 else 0.0
    return 1.0-(ss_res/ss_tot)

if __name__ == "__main__":
    print("Testing Metrics Module...")
    y_actual =[10.0,20.0,30.0,40.0]
    y_forecast =[12.0,19.0,29.0,42.0]

    mse=mean_squared_error(y_actual,y_forecast)
    rmse=root_mean_squared_error(y_actual,y_forecast)
    mae=mean_absolute_error(y_actual,y_forecast)
    r2=r2_score(y_actual,y_forecast)
    print(f"MSE:{mse:.4f}")
    print(f"RMSE:{rmse:.4f}")
    print(f"MAE:{mae:.4f}")
    print(f"R2:{r2:.4f}")
    print("Expected MSE: 4.5, RMSE: 2.1213, MAE: 1.75, R2: 0.9875")

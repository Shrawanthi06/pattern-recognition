import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from src.linear_algebra import transpose,matmul,matvec,solve_linear_system,identity
from src.preprocessing import StandardScaler,create_polynomial_features

class PolynomialRegression:
    def __init__(self,degree=1,use_scaling=True):
        self.degree=degree
        self.use_scaling=use_scaling
        self.scaler=StandardScaler() if use_scaling else None
        self.theta=None

    def fit(self,x,y):
        #feature scaling 
        if self.use_scaling:
            x_proc=self.scaler.fit_transform(x)
        else:
            x_proc=x

        #construct Vandermonde design matrix X (N x (degree + 1))
        X=create_polynomial_features(x_proc, self.degree)

        #compute X^T
        X_T=transpose(X)

        #compute X^T * X and X^T * y
        XTX=matmul(X_T, X)
        XTy=matvec(X_T, y)

        #solve (X^T X) * theta = X^T y
        self.theta=solve_linear_system(XTX,XTy)
        return self

    def predict(self,x):
        if self.theta is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        if self.use_scaling:
            x_proc=self.scaler.transform(x)
        else:
            x_proc=x
        X=create_polynomial_features(x_proc,self.degree)
        return matvec(X,self.theta)


class RidgePolynomialRegression:
    def __init__(self,degree=1,alpha=1.0,use_scaling=True):
        self.degree =degree
        self.alpha =float(alpha)  # Regularization parameter (lambda)
        self.use_scaling =use_scaling
        self.scaler =StandardScaler() if use_scaling else None
        self.theta =None

    def fit(self,x,y):
        if self.use_scaling:
            x_proc=self.scaler.fit_transform(x)
        else:
            x_proc=x

        X =create_polynomial_features(x_proc,self.degree)
        X_T =transpose(X)
        XTX =matmul(X_T,X)
        XTy =matvec(X_T,y)

        k = self.degree + 1
        # Add regularization penalty alpha * I' to XTX
        # Do not penalize w_0 (bias term)
        for i in range(1, k):
            XTX[i][i] += self.alpha
        self.theta = solve_linear_system(XTX, XTy)
        return self

    def predict(self,x):
        if self.theta is None:
            raise RuntimeError("Model must be fitted before calling predict().")

        if self.use_scaling:
            x_proc=self.scaler.transform(x)
        else:
            x_proc=x

        X=create_polynomial_features(x_proc,self.degree)
        return matvec(X,self.theta)


if __name__ == "__main__":
    print("Testing Regression Module...")
    
    # Synthetic quadratic data: y = 3 + 2x + 5x^2
    train_x = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    train_y = [3.0 + 2.0 * xi + 5.0 * (xi ** 2) for xi in train_x]
    
    # Fit degree 2 model
    model = PolynomialRegression(degree=2, use_scaling=False)
    model.fit(train_x, train_y)
    
    print("Fitted Coefficients theta [theta0, theta1, theta2]:")
    print(" ", [round(t, 4) for t in model.theta])
    print(" Expected: [3.0, 2.0, 5.0]")
    
    preds=model.predict([4.0])
    print(f"Prediction at x=4.0: {preds[0]:.4f} (Expected: 91.0000)")
    
    # Fit Ridge model
    ridge=RidgePolynomialRegression(degree=2, alpha=0.1, use_scaling=False)
    ridge.fit(train_x, train_y)
    print(f"Ridge Coefficients: {[round(t, 4) for t in ridge.theta]}")
    print("All Regression tests passed successfully!")

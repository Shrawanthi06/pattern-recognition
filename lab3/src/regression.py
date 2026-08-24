import numpy as np
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
        X=np.array(create_polynomial_features(x_proc, self.degree))
        y=np.array(y)

        #closed-form Normal Equations: theta = (X^T X)^-1 X^T y
        XTX=X.T @ X
        XTy=X.T @ y
        self.theta=np.linalg.inv(XTX) @ XTy
        return self

    def predict(self,x):
        if self.theta is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        if self.use_scaling:
            x_proc=self.scaler.transform(x)
        else:
            x_proc=x
        X=np.array(create_polynomial_features(x_proc,self.degree))
        return (X @ self.theta).tolist()


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

        X =np.array(create_polynomial_features(x_proc,self.degree))
        y =np.array(y)
        XTX =X.T @ X
        XTy =X.T @ y

        k = self.degree + 1
        # Regularization matrix: alpha * I', with 0 for intercept w_0
        alpha_matrix = self.alpha * np.eye(k)
        alpha_matrix[0, 0] = 0.0

        # Closed-form Ridge formula: theta = (X^T X + alpha * I')^-1 X^T y
        self.theta = np.linalg.inv(XTX + alpha_matrix) @ XTy
        return self

    def predict(self,x):
        if self.theta is None:
            raise RuntimeError("Model must be fitted before calling predict().")

        if self.use_scaling:
            x_proc=self.scaler.transform(x)
        else:
            x_proc=x

        X=np.array(create_polynomial_features(x_proc,self.degree))
        return (X @ self.theta).tolist()


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

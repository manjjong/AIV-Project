import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RANSACRegressor


# 선형 회귀를 이용하여 함수 예측
def linear_interpolation(angles, distances, degree=3):
    poly = PolynomialFeatures(degree)
    x_poly = poly.fit_transform(angles)
    model = LinearRegression()
    model.fit(x_poly, distances)
    y_pred = model.predict(x_poly)
    return x_poly, y_pred


# RANSAC 회귀 모델을 이용하여 함수 예측
def ransac_interpolation(test_angles, train_angles, train_distances, degree=5, residual_threshold=1.5):
    poly = PolynomialFeatures(degree)
    x_train_poly = poly.fit_transform(train_angles)
    x_test_poly = poly.fit_transform(test_angles)
    model = RANSACRegressor(residual_threshold=residual_threshold)
    model.fit(x_train_poly, train_distances)
    y_pred = model.predict(x_test_poly)
    return y_pred


def interpolation(angles, distances, debug=False):
    new_angles = angles.reshape(-1, 1)
    x_poly, y_pred = linear_interpolation(new_angles, distances, 3)
    diff = np.abs(distances - y_pred)
    outliers = np.where(diff > 1.2)[0]
    new_angles_clean = np.delete(new_angles, outliers, axis=0)
    distances_clean = np.delete(distances, outliers, axis=0)
    y_pred_clean = ransac_interpolation(
        test_angles=new_angles,
        train_angles=new_angles_clean,
        train_distances=distances_clean,
        degree=7, residual_threshold=2)
    diff = np.abs(distances - y_pred_clean)
    small_outliers = np.where(diff > 2.0)[0]
    wide_outliers = np.where(diff > 0.75)[0]

    if debug:
        # 시각화
        plt.plot(angles, distances, 'o', label="Original Points")
        plt.plot(new_angles_clean, distances_clean, 'o', label="Dist Points")
        plt.plot(angles, y_pred, '-', label="PolynomialFeatures Regressor")
        plt.plot(angles, y_pred_clean, '-', label="RANSACRegressor Interpolation2")
        plt.legend()
        plt.show()
    return small_outliers, wide_outliers


def detect(angles, distances):
    return interpolation(angles, distances)


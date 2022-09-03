import pandas as pd
import numpy as np
from joblib import load
import xgboost as xgb
from sklearn.preprocessing import PolynomialFeatures
import torch
from ann import ResNet

path = "./"

poly = PolynomialFeatures(2)
ridge_face_file = load(path + "interface/poly_ridge_model.joblib")
ridge_slab_file = load(path + "intraslab/poly_ridge_model.joblib")

rf_face_file = load(path + "interface/random_forest_model.joblib")
rf_slab_file = load(path + "intraslab/random_forest_model.joblib")

gbdt_face_file = xgb.Booster()
gbdt_face_file.load_model(path + "interface/boost_model.json")
gbdt_slab_file = xgb.Booster()
gbdt_slab_file.load_model(path + "intraslab/boost_model.json")


svr_face_file = load(path + "interface/svr_model.joblib")
svr_face_scaler = load(path + "interface/svr_x_scaler.joblib")
svr_slab_file = load(path + "intraslab/svr_model.joblib")
svr_slab_scaler = load(path + "intraslab/svr_x_scaler.joblib")


ann_face_transformer = load(path + "interface/ann_transform_x.joblib")
ann_face_file = ResNet(n_feature=5, n_output=1)
ann_face_file.load_state_dict(torch.load(path + "interface/ann.pth", map_location=torch.device("cpu")))
ann_face_file.eval()

ann_slab_transformer = load(path + "intraslab/ann_transform_x.joblib")
ann_slab_file = ResNet(n_feature=5, n_output=1)
ann_slab_file.load_state_dict(torch.load(path + "intraslab/ann.pth", map_location=torch.device("cpu")))
ann_slab_file.eval()


def ML22_interface(KY, T, M, Sa13, PGV):
    # PGV in cm/s

    Tsmall = T < 0.1
    Tbig = 1 - Tsmall
    PGV_small = PGV < 10
    PGV_big = 1 - PGV_small

    a1 = -5.6229 * Tsmall - 6.2047 * Tbig
    a2 = -5.2542 * Tsmall + 2.0584 * Tbig
    a3 = 0.6233 * PGV_small + 0.7339 * PGV_big
    a4 = -0.7311 * Tbig

    lnD = (
        a1
        - 3.2597 * np.log(KY)
        - 0.3649 * np.log(KY) ** 2
        + 0.4795 * np.log(KY) * np.log(Sa13)
        + 2.6232 * np.log(Sa13)
        - 0.1245 * np.log(Sa13) ** 2
        + a2 * T
        + a4 * T ** 2
        + 0.1857 * M
        + a3 * np.log(PGV)
    )

    std = 0.6483

    return lnD, std


def ML22_intraslab(KY, T, M, Sa13, PGV):
    # PGV in cm/s

    Tsmall = T < 0.1
    Tbig = 1 - Tsmall
    PGV_small = PGV < 30
    PGV_big = 1 - PGV_small

    a1 = -5.9129 * Tsmall - 6.3404 * Tbig
    a2 = -3.8938 * Tsmall + 2.3176 * Tbig
    a3 = 0.6835 * PGV_small + 0.7195 * PGV_big
    a4 = -0.9037 * Tbig

    lnD = (
        a1
        - 2.3617 * np.log(KY)
        - 0.2247 * np.log(KY) ** 2
        + 0.2572 * np.log(KY) * np.log(Sa13)
        + 1.9677 * np.log(Sa13)
        - 0.0202 * np.log(Sa13) ** 2
        + a2 * T
        + a4 * T ** 2
        + 0.3797 * M
        + a3 * np.log(PGV)
    )

    std = 0.5275

    return lnD, std


def ridge_model(KY, T, M, Sa13, PGV, mechanism):

    input = np.vstack([np.log(KY), T, M, np.log(PGV), np.log(Sa13)]).T
    X = poly.fit_transform(input)

    if mechanism == "interface":
        lnD = ridge_face_file.predict(X)
        std = 0.59
    if mechanism == "intraslab":
        lnD = ridge_slab_file.predict(X)
        std = 0.51

    return lnD, std


def rf_model(KY, T, M, Sa13, PGV, mechanism):

    X = np.vstack([np.log(KY), T, M, np.log(PGV), np.log(Sa13)]).T

    if mechanism == "interface":
        lnD = rf_face_file.predict(X)
        std = 0.5
    if mechanism == "intraslab":
        lnD = rf_slab_file.predict(X)
        std = 0.5

    return lnD, std


def gbdt_model(KY, T, M, Sa13, PGV, mechanism):

    input = np.vstack([np.log(KY), T, M, np.log(PGV), np.log(Sa13)]).T
    input = pd.DataFrame(input, columns=["KY", "T", "M", "PGV", "Sa1.3"])
    X = xgb.DMatrix(input)

    if mechanism == "interface":
        lnD = gbdt_face_file.predict(X)
        std = 0.5
    if mechanism == "intraslab":
        lnD = gbdt_slab_file.predict(X)
        std = 0.5

    return lnD, std


def svr_model(KY, T, M, Sa13, PGV, mechanism):

    input = np.vstack([np.log(KY), T, M, np.log(PGV), np.log(Sa13)]).T
    input = pd.DataFrame(input, columns=["KY", "T", "M", "PGV", "Sa1.3"])
    input["kysa"] = input["KY"] * input["Sa1.3"]
    input["sa2"] = input["Sa1.3"] ** 2
    input["pgv2"] = input["PGV"] ** 2
    input["t2"] = input["T"] ** 2
    input["ky2"] = input["KY"] ** 2

    if mechanism == "interface":
        lnD = svr_face_file.predict(svr_face_scaler.transform(input))
        std = 0.65
    if mechanism == "intraslab":
        lnD = svr_slab_file.predict(svr_slab_scaler.transform(input))
        std = 0.54

    return lnD, std


def ann_model(KY, T, M, Sa13, PGV, mechanism):

    X = np.vstack([np.log(KY), T, M, np.log(PGV), np.log(Sa13)]).T

    if mechanism == "interface":
        X_transform = ann_face_transformer.transform(X)
        with torch.no_grad():
            lnD = ann_face_file(torch.from_numpy(X_transform.astype("float32")))
        lnD = lnD.cpu().detach().numpy().flatten()
        std = 0.5

    if mechanism == "intraslab":
        X_transform = ann_slab_transformer.transform(X)
        with torch.no_grad():
            lnD = ann_slab_file(torch.from_numpy(X_transform.astype("float32")))
        lnD = lnD.cpu().detach().numpy().flatten()
        std = 0.5

    return lnD, std

"""
prediction.py
--------------
Script untuk memprediksi attrition karyawan menggunakan model yang telah
dilatih pada notebook.ipynb (model_attrition.joblib, scaler.joblib,
encoders.joblib, feature_columns.joblib).

Cara pakai:
    python prediction.py --input employee_data.csv --output predicted_attrition.csv

Jika --input tidak diisi, script akan memakai employee_data.csv secara default
dan otomatis memprediksi baris-baris yang kolom Attrition-nya masih kosong.
"""

import argparse
import sys
import pandas as pd
import joblib

MODEL_DIR = "model"


def load_artifacts(model_dir: str = MODEL_DIR):
    model = joblib.load(f"{model_dir}/model_attrition.joblib")
    scaler = joblib.load(f"{model_dir}/scaler.joblib")
    encoders = joblib.load(f"{model_dir}/encoders.joblib")
    feature_columns = joblib.load(f"{model_dir}/feature_columns.joblib")
    return model, scaler, encoders, feature_columns


def preprocess(df: pd.DataFrame, encoders: dict, feature_columns: list) -> pd.DataFrame:
    data = df.copy()

    drop_cols = [c for c in ["EmployeeId", "EmployeeCount", "StandardHours", "Over18", "Attrition"]
                 if c in data.columns]
    data = data.drop(columns=drop_cols)

    for col, le in encoders.items():
        if col in data.columns:
            data[col] = le.transform(data[col].astype(str))

    missing = [c for c in feature_columns if c not in data.columns]
    if missing:
        raise ValueError(f"Kolom input tidak lengkap, kolom berikut hilang: {missing}")

    return data[feature_columns]


def predict(df: pd.DataFrame, model_dir: str = MODEL_DIR) -> pd.DataFrame:
    model, scaler, encoders, feature_columns = load_artifacts(model_dir)

    X = preprocess(df, encoders, feature_columns)
    X_scaled = scaler.transform(X)

    result = df.copy()
    result["Attrition_Predicted"] = model.predict(X_scaled)
    result["Attrition_Probability"] = model.predict_proba(X_scaled)[:, 1]
    return result


def main():
    parser = argparse.ArgumentParser(description="Prediksi attrition karyawan.")
    parser.add_argument("--input", default="employee_data.csv",
                         help="Path ke file CSV data karyawan (default: employee_data.csv)")
    parser.add_argument("--output", default="predicted_attrition.csv",
                         help="Path file CSV hasil prediksi (default: predicted_attrition.csv)")
    parser.add_argument("--model-dir", default=MODEL_DIR,
                         help="Folder berisi file .joblib hasil training (default: model)")
    parser.add_argument("--only-unlabeled", action="store_true", default=True,
                         help="Jika file input punya kolom Attrition, hanya prediksi baris yang kosong (default: True)")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"File input tidak ditemukan: {args.input}")
        sys.exit(1)

    if args.only_unlabeled and "Attrition" in df.columns and df["Attrition"].isna().any():
        df_to_predict = df[df["Attrition"].isna()].copy()
        print(f"Memprediksi {len(df_to_predict)} baris dengan Attrition kosong...")
    else:
        df_to_predict = df.copy()
        print(f"Memprediksi seluruh {len(df_to_predict)} baris pada file input...")

    result = predict(df_to_predict, model_dir=args.model_dir)
    result.to_csv(args.output, index=False)

    n_resign = int(result["Attrition_Predicted"].sum())
    print(f"Selesai. {n_resign} dari {len(result)} karyawan diprediksi berpotensi resign.")
    print(f"Hasil disimpan ke: {args.output}")


if __name__ == "__main__":
    main()

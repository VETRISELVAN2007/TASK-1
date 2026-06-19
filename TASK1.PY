from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless backend -- saves plots to file, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 1000)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def section(title: str) -> None:
    """Pretty console section header -- used throughout the report."""
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}")


# DOMAIN KNOWLEDGE: title / deck rules specific to this dataset

RARE_TITLES = {
    "Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev",
    "Sir", "Jonkheer", "Dona",
}
TITLE_NORMALIZATION = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}


def extract_title(name: str) -> str:
    """Pull the social title out of a 'Last, Title. First' name string."""
    title = name.split(",")[1].split(".")[0].strip()
    title = title.replace("the ", "").strip()  # handles "the Countess"
    title = TITLE_NORMALIZATION.get(title, title)
    return "Rare" if title in RARE_TITLES else title


def extract_deck(cabin) -> str:
    """First letter of the cabin code, or 'U' (Unknown) if missing."""
    return "U" if pd.isna(cabin) else str(cabin)[0]


# THE PIPELINE

class TitanicPipeline:
    """
    One object, one run() call, every stage of the project.

    Stages (in order):
      1. load                - read the CSV, validate schema
      2. audit_report        - shape / nulls / cardinality / outliers /
                                correlation / target-group stats (printed)
      3. clean_and_engineer  - Titanic-specific ETL + feature engineering
      4. eda_dashboard       - 6-panel survival-focused chart, saved to PNG
      5. model_bakeoff       - cross-validate 4 candidate models
      6. tune_best_model     - GridSearchCV on the cross-val winner
      7. evaluate            - held-out test metrics, confusion matrix, ROC
      8. feature_importance  - top-15 driver chart
      9. finalize_and_save   - refit on all data, persist model + metadata
     10. predict_examples    - run 5 hand-built "what if" passengers
    """

    def __init__(self, csv_path: str = "Titanic-Dataset.csv", output_dir: str = "outputs"):
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.df_raw: pd.DataFrame | None = None
        self.df: pd.DataFrame | None = None
        self.numeric_features: list[str] = []
        self.categorical_features: list[str] = []

        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.best_model_name: str | None = None
        self.best_model: Pipeline | None = None
        self.best_params: dict = {}
        self.cv_score: float = 0.0
        self.test_metrics: dict = {}

        # Every chart saved during the run gets logged here (label, path)
        # so Stage 11 can print a full, verifiable index at the end.
        self.saved_charts: list[tuple[str, Path]] = []

    def _save_fig(self, fig, filename: str, label: str) -> Path:
        """Single choke point for saving every matplotlib figure in this
        pipeline. Saves to disk, closes the figure to free memory, prints
        an immediate confirmation line, and records it for the final
        Stage 11 graph index -- so no chart can silently go unreported."""
        out_path = self.output_dir / filename
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        exists = out_path.exists()
        size_kb = out_path.stat().st_size / 1024 if exists else 0
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {label:<32s} -> {out_path}  ({size_kb:.0f} KB)")
        self.saved_charts.append((label, out_path))
        return out_path

        # 1. LOAD
    
    def load(self) -> "TitanicPipeline":
        section("STAGE 1 / 10 -- LOAD & VALIDATE")
        if not self.csv_path.exists():
            print(f"[FATAL] Could not find {self.csv_path}. "
                  f"Put Titanic-Dataset.csv next to this script.")
            sys.exit(1)

        self.df_raw = pd.read_csv(self.csv_path)
        expected = ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age",
                    "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]
        missing = [c for c in expected if c not in self.df_raw.columns]
        extra = [c for c in self.df_raw.columns if c not in expected]

        print(f"Loaded '{self.csv_path.name}' -> {self.df_raw.shape[0]} rows, "
              f"{self.df_raw.shape[1]} columns")
        if missing or extra:
            print(f"[WARNING] Schema differs from the classic Titanic schema.")
            if missing:
                print(f"  Missing expected columns : {missing}")
            if extra:
                print(f"  Unexpected extra columns : {extra}")
        else:
            print("[OK] Schema matches the classic Kaggle Titanic columns exactly.")
        return self

        # 2. AUDIT REPORT  (printed console report, like a data-quality memo)
    
    def audit_report(self) -> "TitanicPipeline":
        section("STAGE 2 / 10 -- DATA AUDIT REPORT")
        df = self.df_raw

        print(f"Shape            : {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"Memory footprint : {df.memory_usage(deep=True).sum() / 1024**2:.3f} MB")
        print(f"Duplicate rows   : {df.duplicated().sum()}")

        print("\n--- Null Audit ---")
        nulls = df.isnull().sum()
        audit = pd.DataFrame({
            "Null Count": nulls,
            "Null %": (nulls / len(df) * 100).round(2),
            "Non-Null Count": df.notnull().sum(),
        })
        print(audit[audit["Null Count"] >= 0])

        print("\n--- Cardinality Audit (categorical columns) ---")
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for col in cat_cols:
            uniq = df[col].nunique()
            mode = df[col].mode().iloc[0] if not df[col].mode().empty else "n/a"
            freq = df[col].value_counts().iloc[0] if uniq else 0
            print(f"  {col:<10s} unique={uniq:<5d} most common='{mode}' (x{freq})")

        print("\n--- Outlier Audit (IQR rule, numeric columns) ---")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in num_cols:
            s = df[col].dropna()
            if len(s) < 5:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outlier_rate = ((s < lo) | (s > hi)).mean() * 100
            print(f"  {col:<12s} outlier rate = {outlier_rate:5.2f}%")

        print("\n--- Correlation with Target (Survived), numeric columns ---")
        target_corr = df[num_cols].corr(numeric_only=True)["Survived"] \
                                   .drop("Survived").sort_values(ascending=False)
        print(target_corr.round(4))

        print("\n--- Mean Feature Values by Survival Outcome ---")
        group_stats = df.groupby("Survived")[["Age", "Fare", "Pclass", "SibSp", "Parch"]].mean()
        print(group_stats.round(3))

        self._save_missingness_plot(df)
        return self

    def _save_missingness_plot(self, df: pd.DataFrame) -> None:
        null_pct = (df.isnull().sum() / len(df) * 100)
        null_pct = null_pct[null_pct > 0].sort_values(ascending=False)
        if null_pct.empty:
            return
        fig, ax = plt.subplots(figsize=(7, 3.5))
        sns.barplot(x=null_pct.values, y=null_pct.index, hue=null_pct.index,
                    palette="rocket", legend=False, ax=ax)
        ax.set_title("Missing Data by Column (%)")
        ax.set_xlabel("Percent missing")
        ax.set_ylabel("Column")
        fig.tight_layout()
        fig.savefig(self.output_dir / "01_missingness.png", dpi=150)
        plt.close(fig)
        print(f"\nSaved missingness chart -> {self.output_dir / '01_missingness.png'}")

     # 3. CLEAN + DOMAIN-SPECIFIC FEATURE ENGINEERING
    
    def clean_and_engineer(self) -> "TitanicPipeline":
        section("STAGE 3 / 10 -- ETL & FEATURE ENGINEERING")
        data = self.df_raw.copy()

        # --- Title, extracted from the Name string ---
        data["Title"] = data["Name"].apply(extract_title)
        print("Extracted 'Title' from Name:")
        print(data["Title"].value_counts().to_string())

        # --- Family-based features ---
        data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
        data["IsAlone"] = (data["FamilySize"] == 1).astype(int)
        print(f"\nBuilt FamilySize (SibSp+Parch+1) and IsAlone flag. "
              f"{data['IsAlone'].sum()} passengers traveled alone.")

        # --- Deck, extracted from Cabin's first letter ---
        data["Deck"] = data["Cabin"].apply(extract_deck)
        print(f"Extracted 'Deck' from Cabin (U = unknown). "
              f"{(data['Deck'] == 'U').sum()} passengers have no cabin on record.")

        # --- Age imputation: median per (Title, Pclass) group, not a single global value ---
        before_missing = data["Age"].isnull().sum()
        data["Age"] = data.groupby(["Title", "Pclass"])["Age"].transform(
            lambda s: s.fillna(s.median())
        )
        data["Age"] = data["Age"].fillna(data["Age"].median())
        print(f"\nImputed {before_missing} missing Age values using the median "
              f"Age for each (Title, Pclass) group.")

        # --- Fare imputation: median per Pclass ---
        fare_missing = data["Fare"].isnull().sum()
        data["Fare"] = data.groupby("Pclass")["Fare"].transform(lambda s: s.fillna(s.median()))
        if fare_missing:
            print(f"Imputed {fare_missing} missing Fare values using the median Fare for their Pclass.")

        # --- Embarked imputation: most frequent port ---
        embarked_missing = data["Embarked"].isnull().sum()
        data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])
        if embarked_missing:
            print(f"Imputed {embarked_missing} missing Embarked values with the most common port.")

        # --- Binned versions (cleaner split points for tree models) ---
        data["AgeBin"] = pd.cut(
            data["Age"], bins=[0, 12, 18, 35, 60, 100],
            labels=["Child", "Teen", "Adult", "MiddleAge", "Senior"],
        )
        data["FareBin"] = pd.qcut(data["Fare"], q=4, labels=["Low", "Mid", "High", "VeryHigh"])

        # --- Interaction feature ---
        data["Pclass_Sex"] = data["Pclass"].astype(str) + "_" + data["Sex"]

        print(f"\nFinal engineered shape: {data.shape[0]} rows x {data.shape[1]} columns "
              f"(started at {self.df_raw.shape[1]} raw columns).")
        print("\n--- Preview of engineered data (first 5 rows) ---")
        preview_cols = ["Name", "Title", "Pclass", "Sex", "Age", "FamilySize",
                         "IsAlone", "Deck", "AgeBin", "FareBin"]
        print(data[preview_cols].head().to_string(index=False))

        self.df = data
        self.numeric_features = ["Age", "Fare", "FamilySize", "IsAlone", "SibSp", "Parch"]
        self.categorical_features = ["Pclass", "Sex", "Embarked", "Title", "Deck",
                                      "AgeBin", "FareBin", "Pclass_Sex"]
        return self

        # 4. EDA DASHBOARD
    
    def eda_dashboard(self) -> "TitanicPipeline":
        section("STAGE 4 / 10 -- EXPLORATORY DASHBOARD")
        df = self.df
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))

        sns.barplot(data=df, x="Sex", y="Survived", ax=axes[0, 0])
        axes[0, 0].set_title("Survival Rate by Sex")

        sns.barplot(data=df, x="Pclass", y="Survived", ax=axes[0, 1])
        axes[0, 1].set_title("Survival Rate by Passenger Class")

        sns.barplot(data=df, x="Title", y="Survived", ax=axes[0, 2])
        axes[0, 2].set_title("Survival Rate by Title")
        axes[0, 2].tick_params(axis="x", rotation=45)

        sns.histplot(data=df, x="Age", hue="Survived", multiple="stack", bins=30, ax=axes[1, 0])
        axes[1, 0].set_title("Age Distribution by Survival")

        sns.barplot(data=df, x="FamilySize", y="Survived", ax=axes[1, 1])
        axes[1, 1].set_title("Survival Rate by Family Size")

        sns.barplot(data=df, x="Deck", y="Survived", ax=axes[1, 2], order=sorted(df["Deck"].unique()))
        axes[1, 2].set_title("Survival Rate by Deck")

        plt.tight_layout()
        out_path = self.output_dir / "02_eda_dashboard.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved 6-panel EDA dashboard -> {out_path}")

        # Correlation heatmap of engineered numeric features
        fig, ax = plt.subplots(figsize=(7, 5))
        corr = df[self.numeric_features + ["Survived"]].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation Heatmap (Numeric Features)")
        fig.tight_layout()
        out_path = self.output_dir / "03_correlation_heatmap.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved correlation heatmap -> {out_path}")
        return self

       # Helpers for modeling stages
    
    def _build_preprocessor(self) -> ColumnTransformer:
        return ColumnTransformer(transformers=[
            ("num", StandardScaler(), self.numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_features),
        ])

    @staticmethod
    def _candidate_models() -> dict:
        return {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE),
            "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "SVM": SVC(probability=True, random_state=RANDOM_STATE),
        }

    _PARAM_GRIDS = {
        "LogisticRegression": {"model__C": [0.01, 0.1, 1, 10], "model__penalty": ["l2"]},
        "RandomForest": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [4, 6, 8, None],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
        },
        "GradientBoosting": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.03, 0.1],
            "model__max_depth": [2, 3, 4],
        },
        "SVM": {"model__C": [0.5, 1, 5, 10], "model__kernel": ["rbf"], "model__gamma": ["scale", "auto"]},
    }

        # 5. MODEL BAKE-OFF
    

    def model_bakeoff(self) -> "TitanicPipeline":
        section("STAGE 5 / 10 -- MODEL BAKE-OFF (5-FOLD STRATIFIED CV)")
        X = self.df[self.numeric_features + self.categorical_features]
        y = self.df["Survived"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
        )
        print(f"Train rows: {len(self.X_train)}   Test rows: {len(self.X_test)}")

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        preprocessor = self._build_preprocessor()
        rows = []
        for name, model in self._candidate_models().items():
            pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
            t0 = time.time()
            scores = cross_val_score(pipe, self.X_train, self.y_train, cv=cv,
                                      scoring="accuracy", n_jobs=-1)
            elapsed = time.time() - t0
            rows.append({"model": name, "mean_accuracy": scores.mean(),
                         "std_accuracy": scores.std(), "time_sec": round(elapsed, 2)})
            print(f"  {name:<20s} acc = {scores.mean():.4f}  (+/- {scores.std():.4f})  [{elapsed:.2f}s]")

        results = pd.DataFrame(rows).sort_values("mean_accuracy", ascending=False)
        results.to_csv(self.output_dir / "model_comparison.csv", index=False)
        self.best_model_name = results.iloc[0]["model"]
        print(f"\nBest candidate by CV accuracy: {self.best_model_name}")
        return self

        # 6. TUNE BEST MODEL
   
    def tune_best_model(self) -> "TitanicPipeline":
        section(f"STAGE 6 / 10 -- HYPERPARAMETER TUNING ({self.best_model_name})")
        preprocessor = self._build_preprocessor()
        model = self._candidate_models()[self.best_model_name]
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        grid = self._PARAM_GRIDS[self.best_model_name]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        search = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=-1)
        search.fit(self.X_train, self.y_train)

        self.best_model = search.best_estimator_
        self.best_params = search.best_params_
        self.cv_score = search.best_score_
        print(f"Best CV accuracy after tuning: {self.cv_score:.4f}")
        print(f"Best hyperparameters         : {self.best_params}")
        return self

        # 7. EVALUATE ON HELD-OUT TEST SET
    
    def evaluate(self) -> "TitanicPipeline":
        section("STAGE 7 / 10 -- HELD-OUT TEST SET EVALUATION")
        y_pred = self.best_model.predict(self.X_test)
        y_proba = self.best_model.predict_proba(self.X_test)[:, 1]

        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_proba)
        self.test_metrics = {"accuracy": acc, "f1": f1, "roc_auc": auc}

        print(f"Accuracy : {acc:.4f}")
        print(f"F1 score : {f1:.4f}")
        print(f"ROC-AUC  : {auc:.4f}")
        print("\nClassification report:")
        print(classification_report(self.y_test, y_pred, target_names=["Died", "Survived"]))

        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            self.y_test, y_pred, display_labels=["Died", "Survived"], cmap="Blues", ax=ax
        )
        ax.set_title(f"Confusion Matrix -- {self.best_model_name}")
        fig.tight_layout()
        fig.savefig(self.output_dir / "04_confusion_matrix.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_predictions(self.y_test, y_proba, ax=ax)
        ax.set_title(f"ROC Curve -- {self.best_model_name} (AUC = {auc:.3f})")
        fig.tight_layout()
        fig.savefig(self.output_dir / "05_roc_curve.png", dpi=150)
        plt.close(fig)

        print(f"\nSaved confusion matrix -> {self.output_dir / '04_confusion_matrix.png'}")
        print(f"Saved ROC curve        -> {self.output_dir / '05_roc_curve.png'}")
        return self

    
    # 8. FEATURE IMPORTANCE
    

    def feature_importance(self) -> "TitanicPipeline":
        section("STAGE 8 / 10 -- FEATURE IMPORTANCE")
        try:
            preprocessor = self.best_model.named_steps["preprocessor"]
            names = preprocessor.get_feature_names_out()
            estimator = self.best_model.named_steps["model"]

            if hasattr(estimator, "feature_importances_"):
                importances = estimator.feature_importances_
            elif hasattr(estimator, "coef_"):
                importances = np.abs(estimator.coef_[0])
            else:
                print(f"{self.best_model_name} exposes no importances -- skipping.")
                return self

            imp_df = (pd.DataFrame({"feature": names, "importance": importances})
                      .sort_values("importance", ascending=False).head(15))
            print(imp_df.to_string(index=False))

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(data=imp_df, x="importance", y="feature", ax=ax, palette="viridis")
            ax.set_title(f"Top 15 Feature Importances -- {self.best_model_name}")
            fig.tight_layout()
            fig.savefig(self.output_dir / "06_feature_importance.png", dpi=150)
            plt.close(fig)
            print(f"\nSaved feature importance chart -> {self.output_dir / '06_feature_importance.png'}")
        except Exception as exc:
            print(f"Could not compute feature importance: {exc}")
        return self

   

    # 9. FINALIZE + SAVE
   
    def finalize_and_save(self) -> "TitanicPipeline":
        section("STAGE 9 / 10 -- FINALIZE & SAVE MODEL")
        X_full = self.df[self.numeric_features + self.categorical_features]
        y_full = self.df["Survived"]
        self.best_model.fit(X_full, y_full)  # refit on 100% of labeled data

        model_path = self.output_dir / "titanic_model.joblib"
        joblib.dump(self.best_model, model_path)

        metadata = {
            "best_model": self.best_model_name,
            "best_params": self.best_params,
            "cv_accuracy": round(self.cv_score, 4),
            "test_accuracy": round(self.test_metrics["accuracy"], 4),
            "test_f1": round(self.test_metrics["f1"], 4),
            "test_roc_auc": round(self.test_metrics["roc_auc"], 4),
            "n_train_rows": len(self.X_train),
            "n_test_rows": len(self.X_test),
            "feature_columns": self.numeric_features + self.categorical_features,
        }
        with open(self.output_dir / "model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved trained pipeline -> {model_path}")
        print(f"Saved run metadata     -> {self.output_dir / 'model_metadata.json'}")
        print(json.dumps(metadata, indent=2))
        return self

        # 10. PREDICT ON NEW / EXAMPLE PASSENGERS
    
    @staticmethod
    def _demo_passengers() -> pd.DataFrame:
        return pd.DataFrame([
            {"PassengerId": 9001, "Pclass": 1, "Name": "Ashworth, Mrs. Helen Caldwell",
             "Sex": "female", "Age": 32, "SibSp": 1, "Parch": 0, "Ticket": "X1",
             "Fare": 100.0, "Cabin": "C85", "Embarked": "S"},
            {"PassengerId": 9002, "Pclass": 3, "Name": "Dunmore, Mr. James Patrick",
             "Sex": "male", "Age": 28, "SibSp": 0, "Parch": 0, "Ticket": "X2",
             "Fare": 7.9, "Cabin": None, "Embarked": "S"},
            {"PassengerId": 9003, "Pclass": 2, "Name": "Whitfield, Master. Thomas",
             "Sex": "male", "Age": 6, "SibSp": 1, "Parch": 2, "Ticket": "X3",
             "Fare": 26.0, "Cabin": None, "Embarked": "S"},
            {"PassengerId": 9004, "Pclass": 3, "Name": "Moreau, Miss. Adele",
             "Sex": "female", "Age": 19, "SibSp": 0, "Parch": 0, "Ticket": "X4",
             "Fare": 8.05, "Cabin": None, "Embarked": "C"},
            {"PassengerId": 9005, "Pclass": 1, "Name": "Pemberton, Mr. Geoffrey",
             "Sex": "male", "Age": 58, "SibSp": 0, "Parch": 0, "Ticket": "X5",
             "Fare": 78.5, "Cabin": "B22", "Embarked": "S"},
        ])

    def _engineer_for_inference(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Re-applies the same domain feature engineering used in stage 3,
        so predict-time features always match train-time features exactly."""
        data = raw.copy()
        data["Title"] = data["Name"].apply(extract_title)
        data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
        data["IsAlone"] = (data["FamilySize"] == 1).astype(int)
        data["Deck"] = data["Cabin"].apply(extract_deck)

        ref = self.df  # use training distribution for any needed fills
        data["Age"] = data["Age"].fillna(ref["Age"].median())
        data["Fare"] = data["Fare"].fillna(ref["Fare"].median())
        data["Embarked"] = data["Embarked"].fillna(ref["Embarked"].mode()[0])

        data["AgeBin"] = pd.cut(data["Age"], bins=[0, 12, 18, 35, 60, 100],
                                 labels=["Child", "Teen", "Adult", "MiddleAge", "Senior"])
        # Use training fare quartile edges so bins line up with the trained model
        fare_edges = pd.qcut(ref["Fare"], q=4, retbins=True)[1]
        fare_edges[0], fare_edges[-1] = -np.inf, np.inf
        data["FareBin"] = pd.cut(data["Fare"], bins=fare_edges,
                                  labels=["Low", "Mid", "High", "VeryHigh"])
        data["Pclass_Sex"] = data["Pclass"].astype(str) + "_" + data["Sex"]
        return data

    def predict_examples(self) -> "TitanicPipeline":
        section("STAGE 10 / 10 -- PREDICTIONS ON NEW PASSENGERS")
        raw = self._demo_passengers()
        engineered = self._engineer_for_inference(raw)
        X_new = engineered[self.numeric_features + self.categorical_features]

        preds = self.best_model.predict(X_new)
        probs = self.best_model.predict_proba(X_new)[:, 1]

        result = raw.copy()
        result["Survived_Prediction"] = preds
        result["Survival_Probability"] = probs.round(4)

        cols = ["PassengerId", "Name", "Pclass", "Sex", "Age",
                "Survived_Prediction", "Survival_Probability"]
        print(result[cols].to_string(index=False))

        out_path = self.output_dir / "example_predictions.csv"
        result.to_csv(out_path, index=False)
        print(f"\nSaved example predictions -> {out_path}")
        return self

    
    # RUN EVERYTHING
    
    def run(self) -> "TitanicPipeline":
        (self.load()
             .audit_report()
             .clean_and_engineer()
             .eda_dashboard()
             .model_bakeoff()
             .tune_best_model()
             .evaluate()
             .feature_importance()
             .finalize_and_save()
             .predict_examples())

        section("PIPELINE COMPLETE")
        print(f"All charts, the trained model, and prediction CSVs are in: "
              f"{self.output_dir.resolve()}")
        return self


if __name__ == "__main__":
    TitanicPipeline(csv_path="Titanic-Dataset.csv", output_dir="outputs").run()
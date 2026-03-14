import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..data_loader import AdultIncomeDataLoader


class TestingAdultIncomeDataLoader(AdultIncomeDataLoader):
    """
    Testing variant of AdultIncomeDataLoader.

    Extends the base loader with:
    - Ground truth loan approval labels derived from creditworthiness threshold.
    - Exposes the fitted StandardScaler for reproducibility.
    """

    def __init__(
        self,
        filepath: str = None,
        sample_size: int = 50000,
        credit_threshold: float = 0.5,
    ):
        super().__init__(filepath=filepath, sample_size=sample_size)
        self.credit_threshold = credit_threshold
        self.scaler = None  # Stored after fitting in preprocess()

    def _derive_creditworthiness_optimal(self, df):
        """
        Derive creditworthiness AND store the fitted scaler for later use.
        Ground truth approvals are set by thresholding the creditworthiness score.
        """
        print(
            "\n  Deriving creditworthiness (ground truth) using income prediction model..."
        )

        feature_cols = [
            "age",
            "education_num",
            "hours_per_week",
            "capital_gain",
            "capital_loss",
        ]
        X_features = df[feature_cols].copy()
        X_features["capital_gain_log"] = np.log1p(X_features["capital_gain"])
        X_features["capital_loss_log"] = np.log1p(X_features["capital_loss"])
        X_features.drop(["capital_gain", "capital_loss"], axis=1, inplace=True)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_features)

        y = (df["income"].str.strip() == ">50K").astype(int)

        self.creditworthiness_model = LogisticRegression(random_state=42, max_iter=1000)
        self.creditworthiness_model.fit(X_scaled, y)

        creditworthiness_probs = self.creditworthiness_model.predict_proba(X_scaled)[
            :, 1
        ]
        return creditworthiness_probs

    def preprocess(self):
        """Preprocess data and compute ground truth approvals."""
        print("Preprocessing Adult Income data...")

        df = self.data.copy()
        df["S"] = (df["sex"].str.strip() == "Male").astype(int)
        df["approved"] = (df["income"].str.strip() == ">50K").astype(int)
        df["creditworthiness"] = self._derive_creditworthiness_optimal(df)
        df["X"] = 10 + df["creditworthiness"] * 190

        # Ground truth: approve if creditworthiness >= threshold
        df["ground_truth_approval"] = (
            df["creditworthiness"] >= self.credit_threshold
        ).astype(int)

        self.male_data = df[df["S"] == 1].reset_index(drop=True)
        self.female_data = df[df["S"] == 0].reset_index(drop=True)

        print(f"\nMale applicants: {len(self.male_data)}")
        print(f"Female applicants: {len(self.female_data)}")
        print(
            f"Male ground truth approval rate: "
            f"{self.male_data['ground_truth_approval'].mean():.3f}"
        )
        print(
            f"Female ground truth approval rate: "
            f"{self.female_data['ground_truth_approval'].mean():.3f}"
        )

        self.data = df
        return df

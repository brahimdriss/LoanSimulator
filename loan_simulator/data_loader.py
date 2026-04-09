import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class AdultIncomeDataLoader:
    """Load and preprocess Adult Income dataset (Census Bureau)."""

    def __init__(self, filepath: str = None, sample_size: int = 50000):
        self.filepath = filepath
        self.sample_size = sample_size
        self.data = None
        self.male_data = None
        self.female_data = None
        self.creditworthiness_model = None

    def load_data(self):
        """Load Adult Income data."""
        if self.filepath is None:
            print("Downloading Adult Income dataset...")
            self.data = self._download_adult_income_data()
        else:
            print(f"Loading Adult Income data from {self.filepath}...")
            try:
                self.data = pd.read_csv(
                    self.filepath, skipinitialspace=True, na_values="?"
                )
                self.data = self.data.dropna()

                column_mapping = {
                    "educational-num": "education_num",
                    "marital-status": "marital_status",
                    "capital-gain": "capital_gain",
                    "capital-loss": "capital_loss",
                    "hours-per-week": "hours_per_week",
                    "native-country": "native_country",
                    "gender": "sex",
                }
                self.data = self.data.rename(columns=column_mapping)

                if len(self.data) > self.sample_size:
                    self.data = self.data.sample(n=self.sample_size, random_state=42)

                print(f"✓ Loaded {len(self.data):,} records from {self.filepath}")

            except Exception as e:
                print(f"Error loading file: {e}")
                print("Trying alternative sources...")
                self.data = self._download_adult_income_data()

        return self.data

    def _download_adult_income_data(self):
        """Download Adult Income dataset from UCI, with fallback to synthetic."""
        try:
            print("  Trying UCI Repository...")
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
            columns = [
                "age",
                "workclass",
                "fnlwgt",
                "education",
                "education_num",
                "marital_status",
                "occupation",
                "relationship",
                "race",
                "sex",
                "capital_gain",
                "capital_loss",
                "hours_per_week",
                "native_country",
                "income",
            ]
            df = pd.read_csv(url, names=columns, skipinitialspace=True, na_values="?")
            df = df.dropna()

            if len(df) > self.sample_size:
                df = df.sample(n=self.sample_size, random_state=42)

            print(f"  ✓ Downloaded {len(df):,} records from UCI repository")
            return df

        except Exception as e:
            print(f"  UCI failed: {e}")
            return self._generate_synthetic_adult_data()

    def _generate_synthetic_adult_data(self):
        """Generate synthetic Adult Income-like data."""
        print("  Generating realistic synthetic Adult Income data...")
        np.random.seed(42)
        n = self.sample_size

        sex = np.random.choice(["Male", "Female"], size=n, p=[0.67, 0.33])

        data = []
        for i in range(n):
            is_male = sex[i] == "Male"

            age = np.clip(np.random.normal(38 if is_male else 36, 13), 17, 90)
            education_num = np.clip(
                np.random.normal(10.2 if is_male else 10.0, 2.5), 1, 16
            )
            hours_per_week = np.clip(np.random.normal(42 if is_male else 37, 12), 1, 99)

            capital_gain = (
                np.random.exponential(15000 if is_male else 12000)
                if np.random.random() < 0.08
                else 0
            )
            capital_loss = (
                np.random.exponential(1500) if np.random.random() < 0.04 else 0
            )

            base_prob = 0.15
            base_prob += 0.12 if is_male else 0
            base_prob += 0.02 * max(0, (education_num - 10))
            base_prob += 0.005 * max(0, (age - 30))
            base_prob -= 0.003 * max(0, (age - 50))
            base_prob += 0.002 * max(0, (hours_per_week - 40))
            base_prob += 0.0001 * capital_gain
            base_prob = np.clip(base_prob, 0.05, 0.85)
            income = ">50K" if np.random.random() < base_prob else "<=50K"

            data.append(
                {
                    "age": int(age),
                    "workclass": np.random.choice(
                        ["Private", "Self-emp", "Gov", "Other"],
                        p=[0.7, 0.15, 0.12, 0.03],
                    ),
                    "fnlwgt": np.random.randint(10000, 1000000),
                    "education": "HS-grad",
                    "education_num": int(education_num),
                    "marital_status": np.random.choice(
                        ["Married", "Single", "Divorced"], p=[0.5, 0.35, 0.15]
                    ),
                    "occupation": "Other",
                    "relationship": "Other",
                    "race": "White",
                    "sex": sex[i],
                    "capital_gain": int(capital_gain),
                    "capital_loss": int(capital_loss),
                    "hours_per_week": int(hours_per_week),
                    "native_country": "United-States",
                    "income": income,
                }
            )

        df = pd.DataFrame(data)
        print(f"  ✓ Generated {len(df):,} synthetic records")
        return df

    def _derive_creditworthiness_optimal(self, df):
        """Derive creditworthiness using logistic regression on income prediction."""
        print("\n  Deriving creditworthiness using income prediction model...")

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

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_features)

        y = (df["income"].str.strip() == ">50K").astype(int)

        self.creditworthiness_model = LogisticRegression(random_state=42, max_iter=1000)
        self.creditworthiness_model.fit(X_scaled, y)

        creditworthiness_probs = self.creditworthiness_model.predict_proba(X_scaled)[
            :, 1
        ]
        return creditworthiness_probs

    def preprocess(self):
        """Preprocess Adult Income data."""
        print("Preprocessing Adult Income data...")

        df = self.data.copy()
        df["S"] = (df["sex"].str.strip() == "Male").astype(int)
        df["approved"] = (df["income"].str.strip() == ">50K").astype(int)
        df["creditworthiness"] = self._derive_creditworthiness_optimal(df)
        df["X"] = 10 + df["creditworthiness"] * 190

        self.male_data = df[df["S"] == 1].reset_index(drop=True)
        self.female_data = df[df["S"] == 0].reset_index(drop=True)

        print(f"\nMale applicants: {len(self.male_data)}")
        print(f"Female applicants: {len(self.female_data)}")
        print(f"Male high-income rate: {self.male_data['approved'].mean():.3f}")
        print(f"Female high-income rate: {self.female_data['approved'].mean():.3f}")

        self.data = df
        return df

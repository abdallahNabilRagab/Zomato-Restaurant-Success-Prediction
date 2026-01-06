# ============================================================
# src/data_pipeline.py
# ============================================================
# ============================================================
# Standard Libraries
# ============================================================
import os
import re
import logging
from datetime import datetime
from typing import Any
import argparse

# ============================================================
# Data Manipulation
# ============================================================
import pandas as pd
import numpy as np

# ============================================================
# Visualization
# ============================================================
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots

# ============================================================
# Machine Learning / Preprocessing
# ============================================================
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# ============================================================
# Global Logger (Enterprise Level)
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger("DataPipeline")



# ============================================================
# 0) EDA PIPELINE
# ============================================================
class EDAPipeline:
    def __init__(self, df, id_col="RowCount"):
        self.df = df.copy()
        self.id_col = id_col
        
        # Auto detect numerical & categorical
        self.cat_cols = df.select_dtypes("O").columns.tolist()
        self.num_cols = df.select_dtypes(exclude="O").columns.tolist()

        print(f"[INFO] Detected {len(self.cat_cols)} categorical columns.")
        print(f"[INFO] Detected {len(self.num_cols)} numerical columns.")

    # -----------------------------------------------------------
    # 🔹 1. UNIVARIATE ANALYSIS — Categorical
    # -----------------------------------------------------------
    def univariate_categorical(self):
        print("\n===== 📌 UNIVARIATE — CATEGORICAL =====\n")
        for col in self.cat_cols:
            print(f"[CAT] {col}")
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Explode list columns if needed
            if self.df[col].apply(lambda x: isinstance(x, list)).any():
                df_exp = self.df.explode(col)
            else:
                df_exp = self.df

            # Count per category using size()
            top_vals = df_exp.groupby(col).size().nlargest(20).reset_index(name='count')
            least_vals = df_exp.groupby(col).size().nsmallest(20).reset_index(name='count')

            axes[0].bar(top_vals[col], top_vals['count'], color="skyblue")
            axes[0].set_title(f"Top 20 - {col}")
            axes[0].tick_params(axis='x', rotation=90)

            axes[1].bar(least_vals[col], least_vals['count'], color="lightpink")
            axes[1].set_title(f"Least 20 - {col}")
            axes[1].tick_params(axis='x', rotation=90)

            fig.suptitle(f"Category Per Top/Less for {col}", fontsize=16)
            plt.tight_layout()
            plt.show()

    # -----------------------------------------------------------
    # 🔹 2. UNIVARIATE ANALYSIS — Numerical
    # -----------------------------------------------------------
    def univariate_numerical(self, exclude=[]):
        print("\n===== 📌 UNIVARIATE — NUMERICAL =====\n")
        cols = [c for c in self.num_cols if c not in exclude]
        for col in cols:
            print(f"[NUM] {col}")
            count_df = self.df.groupby(col).size().reset_index(name='count')
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(count_df[col], count_df['count'], color="skyblue")
            ax.set_title(f"Count of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            plt.xticks(rotation=90)

            # Add value labels
            for index, value in enumerate(count_df['count']):
                ax.text(index, value, str(value), ha='center', va='bottom')

            plt.tight_layout()
            plt.show()
            print("")

    # -----------------------------------------------------------
    # 🔹 3. BIVARIATE ANALYSIS
    # -----------------------------------------------------------
    def bivariate(self, cat_cols=[], num_cols=[]):
        print("\n===== 📌 BIVARIATE ANALYSIS =====\n")
        
        # Categorical
        for col in cat_cols:
            # تجاهل العمود 'name' تمامًا
            if col == 'name' or col not in self.df.columns:
                continue
            print(f"[BI-CAT] {col}")
            
            # Explode list columns if needed
            if self.df[col].apply(lambda x: isinstance(x, list)).any():
                df_exp = self.df.explode(col)
            else:
                df_exp = self.df

            group_cols = [col]
            if 'score' in df_exp.columns:
                group_cols.append('score')

            bi = df_exp.groupby(group_cols).size().reset_index(name='count')
            color_col = 'score' if 'score' in df_exp.columns else None

            px.histogram(
                bi, x=col, y='count', color=color_col,
                histfunc="sum", template="plotly_dark",
                text_auto=True, barmode="group",
                title=f"{col} vs score" if color_col else f"{col} Counts"
            ).show()

        # Numerical
        for col in num_cols:
            if col not in self.df.columns:
                continue
            print(f"[BI-NUM] {col}")
            
            group_cols = [col]
            if 'score' in self.df.columns:
                group_cols.append('score')

            bi = self.df.groupby(group_cols).size().reset_index(name='count')
            color_col = 'score' if 'score' in self.df.columns else None

            px.histogram(
                bi, x=col, y='count', color=color_col,
                histfunc="sum", template="plotly_dark",
                text_auto=True, barmode="group",
                title=f"{col} vs score" if color_col else f"{col} Counts"
            ).show()

    # -----------------------------------------------------------
    # 🔹 4. MULTIVARIATE ANALYSIS
    # -----------------------------------------------------------
    def multivariate(self):
        print("\n===== 📌 MULTIVARIATE ANALYSIS =====\n")
        for col in self.cat_cols:
            if col not in self.df.columns:
                continue
            print(f"[Multi-CAT] {col}")
            if 'score' not in self.df.columns:
                continue

            top_cat = self.df.groupby([col, 'score'])['approx_cost'].mean().nlargest(20).reset_index()
            less_cat = self.df.groupby([col, 'score'])['approx_cost'].mean().nsmallest(20).reset_index()
            px.histogram(
                top_cat, x=col, y='approx_cost', color='score',
                histfunc='sum', template='plotly_dark',
                text_auto=True, barmode='group',
                title=f"Top 20 of {col} by approx_cost"
            ).show()
            px.histogram(
                less_cat, x=col, y='approx_cost', color='score',
                histfunc='sum', template='plotly_dark',
                text_auto=True, barmode='group',
                title=f"Least 20 of {col} by approx_cost"
            ).show()

    # -----------------------------------------------------------
    # 🔹 5. DISTRIBUTION ANALYSIS
    # -----------------------------------------------------------
    def distribution(self, dist_cols=["rate", "votes", "approx_cost"]):
        print("\n===== 📌 DISTRIBUTIONS =====\n")
        plt.figure(figsize=(20, 4))
        for i, col in enumerate(dist_cols):
            if col not in self.df.columns:
                continue
            plt.subplot(1, 3, i+1)
            sns.histplot(self.df[col], kde=True, bins=20, color="r")
            plt.title(f"Distribution of {col}")
        plt.show()

    # -----------------------------------------------------------
    # 🔹 6. CORRELATION ANALYSIS
    # -----------------------------------------------------------
    def correlation(self):
        print("\n===== 📌 CORRELATION =====\n")
        if 'score' not in self.df.columns:
            return None
        corr = self.df.select_dtypes("number").corr()['score'].sort_values(ascending=False).reset_index()
        corr = corr.rename(columns={'index': 'feature'})
        print(corr)
        corr.plot.bar(x='feature', y='score', figsize=(12, 4), color='green')
        plt.title("Correlation with Score")
        plt.show()
        return corr

    # -----------------------------------------------------------
    # 🔹 RUN COMPLETE PIPELINE
    # -----------------------------------------------------------
    def run_all(self):
        self.univariate_categorical()
        self.univariate_numerical(exclude=['rate', 'votes', 'approx_cost', 'RowCount'])
        self.bivariate(
            cat_cols=['name','cuisines','location','listed_city','rest_type','listed_type'],
            num_cols=['votes_Range', 'rateRange', 'approx_cost_Range']
        )
        self.multivariate()
        self.distribution()
        corr = self.correlation()
        return corr

# ============================================================
# 1) HELPER FUNCTIONS
# ============================================================

def RATE(x):
    if pd.isna(x): return float("nan")
    s = str(x).strip()
    if s.upper() in {"NEW", "NAN", "-"}:
        return float("nan")
    s = s.replace(" ", "")
    if "/" in s:
        try: return float(s.split("/")[0])
        except: return float("nan")
    try: return float(s)
    except: return float("nan")


def PHONE_count(x):
    if pd.isna(x): return 0
    parts = re.split(r"[\n\r;]+", str(x))
    parts = [p.strip() for p in parts if p.strip()]
    return len(parts)


def PHONE_binary(x):
    return 0 if pd.isna(x) or str(x).strip() == "" else 1


def MENU_binary(x):
    if pd.isna(x): return 0
    s = str(x).strip()
    return 0 if s in {"[]", "[] "} else 1


def AppCost(x):
    if pd.isna(x): return float("nan")
    x = str(x).replace(",", "").strip()
    try: return float(x)
    except: return float("nan")


# ============================================================
# 2) FEATURE ENGINEERING PIPELINE
# ============================================================

class FeatureEngineeringPipeline:
    """
    Feature Engineering Pipeline — Restaurant Success Prediction
    ------------------------------------------------------------
    Success Definition (as required by the business description):
        SUCCESS = 1  if rate >= 3.75
                 = 0  otherwise

    Why fixed threshold?
    --------------------
    - This aligns with the official project requirement (Rating-based).
    - Keeps the meaning of target interpretable and constant.
    - Avoids score-based shifting of success definition when data changes.
    """

    def __init__(self, enable_logging=True, success_threshold: float = 3.75):
        self.scaler = RobustScaler()
        self.threshold = success_threshold   # Fixed business rule
        if enable_logging:
            logging.basicConfig(level=logging.INFO,
                                format='[%(levelname)s] %(message)s')
        self._init_groups()

    # =========================================================
    # Cuisine & RestType Groups
    # =========================================================
    def _init_groups(self):
        self.cuisine_groups = {
            'Total_North_Indian': ['Awadhi','Bihari','Kashmiri','Lucknowi','Mughlai','Punjabi','Rajasthani'],
            'Total_South_Indian': ['Andhra','Chettinad','Hyderabadi','Kerala','Mangalorean','Tamil','Udupi'],
            'Total_East_Indian':  ['Assamese','Bengali','Oriya'],
            'Total_West_Indian':  ['Gujarati','Maharashtrian','Goan','Sindhi'],
            'Total_International':['American','British','French','German','Greek','Italian','Japanese','Lebanese',
                                   'Mediterranean','Mexican','Portuguese','Spanish','Turkish','Vietnamese','Russian',
                                   'South American','Sri Lankan','Tibetan','Middle Eastern','Asian','European'],
            'Total_Asian': ['Burmese','Cantonese','Chinese','Indonesian','Korean','Mongolian','Nepali',
                            'North Eastern','Pan Asian','Singaporean','Thai','Tibetan','Vietnamese','Japanese'],
            'Total_Grill_BBQ_Bar': ['BBQ','Bar Food','Charcoal Chicken','Grill','Kebab','Roast Chicken','Rolls','Steak'],
            'Total_Fast_Food': ['Burger','Fast Food','Finger Food','Hot dogs','Sandwich','Street Food','Wraps','Tex-Mex'],
            'Total_Beverages_Desserts': ['Beverages','Bubble Tea','Coffee','Desserts','Ice Cream','Juices','Mithai','Paan','Tea'],
            'Total_Healthy_Fusion': ['Healthy Food','Vegan','Modern Indian','Salad'],
            'Total_Bakery': ['Bakery','Parsi','Cake','Sweets']
        }

        self.rest_type_groups = {
            'Rest_SweetOrBakery': ['Bakery','Confectionery','Dessert Parlor','Sweet Shop'],
            'Rest_Drink_Oriented_Establishments': ['Bar','Club','Lounge','Microbrewery','Pub'],
            'Rest_Specialty_Shops': ['Beverage Shop','Bhojanalya','Food Truck','Irani Cafee','Kiosk','Meat Shop'],
            'Rest_Dining_Establishments': ['Cafe','Casual Dining','Dhaba','Fine Dining','Food Court','Mess','Quick Bites'],
            'Rest_Takeaway_and_Delivery': ['Delivery','Takeaway']
        }

    # =========================================================
    # Helper Utilities
    # =========================================================
    def _safe_split(self, val):
        try: return [x.strip() for x in val.split(',')]
        except: return []

    def _group_counter(self, items, groups_dict):
        counts = {name: 0 for name in groups_dict}
        for itm in items:
            for group, group_items in groups_dict.items():
                if itm in group_items:
                    counts[group] += 1
        return counts

    # =========================================================
    # Target Creation — Fixed Business Rule
    # =========================================================
    def create_target(self, df, training=True):
        logging.info(f"Applying success threshold: rate >= {self.threshold}")
        
        df['target'] = (df['rate'] >= self.threshold).astype(int)

        # Reporting Target Distribution to help class-balance understanding
        positives = df['target'].mean()
        logging.info(f"Success ratio: {positives:.4f} "
                     f"({df['target'].sum()} successful out of {len(df)})")

        return df

    # =========================================================
    # Feature Transformations
    # =========================================================
    def build_bins(self, df):
        logging.info("Creating bins ...")

        df['rateRange'] = pd.cut(df['rate'], bins=[0,1,2,3,4,5],
                                 labels=['0-1','1-2','2-3','3-4','4-5'])

        vote_bins = [df['votes'].min(),1000,3000,5000,7000,9000,11000,13000,15000,df['votes'].max()]
        vote_lbls = ['0-1000','1000-3000','3000-5000','5000-7000','7000-9000','9000-11000',
                     '11000-13000','13000-15000','15000-17000']
        df['votesRange'] = pd.cut(df['votes'], bins=vote_bins,
                                  labels=vote_lbls).fillna('0-1000')

        cost_bins = [0,100,200,500,1000,2000,3000,4000,5000,6000]
        cost_lbls = ['0-100','100-200','200-500','500-1000','1000-2000',
                     '2000-3000','3000-4000','4000-5000','5000-6000']
        df['approx_cost_range'] = pd.cut(df['approx_cost'], bins=cost_bins,
                                         labels=cost_lbls)

        return df

    def transform_cuisines(self, df):
        counts_df = df['cuisines'].apply(
            lambda x: self._group_counter(self._safe_split(x), self.cuisine_groups)
        ).apply(pd.Series)
        return pd.concat([df, counts_df], axis=1)

    def transform_rest_type(self, df):
        rest_df = df['rest_type'].apply(
            lambda x: self._group_counter(self._safe_split(x), self.rest_type_groups)
        ).apply(pd.Series)
        return pd.concat([df, rest_df], axis=1)

    # =========================================================
    # Main Pipeline Entry
    # =========================================================
    def transform(self, df, training=True):
        df = df.copy()
        logging.info("=== Feature Engineering Pipeline START ===")

        df = self.create_target(df, training)
        df = self.build_bins(df)
        df = self.transform_cuisines(df)
        df = self.transform_rest_type(df)

        logging.info("=== Feature Engineering Pipeline DONE ===")
        return df

# ============================================================
# 3) DEEP CHECKING
# ============================================================

def deep_checking(df: pd.DataFrame, essential_cols: list, fe_instance, training: bool):
    logger.info("\n===== RUNNING DEEP CHECKING BLOCK =====")
    duplicated = df.columns[df.columns.duplicated()].tolist()
    if duplicated: raise ValueError(f"Duplicate columns detected: {duplicated}")
    logger.info("✔ No duplicate columns detected")
    missing_report = {}
    for col in essential_cols:
        if col not in df: logger.warning(f"⚠ Essential column missing → {col}"); continue
        miss = df[col].isna().sum()
        if miss > 0: missing_report[col] = int(miss)
    if missing_report:
        logger.warning("⚠ Missing values in essential columns:")
        for c, v in missing_report.items(): logger.warning(f"   - {c} → {v} rows missing")
    else: logger.info("✔ No essential missing values found")
    if not training:
        if fe_instance.threshold is not None:
            positives = df["target"].mean()
            logger.info(f"Target positive ratio = {positives:.4f}")
            if positives < 0.20 or positives > 0.80:
                logger.warning(f"⚠ Target distribution is imbalanced (ratio={positives:.3f})")
        else: raise ValueError("Threshold missing for inference mode")
    logger.info("===== DEEP CHECKING DONE =====\n")
    return df


# ============================================================
# 4) RAW LOADING & BASIC CLEANING
# ============================================================

def load_data(raw_path: str) -> pd.DataFrame:
    logger.info("Loading raw dataset ...")
    df = pd.read_csv(raw_path)
    logger.info(f"Loaded raw: {df.shape}")
    return df

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    logger.info("Basic Cleaning ...")
    df = df.rename(columns={
        "approx_cost(for two people)": "approx_cost",
        "listed_in(type)": "listed_type",
        "listed_in(city)": "listed_city"
    })
    drop_cols = ["url", "address", "dish_liked", "reviews_list"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    before = df.shape
    df = df.drop_duplicates()
    logger.info(f"Duplicates removed: {before} → {df.shape}")
    return df

def transform_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    logger.info("Transforming base columns ...")
    if "rate" in df.columns:
        df["rate_parsed"] = df["rate"].astype(str).apply(RATE)
        df = df.dropna(subset=["rate_parsed"])
        df["rate"] = df["rate_parsed"]
        df = df.drop(columns=["rate_parsed"])
    if "votes" in df.columns:
        df["votes"] = pd.to_numeric(df["votes"], errors="coerce")
        df = df.dropna(subset=["votes"])
        df["votes"] = df["votes"].astype(int)
    if "approx_cost" in df.columns:
        df["approx_cost"] = df["approx_cost"].apply(AppCost)
        df = df.dropna(subset=["approx_cost"])
        df["approx_cost"] = df["approx_cost"].astype(float)
    if "phone" in df.columns:
        df["phone_lines"] = df["phone"].apply(PHONE_count)
        df["phone"] = df["phone"].apply(PHONE_binary)
    else:
        df["phone"] = 1
        df["phone_lines"] = 0
    if "menu_item" in df.columns:
        df["menu_item"] = df["menu_item"].apply(MENU_binary)
    else:
        df["menu_item"] = 1
    return df


def process_business_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    logger.info("Processing business fields ...")

    # =============================================================
    # 🔹 1) Validate & Normalize 'cuisines'
    # =============================================================
    if "cuisines" in df.columns:
        before = len(df)

        # Ensure string → list → cleanup → remove invalid
        df["cuisines"] = (
            df["cuisines"]
            .astype(str)
            .str.split(",")
            .apply(lambda lst: [c.strip() for c in lst if c.strip()])
        )

        # Remove rows with 0 cuisines detected
        df = df[df["cuisines"].map(len) > 0]
        removed = before - len(df)
        if removed > 0:
            logger.warning(f"Removed {removed} rows with no valid cuisines")

        # Convert list → string again for FE grouping
        df["cuisines"] = df["cuisines"].apply(lambda lst: ", ".join(lst))

    # =============================================================
    # 🔹 2) Validate & Normalize 'rest_type'
    # =============================================================
    if "rest_type" in df.columns:
        before = len(df)

        df["rest_type"] = (
            df["rest_type"]
            .astype(str)
            .str.split(",")
            .apply(lambda lst: [r.strip() for r in lst if r.strip()])
        )

        df = df[df["rest_type"].map(len) > 0]
        removed = before - len(df)
        if removed > 0:
            logger.warning(f"Removed {removed} rows with no valid rest_type")

        df["rest_type"] = df["rest_type"].apply(lambda lst: ", ".join(lst))

    # =============================================================
    # 🔹 3) City Consistency Check
    # =============================================================
    if "listed_city" in df.columns and "location" in df.columns:
        df["location_city"] = df["location"].astype(str).apply(
            lambda x: x.split(",")[-1].strip()
        )
        df["city_match"] = (df["location_city"] == df["listed_city"]).astype(int)

    # =============================================================
    # 🔹 4) Final business rules check report
    # =============================================================
    logger.info(f"Processing business fields DONE → Shape: {df.shape}")

    return df

# ============================================================
# 📌 Save preprocessed dataset
# ============================================================
def save_preprocessed(df: pd.DataFrame, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    logger.info(f"Saved preprocessed dataset → {save_path}")


# ============================================================
# 📌 Split dataframe into Train / Test and save with timestamp
# ============================================================
def split_train_test(df: pd.DataFrame, base_save_path: str, test_size: float = 0.2, random_seed: int = 42):
    """
    Split dataframe into train and test sets and save them as CSV
    in the same folder as base_save_path with timestamped filenames.
    """
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_seed, shuffle=True)

    folder = os.path.dirname(base_save_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_path = os.path.join(folder, f"train_{timestamp}.csv")
    test_path = os.path.join(folder, f"test_{timestamp}.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Saved train set → {train_path} ({train_df.shape})")
    logger.info(f"Saved test set → {test_path} ({test_df.shape})")

    return train_df, test_df


# ============================================================
# 5) END-TO-END PIPELINE
# ============================================================
def run_pipeline(
    raw_path: str = "data/raw/zomato.csv",
    save_path: str = "data/preprocessed/zomato_preprocessed.csv",
    training: bool = True,
    run_eda: bool = True,       # Option to run EDA
    split_data: bool = True,    # Option to split dataset
    test_size: float = 0.2,
    random_seed: int = 42
):
    logger.info("\n======= RUNNING FULL PRODUCTION PIPELINE =======\n")

    # Load and clean data
    df = load_data(raw_path)
    df = basic_cleaning(df)
    df = transform_columns(df)
    df = process_business_fields(df)

    # Feature Engineering
    fe = FeatureEngineeringPipeline()
    df = fe.transform(df, training=training)

    # Deep Checking
    essential_cols = ["rate", "votes", "approx_cost", "cuisines", "rest_type", "target"]
    df = deep_checking(df, essential_cols, fe, training)

    # Optional EDA
    if run_eda:
        logger.info("\n=== RUNNING EDA PIPELINE ===")
        eda = EDAPipeline(df)
        eda.run_all()

    # Save preprocessed dataset
    logger.info(f"\n[FINAL] Data Shape After FE: {df.shape}")
    save_preprocessed(df, save_path)

    # Split into train/test with timestamped filenames
    if split_data:
        split_train_test(df, save_path, test_size=test_size, random_seed=random_seed)

    logger.info("\n======= PIPELINE COMPLETED SUCCESSFULLY =======\n")
    return df


# ============================================================
# 6) MAIN (Local Execution)
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-eda", action="store_true", help="Skip running EDA")
    args = parser.parse_args()

    raw = r"C:\Users\user\Downloads\project_structure\data\raw\zomato.csv"
    save = r"C:\Users\user\Downloads\project_structure\data\preprocessed\zomato_preprocessed.csv"

    run_pipeline(
        raw_path=raw,
        save_path=save,
        split_data=True,
        test_size=0.2,
        random_seed=42,
        run_eda=not args.no_eda  
    )

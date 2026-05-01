"""
Dummy Data Generator for CTR Prediction Project
================================================
Simulates impression and click logs from a content recommendation platform.

Output files:
  - users.csv          : User profile data
  - items.csv          : Content/item metadata
  - impressions.csv    : Served impressions with labels (clicked or not)
  - clicks.csv         : Click events with dwell time

Design notes:
  - CTR is intentionally imbalanced (~3-5%) like real systems
  - Click probability depends on user-item affinity (real signal to learn)
  - Includes noise, missing values, position bias, and time-based patterns
  - Scales: tweak N_USERS, N_ITEMS, N_DAYS at the top
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

# Reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ------------------------------------------------------------------
# Configuration - tweak these to scale the dataset
# ------------------------------------------------------------------
N_USERS = 50_000
N_ITEMS = 5_000
N_DAYS = 30
AVG_IMPRESSIONS_PER_USER_PER_DAY = 8
OUTPUT_DIR = "data/raw"

CATEGORIES = [
    "tech", "sports", "politics", "entertainment", "finance",
    "lifestyle", "travel", "food", "science", "gaming"
]
DEVICES = ["mobile", "desktop", "tablet"]
COUNTRIES = ["IN", "US", "UK", "CA", "AU", "DE", "FR", "JP", "BR", "SG"]
SOURCES = [f"source_{i}" for i in range(50)]  # publishers/authors

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 1. Generate users
# ------------------------------------------------------------------
print(f"Generating {N_USERS:,} users...")

users = pd.DataFrame({
    "user_id": [f"u_{i:07d}" for i in range(N_USERS)],
    "signup_date": [
        datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 700))
        for _ in range(N_USERS)
    ],
    "country": np.random.choice(COUNTRIES, N_USERS, p=[0.25, 0.30, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]),
    "age_bucket": np.random.choice(
        ["18-24", "25-34", "35-44", "45-54", "55+", None],
        N_USERS,
        p=[0.20, 0.30, 0.25, 0.15, 0.05, 0.05]  # 5% missing - realistic
    ),
    "gender": np.random.choice(["M", "F", "O", None], N_USERS, p=[0.45, 0.45, 0.02, 0.08]),
})

# Each user has a latent preference vector over categories (their "taste")
# This is the SIGNAL the model should learn
user_prefs = np.random.dirichlet(np.ones(len(CATEGORIES)) * 0.5, N_USERS)
user_pref_df = pd.DataFrame(user_prefs, columns=[f"pref_{c}" for c in CATEGORIES])
users = pd.concat([users.reset_index(drop=True), user_pref_df], axis=1)

# User activity level (how engaged they are overall) - log-normal distribution
users["activity_level"] = np.clip(np.random.lognormal(0, 0.6, N_USERS), 0.1, 5.0)

users.to_csv(f"{OUTPUT_DIR}/users.csv", index=False)
print(f"  -> users.csv ({len(users):,} rows)")

# ------------------------------------------------------------------
# 2. Generate items
# ------------------------------------------------------------------
print(f"Generating {N_ITEMS:,} items...")

items = pd.DataFrame({
    "item_id": [f"i_{i:06d}" for i in range(N_ITEMS)],
    "category": np.random.choice(CATEGORIES, N_ITEMS),
    "source": np.random.choice(SOURCES, N_ITEMS),
    "publish_date": [
        datetime(2024, 6, 1) + timedelta(
            days=np.random.randint(0, 500),
            hours=np.random.randint(0, 24)
        )
        for _ in range(N_ITEMS)
    ],
    "title_length": np.random.randint(20, 120, N_ITEMS),  # chars
    "has_image": np.random.choice([0, 1], N_ITEMS, p=[0.15, 0.85]),
    "has_video": np.random.choice([0, 1], N_ITEMS, p=[0.7, 0.3]),
})

# Latent "item quality" - some items are inherently more clickable
# Drawn from beta dist so most items are average, few are great or bad
items["quality_score"] = np.random.beta(2, 5, N_ITEMS)  # mean ~0.28

items.to_csv(f"{OUTPUT_DIR}/items.csv", index=False)
print(f"  -> items.csv ({len(items):,} rows)")

# ------------------------------------------------------------------
# 3. Generate impressions & clicks
# ------------------------------------------------------------------
print(f"\nGenerating impressions over {N_DAYS} days...")
print("(This is the main loop - takes 30-60 seconds for default config)")

# Build category lookup for items (faster than dataframe indexing in loop)
item_lookup = items.set_index("item_id").to_dict("index")
user_lookup = users.set_index("user_id").to_dict("index")

# Pre-compute category index for quick affinity lookup
cat_to_idx = {c: i for i, c in enumerate(CATEGORIES)}

impressions = []
clicks = []
imp_id = 0
click_id = 0

start_date = datetime(2025, 9, 1)

for day_offset in range(N_DAYS):
    current_date = start_date + timedelta(days=day_offset)
    day_of_week = current_date.weekday()  # 0 = Monday

    # Weekend traffic boost (realistic seasonality)
    daily_multiplier = 1.2 if day_of_week >= 5 else 1.0

    # Sample which users are active today (not all users every day)
    n_active = int(N_USERS * 0.4 * daily_multiplier)  # 40% DAU on weekdays
    active_users = np.random.choice(users["user_id"].values, n_active, replace=False)

    if day_offset % 5 == 0:
        print(f"  Day {day_offset+1}/{N_DAYS} - {n_active:,} active users")

    for user_id in active_users:
        u = user_lookup[user_id]
        # Number of impressions for this user today (Poisson around their activity level)
        n_imps = max(1, np.random.poisson(AVG_IMPRESSIONS_PER_USER_PER_DAY * u["activity_level"]))
        n_imps = min(n_imps, 50)  # cap

        # Sample items shown to this user (random with some quality bias)
        # Real systems would have a candidate generator here; we approximate
        item_probs = np.array([item_lookup[iid]["quality_score"] for iid in items["item_id"]])
        item_probs = item_probs / item_probs.sum()
        shown_items = np.random.choice(items["item_id"].values, n_imps, replace=False, p=item_probs)

        # Session timestamp - random hour during the day
        # Hourly distribution (peak in evening)
        hourly_probs = np.array([0.01, 0.01, 0.005, 0.005, 0.01, 0.02, 0.04, 0.06,
                                  0.07, 0.08, 0.07, 0.06, 0.06, 0.06, 0.06, 0.06,
                                  0.07, 0.08, 0.08, 0.07, 0.06, 0.04, 0.02, 0.015])
        hourly_probs = hourly_probs / hourly_probs.sum()
        session_start_hour = np.random.choice(range(24), p=hourly_probs)
        session_id = f"s_{user_id}_{current_date.strftime('%Y%m%d')}_{session_start_hour}"

        for position, item_id in enumerate(shown_items):
            it = item_lookup[item_id]

            # Compute timestamp
            ts = current_date + timedelta(
                hours=int(session_start_hour),
                minutes=position * 2 + np.random.randint(0, 3),
                seconds=np.random.randint(0, 60)
            )

            # ====================================================
            # CORE CLICK PROBABILITY MODEL
            # This is the underlying truth the ML model must learn
            # ====================================================

            # 1. User-item affinity (the main signal)
            cat_idx = cat_to_idx[it["category"]]
            affinity = u[f"pref_{it['category']}"]  # 0 to 1, user's pref for this cat

            # 2. Item quality
            quality = it["quality_score"]

            # 3. Position bias (top items get clicked more)
            position_factor = 1.0 / (1 + position * 0.15)

            # 4. Item freshness decay
            item_age_days = (ts - it["publish_date"]).total_seconds() / 86400
            freshness = np.exp(-item_age_days / 30)  # half-life ~3 weeks

            # 5. Has image/video lift
            media_boost = 1 + 0.2 * it["has_image"] + 0.15 * it["has_video"]

            # 6. Time-of-day effect (evenings have higher CTR)
            hour = ts.hour
            tod_boost = 1.1 if 18 <= hour <= 23 else 1.0

            # Combine - this gives us click probability
            base_ctr = 0.04  # ~4% baseline
            click_prob = (
                base_ctr
                * (1 + 4 * affinity)
                * (0.5 + 1.5 * quality)
                * position_factor
                * (0.5 + 0.5 * freshness)
                * media_boost
                * tod_boost
            )

            # Add noise + clamp
            click_prob = np.clip(click_prob + np.random.normal(0, 0.005), 0.001, 0.5)

            # Sample click
            clicked = np.random.random() < click_prob

            impressions.append({
                "impression_id": f"imp_{imp_id:010d}",
                "user_id": user_id,
                "item_id": item_id,
                "session_id": session_id,
                "timestamp": ts.isoformat(),
                "position": position,
                "device": np.random.choice(DEVICES, p=[0.65, 0.30, 0.05]),
                "country": u["country"],
                "clicked": int(clicked),
            })
            imp_id += 1

            # If clicked, generate a click event with dwell time
            if clicked:
                # Dwell time: longer if affinity high (more interesting content)
                # Mix of "real" reads and quick bounces
                if np.random.random() < 0.15:
                    # Accidental click / quick bounce
                    dwell_seconds = np.random.uniform(0.5, 3)
                else:
                    # Real engagement, log-normal
                    dwell_seconds = np.clip(
                        np.random.lognormal(3 + 2 * affinity, 0.8),
                        2, 600
                    )

                click_ts = ts + timedelta(seconds=np.random.randint(1, 30))
                clicks.append({
                    "click_id": f"clk_{click_id:09d}",
                    "impression_id": f"imp_{imp_id-1:010d}",
                    "user_id": user_id,
                    "item_id": item_id,
                    "click_timestamp": click_ts.isoformat(),
                    "dwell_seconds": round(dwell_seconds, 2),
                    "scrolled": int(np.random.random() < (0.3 + 0.4 * affinity)),
                })
                click_id += 1

print(f"\nWriting impressions ({len(impressions):,} rows)...")
imp_df = pd.DataFrame(impressions)
imp_df.to_csv(f"{OUTPUT_DIR}/impressions.csv", index=False)

print(f"Writing clicks ({len(clicks):,} rows)...")
clk_df = pd.DataFrame(clicks)
clk_df.to_csv(f"{OUTPUT_DIR}/clicks.csv", index=False)

# ------------------------------------------------------------------
# Summary stats
# ------------------------------------------------------------------
print("\n" + "="*60)
print("DATA GENERATION COMPLETE")
print("="*60)
print(f"Users         : {len(users):>12,}")
print(f"Items         : {len(items):>12,}")
print(f"Impressions   : {len(imp_df):>12,}")
print(f"Clicks        : {len(clk_df):>12,}")
print(f"Overall CTR   : {len(clk_df) / len(imp_df) * 100:>11.2f}%")
print(f"Date range    : {imp_df['timestamp'].min()[:10]} to {imp_df['timestamp'].max()[:10]}")

print("\nCTR by category:")
merged = imp_df.merge(items[["item_id", "category"]], on="item_id")
ctr_by_cat = merged.groupby("category")["clicked"].agg(["mean", "count"])
ctr_by_cat.columns = ["ctr", "impressions"]
ctr_by_cat["ctr"] = (ctr_by_cat["ctr"] * 100).round(2)
print(ctr_by_cat.sort_values("ctr", ascending=False).to_string())

print("\nCTR by position (top 10):")
ctr_by_pos = imp_df[imp_df["position"] < 10].groupby("position")["clicked"].mean() * 100
print(ctr_by_pos.round(2).to_string())

print(f"\nFiles saved in: {OUTPUT_DIR}/")
print("Done!")

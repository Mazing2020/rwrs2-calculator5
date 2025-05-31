
import streamlit as st
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import KBinsDiscretizer

# === TRAIN THE MODEL INLINE (no pickles or external files) ===

# Simulated training data (100 samples, 9 features)
X_train = np.random.rand(100, 9) * 100
y_train = np.random.rand(100) * 100

# Bin encoder for draft score
draft_encoder = KBinsDiscretizer(n_bins=4, encode='onehot-dense', strategy='uniform')
draft_scores = X_train[:, [0]]  # first feature = draft score
draft_encoder.fit(draft_scores)

# Gradient Boosting Model
model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# === SCORING FUNCTIONS ===

def draft_score(pick):
    return (1 - pick / 256) * 100

def early_declare_score(early):
    return 100 if early else 50

def breakout_age_score(age):
    if age <= 19:
        return 100
    elif age == 20:
        return 80
    elif age == 21:
        return 60
    else:
        return 40

# === STREAMLIT UI ===

st.title("RWRS²: Rookie WR Success Score")
st.write("Fill in the rookie WR's data to calculate the score:")

draft_pick = st.number_input("Draft Pick (1–256)", min_value=1, max_value=256, value=22)
early_declare = st.selectbox("Declared Early?", ["Yes", "No"]) == "Yes"
breakout_age = st.slider("Breakout Age", 18, 23, 20)
dominator = st.slider("College Dominator (0–100)", 0, 100, 30)
athleticism = st.slider("Athleticism (0–100)", 0, 100, 88)
route_running = st.slider("Route Running (0–100)", 0, 100, 95)
landing_spot = st.slider("Landing Spot Fit (0–100)", 0, 100, 95)

# Calculate individual scores
draft = draft_score(draft_pick)
early = early_declare_score(early_declare)
breakout = breakout_age_score(breakout_age)

# Feature vector
features = [
    draft,
    early,
    breakout,
    dominator,
    athleticism,
    route_running,
    landing_spot
]

# Add interaction + draft bin encoding
interaction = breakout * athleticism
draft_bin = draft_encoder.transform(np.array([[draft]]))  # shape (1, 4)

# Final input: 7 base + 1 interaction + 1 bin total (average of 4 bins → pick 1)
X_input = np.concatenate([features, [interaction], draft_bin[0][:1]]).reshape(1, -1)

# Predict
if st.button("Calculate RWRS² Score"):
    try:
        score = model.predict(X_input)[0]
        st.success(f"RWRS² Score: {round(score, 2)} (Lower is Better)")
    except ValueError as e:
        st.error(f"Error in model prediction: {str(e)}")

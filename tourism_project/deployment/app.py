
import streamlit as st
import pandas as pd
import pickle
import os
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Visit With Us — Wellness Package Predictor",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Visit With Us")
st.subheader("Wellness Tourism Package — Purchase Predictor")
st.write("Fill in the customer details below to predict whether they will purchase the Wellness Tourism Package.")
st.markdown("---")

# Load model and feature columns from Hugging Face Model Hub
@st.cache_resource
def load_model_and_features():
    HF_TOKEN   = os.environ.get("HF_TOKEN", None)
    MODEL_REPO = "sriramrs1804/visit-with-us-wellness-model"
    model_path    = hf_hub_download(repo_id=MODEL_REPO, filename="model.pkl",    token=HF_TOKEN)
    features_path = hf_hub_download(repo_id=MODEL_REPO, filename="features.pkl", token=HF_TOKEN)
    with open(model_path,    "rb") as f: model    = pickle.load(f)
    with open(features_path, "rb") as f: features = pickle.load(f)
    return model, features

model, feature_cols = load_model_and_features()

st.markdown("### 👤 Customer Details")
col1, col2 = st.columns(2)
with col1:
    age                   = st.number_input("Age", min_value=18, max_value=100, value=35)
    city_tier             = st.selectbox("City Tier", [1, 2, 3])
    monthly_income        = st.number_input("Monthly Income (INR)", min_value=1000, max_value=200000, value=30000)
    num_trips             = st.number_input("Avg Number of Trips/Year", min_value=0, max_value=20, value=2)
    num_person_visiting   = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    num_children_visiting = st.number_input("Number of Children (<5 yrs)", min_value=0, max_value=5, value=0)
    preferred_star        = st.slider("Preferred Property Stars", 1, 5, 3)
with col2:
    type_of_contact = st.selectbox("Type of Contact",  ["Company Invited", "Self Enquiry"])
    occupation      = st.selectbox("Occupation",        ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender          = st.selectbox("Gender",            ["Male", "Female"])
    marital_status  = st.selectbox("Marital Status",    ["Single", "Married", "Divorced"])
    designation     = st.selectbox("Designation",       ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    product_pitched = st.selectbox("Product Pitched",   ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    passport        = st.selectbox("Has Passport?",     [0, 1], format_func=lambda x: "Yes" if x else "No")
    own_car         = st.selectbox("Owns a Car?",       [0, 1], format_func=lambda x: "Yes" if x else "No")

st.markdown("### 📞 Interaction Details")
col3, col4 = st.columns(2)
with col3:
    pitch_score   = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    num_followups = st.number_input("Number of Followups", min_value=0, max_value=10, value=2)
with col4:
    duration_of_pitch = st.number_input("Duration of Pitch (mins)", min_value=1, max_value=120, value=20)

# Label encoding maps — must match training encoding exactly
ENCODE_MAP = {
    "TypeofContact":  {"Company Invited": 0, "Self Enquiry": 1},
    "Occupation":     {"Free Lancer": 0, "Large Business": 1, "Salaried": 2, "Small Business": 3},
    "Gender":         {"Female": 0, "Male": 1},
    "MaritalStatus":  {"Divorced": 0, "Married": 1, "Single": 2},
    "Designation":    {"AVP": 0, "Executive": 1, "Manager": 2, "Senior Manager": 3, "VP": 4},
    "ProductPitched": {"Basic": 0, "Deluxe": 1, "King": 2, "Standard": 3, "Super Deluxe": 4},
}

st.markdown("---")
if st.button("🔍 Predict Purchase", use_container_width=True):
    # Compute engineered features from user inputs
    income_per_person   = round(monthly_income / num_person_visiting, 2)
    pitch_effectiveness = round(pitch_score * num_followups, 2)
    family_size         = num_person_visiting + num_children_visiting
    is_frequent         = int(num_trips > 3)  # 3 is the median from training data

    # Build input dict matching training feature order
    input_data = {
        "Age":                      age,
        "TypeofContact":            ENCODE_MAP["TypeofContact"][type_of_contact],
        "CityTier":                 city_tier,
        "DurationOfPitch":          duration_of_pitch,
        "Occupation":               ENCODE_MAP["Occupation"][occupation],
        "Gender":                   ENCODE_MAP["Gender"][gender],
        "NumberOfPersonVisiting":   num_person_visiting,
        "NumberOfFollowups":        num_followups,
        "ProductPitched":           ENCODE_MAP["ProductPitched"][product_pitched],
        "PreferredPropertyStar":    preferred_star,
        "MaritalStatus":            ENCODE_MAP["MaritalStatus"][marital_status],
        "NumberOfTrips":            num_trips,
        "Passport":                 passport,
        "PitchSatisfactionScore":   pitch_score,
        "OwnCar":                   own_car,
        "NumberOfChildrenVisiting": num_children_visiting,
        "Designation":              ENCODE_MAP["Designation"][designation],
        "MonthlyIncome":            monthly_income,
        "IncomePerPerson":          income_per_person,
        "PitchEffectiveness":       pitch_effectiveness,
        "FamilySize":               family_size,
        "IsFrequentTraveller":      is_frequent,
    }

    # Create DataFrame and align column order to training
    input_df    = pd.DataFrame([input_data])[feature_cols]
    prediction  = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("### 📊 Prediction Result")
    if prediction == 1:
        st.success("✅ LIKELY TO PURCHASE the Wellness Tourism Package")
        st.info(f"📈 Purchase Probability: **{probability * 100:.1f}%**")
    else:
        st.warning("❌ UNLIKELY TO PURCHASE the Wellness Tourism Package")
        st.info(f"📉 Purchase Probability: **{probability * 100:.1f}%**")

    with st.expander("🔎 View Input Data Sent to Model"):
        st.dataframe(input_df)

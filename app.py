import streamlit as st
import pandas as pd
import joblib


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💼",
    layout="wide"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #666;
    margin-bottom: 30px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    background-color: #ffffff;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.10);
}

.result-text {
    font-size: 32px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODEL
# ==========================================

try:

    model = joblib.load("salary_model.pkl")

    label_encoders = joblib.load(
        "label_encoders.pkl"
    )

except FileNotFoundError:

    st.error(
        "Model files not found. "
        "Run main.py first."
    )

    st.stop()


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="title">💼 Employee Salary Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Decision Tree Machine Learning System'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("📌 Project Information")

st.sidebar.info(
    """
    This application predicts an
    employee's salary category using
    a Decision Tree Classifier.

    Categories:

    🔵 Low

    🟡 Medium

    🟢 High
    """
)


# ==========================================
# INPUT SECTION
# ==========================================

st.subheader("👤 Employee Details")

col1, col2 = st.columns(2)


# Department
with col1:

    department = st.selectbox(
        "Department",
        label_encoders["Department"].classes_
    )


# Experience
with col2:

    experience = st.number_input(
        "Experience (Years)",
        min_value=0.0,
        max_value=50.0,
        value=2.0,
        step=1.0
    )


# Education
with col1:

    education = st.selectbox(
        "Education Level",
        label_encoders["Education_Level"].classes_
    )


# Age
with col2:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=70,
        value=25,
        step=1
    )


# Gender
with col1:

    gender = st.selectbox(
        "Gender",
        label_encoders["Gender"].classes_
    )


# City
with col2:

    city = st.selectbox(
        "City",
        label_encoders["City"].classes_
    )


# ==========================================
# PREDICTION BUTTON
# ==========================================

st.write("")

predict_button = st.button(
    "🔮 Predict Salary Category",
    use_container_width=True
)


# ==========================================
# PREDICTION
# ==========================================

if predict_button:

    input_data = pd.DataFrame({
        "Department": [department],
        "Experience_Years": [experience],
        "Education_Level": [education],
        "Age": [age],
        "Gender": [gender],
        "City": [city]
    })


    # Encode categorical columns

    for column, encoder in label_encoders.items():

        input_data[column] = encoder.transform(
            input_data[column]
        )


    # Prediction

    prediction = model.predict(
        input_data
    )[0]


    # Display numerical prediction

    st.success(
        f"Prediction Code: {prediction}"
    )


    # Convert prediction code
    # according to LabelEncoder order

    if prediction == 0:

        category = "High"

    elif prediction == 1:

        category = "Low"

    else:

        category = "Medium"


    # ======================================
    # RESULT
    # ======================================

    st.markdown(
        f"""
        <div class="result-box">

        <div class="result-text">
        🎯 Predicted Salary Category
        </div>

        <h1>{category}</h1>

        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "Employee Salary Prediction System | "
    "Machine Learning | Decision Tree"
)


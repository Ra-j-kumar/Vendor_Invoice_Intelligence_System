import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

FREIGHT_MODEL_PATH = BASE_DIR / 'freight_cost_prediction' / 'models' / 'predict_freight_model.pkl'
INVOICE_MODEL_PATH = BASE_DIR / 'invoice_flagging' / 'models' / 'predict_flag_invoice.pkl'
INVOICE_SCALER_PATH = BASE_DIR / 'invoice_flagging' / 'models' / 'scaler.pkl'

INVOICE_FEATURES = [
    'invoice_quantity',
    'invoice_dollars',
    'Freight',
    'total_item_quantity',
    'total_item_dollars'
]


# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title='Vendor Invoice Intelligence Portal',
    page_icon='📊',
    layout='wide'
)


# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>

.main {
    background-color: #0f172a;
}

.stApp {
    background: linear-gradient(to right, #0f172a, #111827);
    color: white;
}

h1, h2, h3 {
    color: white !important;
}

[data-testid="stMetric"] {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #334155;
    text-align: center;
}

.stButton>button {
    width: 100%;
    border-radius: 12px;
    height: 3em;
    background-color: #2563eb;
    color: white;
    font-size: 16px;
    font-weight: bold;
    border: none;
}

.stButton>button:hover {
    background-color: #1d4ed8;
    transition: 0.3s;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODELS ---------------- #

def load_model(model_path: Path):
    with open(model_path, 'rb') as f:
        return joblib.load(f)


freight_model = load_model(FREIGHT_MODEL_PATH)
invoice_model = load_model(INVOICE_MODEL_PATH)
invoice_scaler = load_model(INVOICE_SCALER_PATH)


# ---------------- PREDICTION FUNCTIONS ---------------- #

def predict_freight_cost(dollars: float):
    data = pd.DataFrame({'Dollars': [dollars]})
    prediction = freight_model.predict(data)
    return float(prediction[0])


def predict_invoice_flag(data: dict):
    df = pd.DataFrame([data])

    scaled = invoice_scaler.transform(df[INVOICE_FEATURES])
    prediction = invoice_model.predict(scaled)

    df['Predicted_Flag'] = prediction
    return df


# ---------------- HEADER ---------------- #

st.title("📊 Vendor Invoice Intelligence System")

st.markdown("""
### AI-Driven Freight Cost Prediction & Invoice Risk Detection

This platform helps finance teams:

✅ Predict freight costs accurately  
✅ Detect suspicious invoices instantly  
✅ Reduce operational losses and manual work  
""")

st.divider()


# ---------------- SIDEBAR ---------------- #

st.sidebar.title("⚙️ Navigation")

selected_model = st.sidebar.radio(
    "Choose Module",
    ['Freight Cost Prediction', 'Invoice Flag Prediction']
)

st.sidebar.markdown("---")

st.sidebar.info("""
### 🚀 Business Benefits

- Better Cost Forecasting  
- Fraud & Risk Detection  
- Faster Finance Operations  
- AI-Powered Decision Support  
""")


# ---------------- FREIGHT COST ---------------- #

if selected_model == 'Freight Cost Prediction':

    st.header("🚚 Freight Cost Prediction")

    col1, col2 = st.columns([2, 1])

    with col1:

        dollars = st.number_input(
            'Invoice Amount ($)',
            min_value=0.0,
            value=1000.0,
            step=100.0
        )

        if st.button('Predict Freight Cost'):

            cost = predict_freight_cost(dollars)

            st.success("Prediction Completed Successfully ✅")

            st.metric(
                label="Predicted Freight Cost",
                value=f"${cost:,.2f}"
            )

    with col2:

        st.markdown("""
        ### 📌 Prediction Insights

        Higher invoice values generally lead to:

        - Increased freight charges
        - Higher logistics costs
        - More transportation overhead
        """)


# ---------------- INVOICE FLAG ---------------- #

else:

    st.header("🛡️ Invoice Risk Flag Prediction")

    col1, col2 = st.columns(2)

    with col1:

        invoice_quantity = st.number_input(
            'Invoice Quantity',
            min_value=0,
            value=1
        )

        invoice_dollars = st.number_input(
            'Invoice Dollars',
            min_value=0.0,
            value=1000.0,
            step=10.0
        )

        freight = st.number_input(
            'Freight Cost',
            min_value=0.0,
            value=50.0,
            step=1.0
        )

    with col2:

        total_item_quantity = st.number_input(
            'Total Item Quantity',
            min_value=0,
            value=invoice_quantity
        )

        total_item_dollars = st.number_input(
            'Total Item Dollars',
            min_value=0.0,
            value=invoice_dollars,
            step=10.0
        )

    st.markdown("")

    if st.button('Predict Invoice Risk'):

        data = {
            'invoice_quantity': invoice_quantity,
            'invoice_dollars': invoice_dollars,
            'Freight': freight,
            'total_item_quantity': total_item_quantity,
            'total_item_dollars': total_item_dollars,
        }

        result = predict_invoice_flag(data)

        flag = int(result['Predicted_Flag'].iloc[0])

        st.subheader("📈 Prediction Result")

        if flag == 1:

            st.error("⚠️ High Risk Invoice Detected")

            st.metric(
                label="Risk Status",
                value="RISKY"
            )

        else:

            st.success("✅ Invoice Looks Safe")

            st.metric(
                label="Risk Status",
                value="SAFE"
            )

        st.dataframe(result, use_container_width=True)
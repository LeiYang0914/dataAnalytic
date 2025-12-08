import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Revenue Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .info-box {
        background-color: black;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    h1 {
        color: #1f1f1f;
        font-weight: 700;
    }
    h2, h3 {
        color: #4a4a4a;
    }
    .success-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    .prediction-text {
        font-size: 24px;
        font-weight: bold;
        margin: 10px 0;
    }
    .probability-text {
        font-size: 48px;
        font-weight: bold;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("rf_model.pkl")

model = load_model()

FEATURES = [
    'Administrative', 'Administrative_Duration',
    'ProductRelated', 'ProductRelated_Duration',
    'BounceRates', 'ExitRates', 
    'SpecialDay', 'Month_num', 'OperatingSystems',
    'Browser', 'Region', 'TrafficType',
    'VisitorType_num', 'Weekend_num'
]

# -----------------------------
# Header
# -----------------------------
st.title("🛒 Online Shopper Revenue Prediction")
st.markdown("### Predict whether a visitor will make a purchase using machine learning")
st.markdown("---")

# Info box
st.markdown("""
    <div class="info-box">
        <b>ℹ️ How it works:</b> Enter session details below and click 'Predict' to see if this visitor 
        is likely to generate revenue. The model uses Random Forest algorithm trained on historical shopping data.
    </div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar for Quick Stats/Info
# -----------------------------
with st.sidebar:
    st.header("📊 About This Tool")
    st.markdown("""
    This predictor analyzes:
    - **Page visits** and durations
    - **Bounce & exit rates**
    - **Visitor behavior** patterns
    - **Technical details** (OS, browser)
    - **Temporal factors** (month, weekend)
    
    **Model:** Random Forest Classifier
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.info("Higher product-related page visits and lower bounce rates typically increase purchase likelihood.")

# -----------------------------
# Input Form with Columns
# -----------------------------
st.header("📝 Session Information")

# Tab layout for better organization
tab1, tab2, tab3, tab4 = st.tabs(["📄 Page Activity", "📈 Engagement Metrics", "🖥️ Technical Details", "📅 Temporal Info"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Administrative Pages")
        Administrative = st.number_input(
            "Number of pages visited", 
            min_value=0, step=1, value=0, 
            key="admin_pages",
            help="Count of administrative pages visited"
        )
        Administrative_Duration = st.number_input(
            "Time spent (seconds)", 
            min_value=0.0, step=1.0, value=0.0,
            key="admin_duration",
            help="Total duration on administrative pages"
        )
    
    with col2:
        st.subheader("Product Pages")
        ProductRelated = st.number_input(
            "Number of pages visited", 
            min_value=0, step=1, value=0,
            key="product_pages",
            help="Count of product-related pages visited"
        )
        ProductRelated_Duration = st.number_input(
            "Time spent (seconds)", 
            min_value=0.0, step=1.0, value=0.0,
            key="product_duration",
            help="Total duration on product pages"
        )

with tab2:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        BounceRates = st.slider(
            "Bounce Rate", 
            min_value=0.0, max_value=1.0, step=0.001, value=0.05,
            help="Percentage of visitors who leave after viewing only one page"
        )
    
    with col2:
        ExitRates = st.slider(
            "Exit Rate", 
            min_value=0.0, max_value=1.0, step=0.001, value=0.05,
            help="Percentage of pageviews that were the last in the session"
        )
    
    with col3:
        SpecialDay = st.slider(
            "Special Day Proximity", 
            min_value=0.0, max_value=1.0, step=0.01, value=0.0,
            help="Closeness to special days (0=far, 1=very close)"
        )

with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        OperatingSystems = st.selectbox(
            "Operating System", 
            options=list(range(1, 9)),
            help="Visitor's operating system ID"
        )
        Browser = st.selectbox(
            "Browser", 
            options=list(range(1, 14)),
            help="Visitor's browser ID"
        )
    
    with col2:
        Region = st.selectbox(
            "Region", 
            options=list(range(1, 10)),
            help="Geographic region ID"
        )
        TrafficType = st.selectbox(
            "Traffic Type", 
            options=list(range(1, 21)),
            help="Traffic source type ID"
        )

with tab4:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        month_choice = st.selectbox("Month", month_labels)
        Month_num = month_labels.index(month_choice) + 1
    
    with col2:
        visitor_type_label = st.selectbox(
            "Visitor Type", 
            ["New Visitor", "Returning Visitor"],
            help="Is this a new or returning visitor?"
        )
        VisitorType_num = 1 if visitor_type_label == "Returning Visitor" else 0
    
    with col3:
        weekend_label = st.selectbox(
            "Weekend Visit?", 
            ["No", "Yes"],
            help="Did the session occur on weekend?"
        )
        Weekend_num = 1 if weekend_label == "Yes" else 0

# -----------------------------
# Build Input DataFrame
# -----------------------------
input_dict = {
    'Administrative': Administrative,
    'Administrative_Duration': Administrative_Duration,
    'ProductRelated': ProductRelated,
    'ProductRelated_Duration': ProductRelated_Duration,
    'BounceRates': BounceRates,
    'ExitRates': ExitRates,
    'SpecialDay': SpecialDay,
    'Month_num': Month_num,
    'OperatingSystems': OperatingSystems,
    'Browser': Browser,
    'Region': Region,
    'TrafficType': TrafficType,
    'VisitorType_num': VisitorType_num,
    'Weekend_num': Weekend_num
}

input_df = pd.DataFrame([input_dict], columns=FEATURES)

# -----------------------------
# Prediction Section
# -----------------------------
st.markdown("---")
st.header("🎯 Make Prediction")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔮 Predict Revenue Potential", use_container_width=True)

if predict_button:
    with st.spinner("Analyzing session data..."):
        pred_class = model.predict(input_df)[0]
        
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_df)[0][1]
        else:
            prob = None
    
    st.markdown("---")
    
    # Display results with attractive styling
    if pred_class == 1:
        st.markdown(f"""
            <div class="success-box">
                <h2>✅ Prediction: LIKELY TO PURCHASE</h2>
                <p class="prediction-text">This session shows strong purchase intent!</p>
                {"<p class='probability-text'>" + f"{prob:.1%}" + "</p>" if prob is not None else ""}
                <p>Revenue Probability</p>
            </div>
        """, unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown(f"""
            <div class="success-box" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <h2>❌ Prediction: UNLIKELY TO PURCHASE</h2>
                <p class="prediction-text">This session shows low purchase intent</p>
                {"<p class='probability-text'>" + f"{prob:.1%}" + "</p>" if prob is not None else ""}
                <p>Revenue Probability</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Additional insights
    st.subheader("📊 Session Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Pages", Administrative + ProductRelated)
    with col2:
        st.metric("Total Duration", f"{Administrative_Duration + ProductRelated_Duration:.0f}s")
    with col3:
        st.metric("Bounce Rate", f"{BounceRates:.1%}")
    with col4:
        st.metric("Visitor Type", visitor_type_label)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #888; padding: 20px;'>
        <p>Built with Streamlit 🎈 | Powered by Random Forest ML 🤖</p>
    </div>
""", unsafe_allow_html=True)
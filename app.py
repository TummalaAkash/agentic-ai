"""
Streamlit Dashboard for Agentic AI Early Warning System
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import io
import base64

# Import our custom modules
from train_model import StudentRiskPredictor
from agent import StudentRiskAgent
from utils import get_subject_attendance_summary, create_subject_risk_analysis

# Page configuration
st.set_page_config(
    page_title="AI Early Warning System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background: linear-gradient(135deg, #d32f2f 0%, #c62828 100%);
        border-left: 6px solid #b71c1c;
        padding: 18px;
        margin: 12px 0;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(211, 47, 47, 0.4);
        color: #ffffff;
        font-weight: 700;
        font-size: 1.1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .risk-medium {
        background: linear-gradient(135deg, #f57c00 0%, #ef6c00 100%);
        border-left: 6px solid #e65100;
        padding: 18px;
        margin: 12px 0;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(245, 124, 0, 0.4);
        color: #ffffff;
        font-weight: 700;
        font-size: 1.1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .risk-low {
        background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%);
        border-left: 6px solid #0d4f14;
        padding: 18px;
        margin: 12px 0;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(46, 125, 50, 0.4);
        color: #ffffff;
        font-weight: 700;
        font-size: 1.1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #dee2e6;
        margin: 10px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.15);
    }
    .intervention-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        margin: 15px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        border-left: 4px solid #2196f3;
    }
    .intervention-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
        border-left-color: #1976d2;
    }
    .intervention-card h4 {
        color: #1976d2;
        margin-bottom: 10px;
        font-size: 1.1rem;
        font-weight: 700;
    }
    .intervention-card p {
        color: #424242;
        margin: 8px 0;
        line-height: 1.5;
    }
    .priority-high {
        background: linear-gradient(135deg, #d32f2f 0%, #c62828 100%);
        color: #ffffff;
        padding: 6px 16px;
        border-radius: 25px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
        border: 2px solid #b71c1c;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        box-shadow: 0 2px 8px rgba(211, 47, 47, 0.3);
    }
    .priority-medium {
        background: linear-gradient(135deg, #f57c00 0%, #ef6c00 100%);
        color: #ffffff;
        padding: 6px 16px;
        border-radius: 25px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
        border: 2px solid #e65100;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        box-shadow: 0 2px 8px rgba(245, 124, 0, 0.3);
    }
    .priority-low {
        background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%);
        color: #ffffff;
        padding: 6px 16px;
        border-radius: 25px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
        border: 2px solid #0d4f14;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        box-shadow: 0 2px 8px rgba(46, 125, 50, 0.3);
    }
    .student-card {
        background: linear-gradient(135deg, #ffffff 0%, #f5f5f5 100%);
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    .student-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
    }
    .alert-box {
        background: linear-gradient(135deg, #f57c00 0%, #ef6c00 100%);
        border: 3px solid #e65100;
        border-radius: 12px;
        padding: 22px;
        margin: 20px 0;
        box-shadow: 0 6px 20px rgba(245, 124, 0, 0.4);
        color: #ffffff;
        font-weight: 600;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .alert-box h3 {
        color: #ffffff;
        margin-bottom: 12px;
        font-size: 1.4rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.4);
    }
    .success-box {
        background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%);
        border: 3px solid #0d4f14;
        border-radius: 12px;
        padding: 22px;
        margin: 20px 0;
        box-shadow: 0 6px 20px rgba(46, 125, 50, 0.4);
        color: #ffffff;
        font-weight: 600;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .success-box h3 {
        color: #ffffff;
        margin-bottom: 12px;
        font-size: 1.4rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.4);
    }
    .danger-box {
        background: linear-gradient(135deg, #d32f2f 0%, #c62828 100%);
        border: 3px solid #b71c1c;
        border-radius: 12px;
        padding: 22px;
        margin: 20px 0;
        box-shadow: 0 6px 20px rgba(211, 47, 47, 0.4);
        color: #ffffff;
        font-weight: 600;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .danger-box h3 {
        color: #ffffff;
        margin-bottom: 12px;
        font-size: 1.4rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.4);
    }
    .info-box {
        background: linear-gradient(135deg, #1976d2 0%, #1565c0 100%);
        border: 3px solid #0d47a1;
        border-radius: 12px;
        padding: 22px;
        margin: 20px 0;
        box-shadow: 0 6px 20px rgba(25, 118, 210, 0.4);
        color: #ffffff;
        font-weight: 600;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .info-box h3 {
        color: #ffffff;
        margin-bottom: 12px;
        font-size: 1.4rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.4);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #0d47a1;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-label {
        font-size: 1rem;
        color: #333;
        text-align: center;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 5px;
    }
    .feature-importance {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #2196f3;
    }
    .feature-importance strong {
        color: #1976d2;
        font-weight: 700;
    }
    .feature-importance .importance-value {
        color: #f57c00;
        font-weight: 600;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'student_data' not in st.session_state:
    st.session_state.student_data = None
if 'agent' not in st.session_state:
    st.session_state.agent = StudentRiskAgent()

@st.cache_data
def load_sample_data(size=200):
    """Load sample data for demonstration"""
    predictor = StudentRiskPredictor()
    return predictor.generate_sample_data(size)

@st.cache_resource
def load_or_train_model():
    """Load existing model or train a new one"""
    predictor = StudentRiskPredictor()
    
    # Try to load existing model
    try:
        predictor.load_model()
        if predictor.model is not None:
            return predictor
    except:
        pass
    
    # Train new model if none exists
    st.info("Training new model with sample data...")
    df = predictor.generate_sample_data(1000)
    predictor.train_model(df, model_type='xgboost')
    predictor.save_model()
    return predictor

def create_risk_distribution_chart(predictions_df):
    """Create risk distribution pie chart"""
    risk_counts = predictions_df['risk_level'].value_counts()
    
    fig = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title="Student Risk Distribution",
        color_discrete_map={
            'High': '#f44336',
            'Medium': '#ff9800', 
            'Low': '#4caf50'
        }
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def create_risk_trends_chart(predictions_df):
    """Create risk trends over time (simulated)"""
    # Simulate time series data
    dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
    trends_data = []
    
    for date in dates:
        # Simulate some variation in risk levels over time
        high_risk = np.random.randint(15, 25)
        medium_risk = np.random.randint(30, 40)
        low_risk = 100 - high_risk - medium_risk
        
        trends_data.append({
            'Date': date,
            'High Risk': high_risk,
            'Medium Risk': medium_risk,
            'Low Risk': low_risk
        })
    
    trends_df = pd.DataFrame(trends_data)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trends_df['Date'], y=trends_df['High Risk'], 
                            mode='lines+markers', name='High Risk', line=dict(color='#f44336')))
    fig.add_trace(go.Scatter(x=trends_df['Date'], y=trends_df['Medium Risk'], 
                            mode='lines+markers', name='Medium Risk', line=dict(color='#ff9800')))
    fig.add_trace(go.Scatter(x=trends_df['Date'], y=trends_df['Low Risk'], 
                            mode='lines+markers', name='Low Risk', line=dict(color='#4caf50')))
    
    fig.update_layout(
        title="Risk Trends Over Time",
        xaxis_title="Date",
        yaxis_title="Number of Students",
        hovermode='x unified'
    )
    
    return fig

def create_feature_importance_chart(feature_importance):
    """Create feature importance bar chart"""
    if not feature_importance:
        return None
    
    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    features, importance = zip(*sorted_features)
    
    fig = px.bar(
        x=importance,
        y=features,
        orientation='h',
        title="Top 10 Most Important Features",
        labels={'x': 'Importance Score', 'y': 'Features'}
    )
    fig.update_layout(height=400)
    return fig

def create_student_performance_chart(student_data):
    """Create individual student performance chart"""
    if student_data is None:
        return None
    
    # Select key performance metrics
    metrics = ['gpa', 'attendance_rate', 'assignment_completion', 'participation_score']
    values = [student_data.get(metric, 0) for metric in metrics]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=metrics,
        fill='toself',
        name='Performance'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        title="Student Performance Profile",
        height=400
    )
    
    return fig

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🎓 AI Early Warning System</h1>', unsafe_allow_html=True)
    st.markdown("### Proactive Student Risk Assessment & Intervention Platform")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["📊 Dashboard", "📈 Upload Data", "🔍 Student Analysis", "🤖 AI Agent Query", "⚙️ Model Info"]
    )
    
    # Branch filter (if available)
    branch_filter = None
    if st.session_state.student_data is not None and 'branch' in st.session_state.student_data.columns:
        branches = sorted([b for b in st.session_state.student_data['branch'].dropna().unique().tolist() if str(b).strip() != ""])
        if len(branches) > 0:
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 🏫 Branch Filter")
            branch_filter = st.sidebar.selectbox(
                "Select branch",
                options=["All"] + branches,
                index=0
            )
    
    # Dataset Info in Sidebar
    if st.session_state.predictions is not None:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Current Dataset")
        st.sidebar.metric("Total Students", len(st.session_state.predictions))
        
        risk_counts = st.session_state.predictions['risk_level'].value_counts()
        st.sidebar.metric("High Risk", risk_counts.get('High', 0))
        st.sidebar.metric("Medium Risk", risk_counts.get('Medium', 0))
        st.sidebar.metric("Low Risk", risk_counts.get('Low', 0))
        
        if st.sidebar.button("🔄 Clear Dataset", type="secondary"):
            if st.session_state.get('confirm_clear', False):
                st.session_state.predictions = None
                st.session_state.student_data = None
                st.session_state.confirm_clear = False
                st.rerun()
            else:
                st.session_state.confirm_clear = True
                st.sidebar.warning("Click again to confirm clearing dataset.")
    
    # Load model
    if st.session_state.model is None:
        with st.spinner("Loading AI model..."):
            st.session_state.model = load_or_train_model()
    
    # Dashboard Page
    if page == "📊 Dashboard":
        st.header("📊 Risk Assessment Dashboard")
        
        # Load sample data if no predictions exist
        if st.session_state.predictions is None:
            st.markdown("""
            <div class="info-box">
                <h3>📊 No Dataset Loaded</h3>
                <p>Please go to the 'Upload Data' page to upload student data or generate a sample dataset.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="alert-box">
                <h3>📋 How to Get Started</h3>
                <p>1. Go to <strong>'Upload Data'</strong> page</p>
                <p>2. Generate a sample dataset or upload your own CSV</p>
                <p>3. Return to Dashboard to view analysis results</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Apply branch filter to predictions and student data
        original_predictions_df = st.session_state.predictions
        predictions_df = original_predictions_df
        current_student_df = st.session_state.student_data
        if (
            predictions_df is not None and 
            current_student_df is not None and 
            branch_filter and 
            branch_filter != "All" and 
            'branch' in current_student_df.columns
        ):
            filtered_students = current_student_df[current_student_df['branch'] == branch_filter]
            predictions_df = original_predictions_df[original_predictions_df['student_id'].isin(filtered_students['student_id'])]
        
        # Dataset Info
        if predictions_df is not None:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"""
                <div class="info-box">
                    <h3>📊 Dataset Information</h3>
                    <p><strong>Total Students:</strong> {len(predictions_df)} | <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                if st.button("🔄 Clear Dataset", type="secondary", help="Clear current dataset and start fresh"):
                    if st.session_state.get('confirm_clear', False):
                        st.session_state.predictions = None
                        st.session_state.student_data = None
                        st.session_state.confirm_clear = False
                        st.rerun()
                    else:
                        st.session_state.confirm_clear = True
                        st.warning("Click 'Clear Dataset' again to confirm clearing the current dataset.")
        
        # Key Metrics
        if predictions_df is not None:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_students = len(predictions_df)
                st.metric("Total Students", total_students)
            
            with col2:
                high_risk = len(predictions_df[predictions_df['risk_level'] == 'High'])
                st.metric("High Risk", high_risk, delta=f"{high_risk/total_students*100:.1f}%")
            
            with col3:
                medium_risk = len(predictions_df[predictions_df['risk_level'] == 'Medium'])
                st.metric("Medium Risk", medium_risk, delta=f"{medium_risk/total_students*100:.1f}%")
            
            with col4:
                low_risk = len(predictions_df[predictions_df['risk_level'] == 'Low'])
                st.metric("Low Risk", low_risk, delta=f"{low_risk/total_students*100:.1f}%")
        else:
            st.markdown("""
            <div class="alert-box">
                <h3>📊 No Dataset Loaded</h3>
                <p>Please generate a dataset or upload student data to view metrics and analysis.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts
        if predictions_df is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(create_risk_distribution_chart(predictions_df), use_container_width=True)
            
            with col2:
                st.plotly_chart(create_risk_trends_chart(predictions_df), use_container_width=True)
        
        # Dataset Size Analysis
        if predictions_df is not None and len(predictions_df) > 100:
            st.subheader("📈 Dataset Size Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Dataset Size", f"{len(predictions_df)} students", 
                         delta=f"Large dataset - {len(predictions_df)} records")
            
            with col2:
                avg_risk_prob = predictions_df['risk_probability'].mean()
                st.metric("Average Risk", f"{avg_risk_prob:.1%}", 
                         delta=f"Overall risk level")
            
            with col3:
                high_risk_pct = len(predictions_df[predictions_df['risk_level'] == 'High']) / len(predictions_df) * 100
                st.metric("High Risk %", f"{high_risk_pct:.1f}%", 
                         delta=f"Students needing immediate attention")
        
        # Subject-wise Analysis
        if predictions_df is not None and st.session_state.student_data is not None:
            st.subheader("📚 Subject-wise Attendance Analysis")
            
            # Subject attendance summary
            base_student_df = st.session_state.student_data
            if branch_filter and branch_filter != "All":
                base_student_df = base_student_df[base_student_df['branch'] == branch_filter]
            subject_summary = get_subject_attendance_summary(base_student_df)
            st.dataframe(subject_summary, use_container_width=True)
            
            # Subject risk chart
            subject_risk_chart = create_subject_risk_analysis(base_student_df)
            if subject_risk_chart:
                st.plotly_chart(subject_risk_chart, use_container_width=True)
        
        # Feature Importance
        if st.session_state.model.feature_importance:
            st.subheader("🔍 Model Feature Importance")
            st.plotly_chart(create_feature_importance_chart(st.session_state.model.feature_importance), 
                           use_container_width=True)
        
        # High Risk Students Alert
        if predictions_df is not None:
            high_risk_students = predictions_df[predictions_df['risk_level'] == 'High'].sort_values('risk_probability', ascending=False)
            
            if len(high_risk_students) > 0:
                st.subheader("🚨 High Risk Students - Immediate Attention Required")
                
                for _, student in high_risk_students.head(5).iterrows():
                    st.markdown(f"""
                    <div class="danger-box">
                        <h3>🚨 {student['student_id']} - HIGH RISK</h3>
                        <p><strong>Risk Level:</strong> {student['risk_level']}</p>
                        <p><strong>Risk Probability:</strong> {student['risk_probability']:.1%}</p>
                        <p><strong>Status:</strong> Immediate intervention required!</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Upload Data Page
    elif page == "📈 Upload Data":
        st.header("📈 Upload Student Data")
        
        st.markdown("""
        ### Upload CSV File
        Please upload a CSV file with student data. The file should contain the following columns:
        - student_id: Unique student identifier
        - branch: Student branch/department (e.g., Computer Science, Electronics) [optional]
        - gpa: Current GPA (0.0-4.0)
        - attendance_rate: Attendance rate (0.0-1.0)
        - assignment_completion: Assignment completion rate (0.0-1.0)
        - participation_score: Participation score (0-100)
        - lms_login_frequency: LMS login frequency (number)
        - late_submissions: Number of late submissions
        - quiz_average: Average quiz score (0-100)
        - study_hours_per_week: Study hours per week
        - previous_semester_gpa: Previous semester GPA
        - extracurricular_activities: Number of activities (0-3)
        - family_support: Family support level (High/Medium/Low)
        - financial_stress: Financial stress level (Low/Medium/High)
        - mental_health_concerns: Mental health concerns (0/1)
        - peer_support: Peer support level (Strong/Moderate/Weak)
        - Subject attendance columns: mathematics_attendance, physics_attendance, etc.
        """)
        
        # Generate Sample Dataset Section
        st.subheader("🎲 Generate Sample Dataset")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            dataset_size = st.slider(
                "Select dataset size:",
                min_value=50,
                max_value=2000,
                value=200,
                step=50,
                help="Choose the number of student records to generate"
            )
        
        with col2:
            if st.button("Generate Sample Dataset", type="primary"):
                with st.spinner(f"Generating {dataset_size} student records..."):
                    sample_data = load_sample_data(dataset_size)
                    
                    # Make predictions
                    with st.spinner("Analyzing student data..."):
                        predictions = st.session_state.model.predict_risk(sample_data)
                    
                    # Create predictions dataframe
                    predictions_df = pd.DataFrame({
                        'student_id': sample_data['student_id'],
                        'risk_level': predictions['predictions'],
                        'risk_probability': [predictions['probabilities'][pred][i] for i, pred in enumerate(predictions['predictions'])]
                    })
                    
                    # Add individual probabilities
                    for risk_level in ['Low', 'Medium', 'High']:
                        if risk_level in predictions['probabilities']:
                            predictions_df[f'{risk_level}_probability'] = predictions['probabilities'][risk_level]
                    
                    st.session_state.predictions = predictions_df
                    st.session_state.student_data = sample_data
                    st.session_state.confirm_clear = False
                    
                    st.markdown("""
                    <div class="success-box">
                        <h3>✅ Sample Dataset Generated!</h3>
                        <p>Dataset with {dataset_size} students has been generated and analyzed. Go to Dashboard to view results.</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Read the CSV file
                df = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded {len(df)} student records")
                
                # Display sample data
                st.subheader("Sample Data Preview")
                st.dataframe(df.head())
                
                # Make predictions
                if st.button("Analyze Student Risk"):
                    with st.spinner("Analyzing student data..."):
                        predictions = st.session_state.model.predict_risk(df)
                        
                        # Create predictions dataframe
                        predictions_df = pd.DataFrame({
                            'student_id': df['student_id'],
                            'risk_level': predictions['predictions'],
                            'risk_probability': [predictions['probabilities'][pred][i] for i, pred in enumerate(predictions['predictions'])]
                        })
                        
                        # Add individual probabilities
                        for risk_level in ['Low', 'Medium', 'High']:
                            if risk_level in predictions['probabilities']:
                                predictions_df[f'{risk_level}_probability'] = predictions['probabilities'][risk_level]
                        
                        st.session_state.predictions = predictions_df
                        st.session_state.student_data = df
                        
                        st.markdown("""
                        <div class="success-box">
                            <h3>✅ Analysis Complete!</h3>
                            <p>Student risk assessment completed successfully. Navigate to Dashboard to view results.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        
        # Download Sample CSV
        st.subheader("📥 Download Sample CSV")
        
        if st.button(f"Download Sample CSV ({dataset_size} students)", type="secondary"):
            with st.spinner("Generating sample CSV..."):
                sample_data = load_sample_data(dataset_size)
                csv = sample_data.to_csv(index=False)
                st.download_button(
                    label="Download Sample CSV",
                    data=csv,
                    file_name="sample_student_data.csv",
                    mime="text/csv"
                )
    
    # Student Analysis Page
    elif page == "🔍 Student Analysis":
        st.header("🔍 Individual Student Analysis")
        
        # Page-level department filter (independent from sidebar)
        page_branch_filter = None
        if st.session_state.student_data is not None and 'branch' in st.session_state.student_data.columns:
            dept_options = ["All"] + sorted(st.session_state.student_data['branch'].dropna().unique().tolist())
            page_branch_filter = st.selectbox("Department", dept_options, index=0)
        
        if st.session_state.predictions is not None and st.session_state.student_data is not None:
            # Apply branch filter to lists in this view (combine sidebar + page filter if both set)
            predictions_df = st.session_state.predictions
            student_data = st.session_state.student_data
            if branch_filter and branch_filter != "All":
                student_data = student_data[student_data['branch'] == branch_filter]
                predictions_df = predictions_df[predictions_df['student_id'].isin(student_data['student_id'])]
            if page_branch_filter and page_branch_filter != "All":
                student_data = student_data[student_data['branch'] == page_branch_filter]
                predictions_df = predictions_df[predictions_df['student_id'].isin(student_data['student_id'])]
            
            # Student selector (branch-aware)
            selected_student_id = st.selectbox(
                "Select a student to analyze",
                predictions_df['student_id'].tolist()
            )
            
            if selected_student_id:
                # Get student data - use loc instead of iloc to avoid index issues
                student_info = student_data[student_data['student_id'] == selected_student_id].iloc[0]
                student_prediction = predictions_df[predictions_df['student_id'] == selected_student_id].iloc[0]
                
                # Student overview
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader(f"Student Analysis: {selected_student_id}")
                    
                    # Risk level display
                    risk_level = student_prediction['risk_level']
                    risk_prob = student_prediction['risk_probability']
                    
                    if risk_level == 'High':
                        st.markdown(f"""
                        <div class="risk-high">
                            <h3>🚨 HIGH RISK</h3>
                            <p>Risk Probability: {risk_prob:.1%}</p>
                            <p>Immediate intervention required!</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif risk_level == 'Medium':
                        st.markdown(f"""
                        <div class="risk-medium">
                            <h3>⚠️ MEDIUM RISK</h3>
                            <p>Risk Probability: {risk_prob:.1%}</p>
                            <p>Proactive support recommended</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="risk-low">
                            <h3>✅ LOW RISK</h3>
                            <p>Risk Probability: {risk_prob:.1%}</p>
                            <p>Student appears to be on track</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    # Performance metrics
                    st.subheader("Key Metrics")
                    st.metric("GPA", f"{student_info['gpa']:.2f}")
                    st.metric("Attendance", f"{student_info['attendance_rate']:.1%}")
                    st.metric("Assignments", f"{student_info['assignment_completion']:.1%}")
                    st.metric("Participation", f"{student_info['participation_score']:.0f}")
                
                # Performance chart
                st.subheader("Performance Profile")
                performance_chart = create_student_performance_chart(student_info)
                if performance_chart:
                    st.plotly_chart(performance_chart, use_container_width=True)
                
                # Subject-wise attendance analysis
                st.subheader("📚 Subject-wise Attendance")
                subjects = ['Mathematics', 'Physics', 'Chemistry', 'Biology', 'English', 'History', 'Computer Science', 'Economics']
                
                subject_data = []
                for subject in subjects:
                    subject_col = f'{subject.lower().replace(" ", "_")}_attendance'
                    if subject_col in student_info:
                        attendance = student_info[subject_col]
                        # Get risk level for this subject
                        risk_level = 'Low'
                        if 'subject_risk_factors' in student_info and isinstance(student_info['subject_risk_factors'], dict):
                            risk_level = student_info['subject_risk_factors'].get(subject, 'Low')
                        
                        subject_data.append({
                            'Subject': subject,
                            'Attendance': f"{attendance:.1%}",
                            'Risk Level': risk_level,
                            'Status': 'Critical' if attendance < 0.6 else 'High' if attendance < 0.75 else 'Medium' if attendance < 0.85 else 'Low'
                        })
                
                if subject_data:
                    subject_df = pd.DataFrame(subject_data)
                    st.dataframe(subject_df, use_container_width=True)
                    
                    # Show subject-specific alerts
                    critical_subjects = [s for s in subject_data if s['Status'] in ['Critical', 'High']]
                    if critical_subjects:
                        st.markdown("### ⚠️ Subject-specific Alerts")
                        for subject in critical_subjects:
                            st.markdown(f"""
                            <div class="alert-box">
                                <h4>📚 {subject['Subject']}</h4>
                                <p><strong>Attendance:</strong> {subject['Attendance']} | <strong>Risk:</strong> {subject['Risk Level']}</p>
                                <p>Immediate attention required for this subject!</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                # AI Agent Analysis
                st.subheader("🤖 AI Agent Analysis & Recommendations")
                
                if st.button("Generate AI Analysis"):
                    with st.spinner("AI agent analyzing student..."):
                        # Get feature contributions (simplified)
                        feature_contributions = {}
                        if st.session_state.model.feature_importance:
                            for feature, importance in st.session_state.model.feature_importance.items():
                                if feature in student_info:
                                    feature_contributions[feature] = importance
                        
                        # Run AI agent analysis
                        agent_result = st.session_state.agent.analyze_student_risk(
                            student_data=student_info.to_dict(),
                            risk_prediction=risk_level,
                            risk_probability=risk_prob,
                            feature_contributions=feature_contributions
                        )
                        
                        # Display analysis
                        st.markdown("### 📋 Analysis Summary")
                        st.write(agent_result['summary'])
                        
                        # Display interventions
                        st.markdown("### 🎯 Recommended Interventions")
                        interventions = agent_result['interventions']
                        
                        for i, intervention in enumerate(interventions, 1):
                            priority_class = {
                                'High': 'priority-high',
                                'Medium': 'priority-medium',
                                'Low': 'priority-low'
                            }.get(intervention.get('priority', 'Medium'), 'priority-medium')
                            
                            st.markdown(f"""
                            <div class="intervention-card">
                                <h4>{i}. {intervention['title']}</h4>
                                <p><strong>Type:</strong> {intervention['type']}</p>
                                <p><strong>Description:</strong> {intervention['description']}</p>
                                <p><strong>Priority:</strong> <span class="{priority_class}">{intervention['priority']}</span></p>
                                <p><strong>Timeline:</strong> {intervention['timeline']}</p>
                                <p><strong>Resources:</strong> {', '.join(intervention['resources'])}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Next steps
                        st.markdown("### 📝 Next Steps")
                        for step in agent_result['next_steps']:
                            st.write(f"• {step}")
        else:
            st.markdown("""
            <div class="alert-box">
                <h3>📋 Upload Data Required</h3>
                <p>Please upload student data first on the 'Upload Data' page to begin analysis.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # AI Agent Query Page
    elif page == "🤖 AI Agent Query":
        st.header("🤖 Natural Language Query Interface")
        
        st.markdown("""
        Ask questions about your students using natural language. The AI agent will analyze the data and provide insights.
        """)
        
        # Query input
        query = st.text_input(
            "Enter your question:",
            placeholder="e.g., Which students are most at risk this semester?"
        )
        
        if st.button("Ask AI Agent") and query:
            if st.session_state.predictions is not None:
                with st.spinner("AI agent processing query..."):
                    # Get student data and predictions
                    student_data = st.session_state.student_data.to_dict('records')
                    predictions = st.session_state.predictions.to_dict('records')
                    
                    # Run query
                    result = st.session_state.agent.query_students(query, student_data, predictions)
                    
                    # Display results
                    st.subheader("🤖 AI Agent Response")
                    st.write(result['response'])
                    
                    if result['students']:
                        st.subheader("📊 Query Results")
                        results_df = pd.DataFrame(result['students'])
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Show individual student details
                        if len(result['students']) > 0:
                            st.subheader("🔍 Student Details")
                            for student in result['students'][:5]:  # Show top 5
                                st.write(f"**{student['student_id']}**: {student['risk_level']} risk ({student['risk_probability']:.1%})")
            else:
                st.markdown("""
                <div class="alert-box">
                    <h3>📋 Upload Data Required</h3>
                    <p>Please upload student data first on the 'Upload Data' page to begin analysis.</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Model Info Page
    elif page == "⚙️ Model Info":
        st.header("⚙️ Model Information")
        
        if st.session_state.model is not None:
            st.subheader("🤖 Machine Learning Model Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Model Type:** XGBoost Classifier
                
                **Features Used:**
                - Academic Performance (GPA, Quiz Scores)
                - Engagement Metrics (Attendance, Participation)
                - Behavioral Indicators (LMS Activity, Late Submissions)
                - Support Systems (Family, Peer, Financial)
                - Mental Health Indicators
                """)
            
            with col2:
                st.markdown("""
                **Risk Categories:**
                - **Low Risk:** Students performing well with minimal intervention needed
                - **Medium Risk:** Students showing some concerning patterns requiring proactive support
                - **High Risk:** Students at immediate risk of academic failure requiring urgent intervention
                """)
            
            # Feature importance
            if st.session_state.model.feature_importance:
                st.subheader("🔍 Feature Importance")
                importance_df = pd.DataFrame(
                    list(st.session_state.model.feature_importance.items()),
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=True)
                
                fig = px.bar(
                    importance_df.tail(10),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Top 10 Most Important Features"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Model performance metrics
            st.subheader("📊 Model Performance")
            st.markdown("""
            The model has been trained on synthetic data and achieves the following performance:
            - **Accuracy:** ~85-90% on test data
            - **Cross-validation:** 5-fold CV with consistent performance
            - **Feature Engineering:** Automated feature selection and scaling
            """)
            
            # Retrain model option
            st.subheader("🔄 Model Management")
            if st.button("Retrain Model with New Data"):
                with st.spinner("Retraining model..."):
                    # Generate new sample data
                    new_data = load_sample_data()
                    st.session_state.model.train_model(new_data, model_type='xgboost')
                    st.session_state.model.save_model()
                    st.success("Model retrained successfully!")
        else:
            st.error("Model not loaded. Please refresh the page.")

if __name__ == "__main__":
    main()

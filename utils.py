"""
Utility functions for data processing and visualization
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

def validate_student_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate student data format and content
    
    Args:
        df: DataFrame with student data
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    required_columns = [
        'student_id', 'gpa', 'attendance_rate', 'assignment_completion',
        'participation_score', 'lms_login_frequency', 'late_submissions',
        'quiz_average', 'study_hours_per_week', 'previous_semester_gpa',
        'extracurricular_activities', 'family_support', 'financial_stress',
        'mental_health_concerns', 'peer_support'
    ]
    # 'branch' is optional; if present, validate allowed values
    
    errors = []
    
    # Check required columns
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Check data types and ranges
    if 'gpa' in df.columns:
        if not df['gpa'].between(0, 4).all():
            errors.append("GPA values must be between 0 and 4")
    
    if 'attendance_rate' in df.columns:
        if not df['attendance_rate'].between(0, 1).all():
            errors.append("Attendance rate must be between 0 and 1")
    
    if 'assignment_completion' in df.columns:
        if not df['assignment_completion'].between(0, 1).all():
            errors.append("Assignment completion must be between 0 and 1")
    
    if 'participation_score' in df.columns:
        if not df['participation_score'].between(0, 100).all():
            errors.append("Participation score must be between 0 and 100")
    
    # Check categorical values
    if 'branch' in df.columns:
        valid_branches = {
            'Computer Science', 'Electronics', 'Mechanical', 'Civil', 'Biotechnology'
        }
        invalid_branches = set(df['branch'].dropna().unique()) - valid_branches
        if invalid_branches:
            errors.append(f"Invalid branch values: {', '.join(invalid_branches)}")
    if 'family_support' in df.columns:
        valid_family = {'High', 'Medium', 'Low'}
        invalid_family = set(df['family_support'].unique()) - valid_family
        if invalid_family:
            errors.append(f"Invalid family_support values: {', '.join(invalid_family)}")
    
    if 'financial_stress' in df.columns:
        valid_financial = {'Low', 'Medium', 'High'}
        invalid_financial = set(df['financial_stress'].unique()) - valid_financial
        if invalid_financial:
            errors.append(f"Invalid financial_stress values: {', '.join(invalid_financial)}")
    
    if 'peer_support' in df.columns:
        valid_peer = {'Strong', 'Moderate', 'Weak'}
        invalid_peer = set(df['peer_support'].unique()) - valid_peer
        if invalid_peer:
            errors.append(f"Invalid peer_support values: {', '.join(invalid_peer)}")
    
    return len(errors) == 0, errors

def create_risk_heatmap(predictions_df: pd.DataFrame, student_data: pd.DataFrame) -> go.Figure:
    """
    Create a heatmap showing risk distribution across different student characteristics
    
    Args:
        predictions_df: DataFrame with risk predictions
        student_data: DataFrame with student characteristics
        
    Returns:
        Plotly figure
    """
    # Merge data
    merged_df = predictions_df.merge(student_data, on='student_id', how='left')
    
    # Create risk level mapping
    risk_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
    merged_df['risk_numeric'] = merged_df['risk_level'].map(risk_mapping)
    
    # Create pivot table for GPA vs Attendance
    pivot_data = merged_df.pivot_table(
        values='risk_numeric',
        index=pd.cut(merged_df['gpa'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']),
        columns=pd.cut(merged_df['attendance_rate'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']),
        aggfunc='mean'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale='RdYlGn_r',
        hoverongaps=False,
        text=pivot_data.values,
        texttemplate="%{text:.2f}",
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title="Risk Level Heatmap: GPA vs Attendance",
        xaxis_title="Attendance Rate",
        yaxis_title="GPA",
        height=400
    )
    
    return fig

def create_correlation_matrix(student_data: pd.DataFrame) -> go.Figure:
    """
    Create correlation matrix for numerical features
    
    Args:
        student_data: DataFrame with student data
        
    Returns:
        Plotly figure
    """
    # Select numerical columns
    numerical_cols = student_data.select_dtypes(include=[np.number]).columns
    corr_matrix = student_data[numerical_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values,
        texttemplate="%{text:.2f}",
        textfont={"size": 8}
    ))
    
    fig.update_layout(
        title="Feature Correlation Matrix",
        height=500,
        width=500
    )
    
    return fig

def create_risk_timeline(predictions_df: pd.DataFrame) -> go.Figure:
    """
    Create a timeline showing risk progression (simulated)
    
    Args:
        predictions_df: DataFrame with risk predictions
        
    Returns:
        Plotly figure
    """
    # Simulate timeline data
    dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
    
    # Simulate risk progression for different student groups
    high_risk_students = len(predictions_df[predictions_df['risk_level'] == 'High'])
    medium_risk_students = len(predictions_df[predictions_df['risk_level'] == 'Medium'])
    low_risk_students = len(predictions_df[predictions_df['risk_level'] == 'Low'])
    
    timeline_data = []
    for date in dates:
        # Simulate some variation
        high_risk_trend = high_risk_students + np.random.randint(-2, 3)
        medium_risk_trend = medium_risk_students + np.random.randint(-3, 4)
        low_risk_trend = low_risk_students + np.random.randint(-2, 3)
        
        timeline_data.append({
            'Date': date,
            'High Risk': max(0, high_risk_trend),
            'Medium Risk': max(0, medium_risk_trend),
            'Low Risk': max(0, low_risk_trend)
        })
    
    timeline_df = pd.DataFrame(timeline_data)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timeline_df['Date'],
        y=timeline_df['High Risk'],
        mode='lines+markers',
        name='High Risk',
        line=dict(color='#f44336', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=timeline_df['Date'],
        y=timeline_df['Medium Risk'],
        mode='lines+markers',
        name='Medium Risk',
        line=dict(color='#ff9800', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=timeline_df['Date'],
        y=timeline_df['Low Risk'],
        mode='lines+markers',
        name='Low Risk',
        line=dict(color='#4caf50', width=3)
    ))
    
    fig.update_layout(
        title="Student Risk Trends Over Time",
        xaxis_title="Date",
        yaxis_title="Number of Students",
        hovermode='x unified',
        height=400
    )
    
    return fig

def create_intervention_effectiveness_chart() -> go.Figure:
    """
    Create a chart showing intervention effectiveness (simulated)
    
    Returns:
        Plotly figure
    """
    interventions = [
        'Academic Tutoring',
        'Peer Mentoring',
        'Counseling Services',
        'Study Groups',
        'Financial Aid',
        'Mental Health Support',
        'Family Engagement',
        'Career Guidance'
    ]
    
    # Simulate effectiveness data
    effectiveness = np.random.uniform(0.6, 0.95, len(interventions))
    usage = np.random.uniform(0.3, 0.8, len(interventions))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=usage,
        y=effectiveness,
        mode='markers+text',
        text=interventions,
        textposition="top center",
        marker=dict(
            size=15,
            color=effectiveness,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Effectiveness")
        ),
        name='Interventions'
    ))
    
    fig.update_layout(
        title="Intervention Effectiveness vs Usage",
        xaxis_title="Usage Rate",
        yaxis_title="Effectiveness Rate",
        height=500
    )
    
    return fig

def create_demographic_analysis(student_data: pd.DataFrame, predictions_df: pd.DataFrame) -> Dict[str, go.Figure]:
    """
    Create demographic analysis charts
    
    Args:
        student_data: DataFrame with student data
        predictions_df: DataFrame with predictions
        
    Returns:
        Dictionary of Plotly figures
    """
    # Merge data
    merged_df = predictions_df.merge(student_data, on='student_id', how='left')
    
    figures = {}
    
    # Risk by family support
    family_risk = merged_df.groupby(['family_support', 'risk_level']).size().unstack(fill_value=0)
    family_risk_pct = family_risk.div(family_risk.sum(axis=1), axis=0) * 100
    
    fig1 = px.bar(
        family_risk_pct,
        title="Risk Distribution by Family Support",
        labels={'value': 'Percentage', 'index': 'Family Support Level'}
    )
    figures['family_support'] = fig1
    
    # Risk by financial stress
    financial_risk = merged_df.groupby(['financial_stress', 'risk_level']).size().unstack(fill_value=0)
    financial_risk_pct = financial_risk.div(financial_risk.sum(axis=1), axis=0) * 100
    
    fig2 = px.bar(
        financial_risk_pct,
        title="Risk Distribution by Financial Stress",
        labels={'value': 'Percentage', 'index': 'Financial Stress Level'}
    )
    figures['financial_stress'] = fig2
    
    # Risk by peer support
    peer_risk = merged_df.groupby(['peer_support', 'risk_level']).size().unstack(fill_value=0)
    peer_risk_pct = peer_risk.div(peer_risk.sum(axis=1), axis=0) * 100
    
    fig3 = px.bar(
        peer_risk_pct,
        title="Risk Distribution by Peer Support",
        labels={'value': 'Percentage', 'index': 'Peer Support Level'}
    )
    figures['peer_support'] = fig3
    
    return figures

def create_subject_attendance_heatmap(student_data: pd.DataFrame) -> go.Figure:
    """
    Create a heatmap showing subject-wise attendance patterns
    
    Args:
        student_data: DataFrame with student data including subject attendance
        
    Returns:
        Plotly figure
    """
    subjects = ['Mathematics', 'Physics', 'Chemistry', 'Biology', 'English', 'History', 'Computer Science', 'Economics']
    
    # Prepare data for heatmap
    attendance_data = []
    for subject in subjects:
        subject_col = f'{subject.lower().replace(" ", "_")}_attendance'
        if subject_col in student_data.columns:
            attendance_data.append(student_data[subject_col].values)
    
    if not attendance_data:
        return None
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=attendance_data,
        x=[f'Student {i+1}' for i in range(len(student_data))],
        y=subjects,
        colorscale='RdYlGn',
        hoverongaps=False,
        text=[[f'{val:.1%}' for val in row] for row in attendance_data],
        texttemplate="%{text}",
        textfont={"size": 8}
    ))
    
    fig.update_layout(
        title="Subject-wise Attendance Heatmap",
        xaxis_title="Students",
        yaxis_title="Subjects",
        height=400
    )
    
    return fig

def create_subject_risk_analysis(student_data: pd.DataFrame) -> go.Figure:
    """
    Create a bar chart showing subject-wise risk distribution
    
    Args:
        student_data: DataFrame with student data including subject risk factors
        
    Returns:
        Plotly figure
    """
    subjects = ['Mathematics', 'Physics', 'Chemistry', 'Biology', 'English', 'History', 'Computer Science', 'Economics']
    
    # Count risk levels per subject
    subject_risk_counts = {}
    for subject in subjects:
        subject_risk_counts[subject] = {'Critical': 0, 'High': 0, 'Medium': 0, 'Low': 0}
    
    # Analyze subject risk factors
    for _, row in student_data.iterrows():
        if 'subject_risk_factors' in row and isinstance(row['subject_risk_factors'], dict):
            for subject, risk_level in row['subject_risk_factors'].items():
                if subject in subject_risk_counts:
                    subject_risk_counts[subject][risk_level] += 1
    
    # Create stacked bar chart
    fig = go.Figure()
    
    risk_colors = {'Critical': '#d32f2f', 'High': '#f57c00', 'Medium': '#ff9800', 'Low': '#4caf50'}
    
    for risk_level in ['Critical', 'High', 'Medium', 'Low']:
        counts = [subject_risk_counts[subject][risk_level] for subject in subjects]
        fig.add_trace(go.Bar(
            name=risk_level,
            x=subjects,
            y=counts,
            marker_color=risk_colors[risk_level]
        ))
    
    fig.update_layout(
        title="Subject-wise Risk Distribution",
        xaxis_title="Subjects",
        yaxis_title="Number of Students",
        barmode='stack',
        height=400
    )
    
    return fig

def get_subject_attendance_summary(student_data: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary statistics for subject-wise attendance
    
    Args:
        student_data: DataFrame with student data
        
    Returns:
        DataFrame with subject attendance summary
    """
    subjects = ['Mathematics', 'Physics', 'Chemistry', 'Biology', 'English', 'History', 'Computer Science', 'Economics']
    
    summary_data = []
    for subject in subjects:
        subject_col = f'{subject.lower().replace(" ", "_")}_attendance'
        if subject_col in student_data.columns:
            attendance = student_data[subject_col]
            summary_data.append({
                'Subject': subject,
                'Average Attendance': f"{attendance.mean():.1%}",
                'Min Attendance': f"{attendance.min():.1%}",
                'Max Attendance': f"{attendance.max():.1%}",
                'Students < 75%': len(attendance[attendance < 0.75]),
                'Students < 60%': len(attendance[attendance < 0.60])
            })
    
    return pd.DataFrame(summary_data)

def create_performance_distribution(student_data: pd.DataFrame) -> go.Figure:
    """
    Create performance distribution charts
    
    Args:
        student_data: DataFrame with student data
        
    Returns:
        Plotly figure
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('GPA Distribution', 'Attendance Distribution', 
                       'Assignment Completion', 'Participation Scores'),
        specs=[[{"type": "histogram"}, {"type": "histogram"}],
               [{"type": "histogram"}, {"type": "histogram"}]]
    )
    
    # GPA distribution
    fig.add_trace(
        go.Histogram(x=student_data['gpa'], name='GPA', nbinsx=20),
        row=1, col=1
    )
    
    # Attendance distribution
    fig.add_trace(
        go.Histogram(x=student_data['attendance_rate'], name='Attendance', nbinsx=20),
        row=1, col=2
    )
    
    # Assignment completion
    fig.add_trace(
        go.Histogram(x=student_data['assignment_completion'], name='Assignments', nbinsx=20),
        row=2, col=1
    )
    
    # Participation scores
    fig.add_trace(
        go.Histogram(x=student_data['participation_score'], name='Participation', nbinsx=20),
        row=2, col=2
    )
    
    fig.update_layout(
        title="Student Performance Distributions",
        height=600,
        showlegend=False
    )
    
    return fig

def calculate_risk_statistics(predictions_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive risk statistics
    
    Args:
        predictions_df: DataFrame with risk predictions
        
    Returns:
        Dictionary with statistics
    """
    total_students = len(predictions_df)
    
    risk_counts = predictions_df['risk_level'].value_counts()
    risk_percentages = (risk_counts / total_students * 100).round(1)
    
    avg_risk_prob = predictions_df['risk_probability'].mean()
    
    # High risk students details
    high_risk_students = predictions_df[predictions_df['risk_level'] == 'High']
    high_risk_avg_prob = high_risk_students['risk_probability'].mean() if len(high_risk_students) > 0 else 0
    
    # Medium risk students details
    medium_risk_students = predictions_df[predictions_df['risk_level'] == 'Medium']
    medium_risk_avg_prob = medium_risk_students['risk_probability'].mean() if len(medium_risk_students) > 0 else 0
    
    return {
        'total_students': total_students,
        'risk_distribution': risk_counts.to_dict(),
        'risk_percentages': risk_percentages.to_dict(),
        'average_risk_probability': avg_risk_prob,
        'high_risk_count': len(high_risk_students),
        'high_risk_avg_probability': high_risk_avg_prob,
        'medium_risk_count': len(medium_risk_students),
        'medium_risk_avg_probability': medium_risk_avg_prob,
        'low_risk_count': len(predictions_df[predictions_df['risk_level'] == 'Low'])
    }

def export_analysis_report(predictions_df: pd.DataFrame, student_data: pd.DataFrame, 
                          statistics: Dict[str, Any]) -> str:
    """
    Generate a comprehensive analysis report
    
    Args:
        predictions_df: DataFrame with predictions
        student_data: DataFrame with student data
        statistics: Dictionary with statistics
        
    Returns:
        HTML report string
    """
    report = f"""
    <html>
    <head>
        <title>Student Risk Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ color: #1f77b4; font-size: 24px; font-weight: bold; }}
            .section {{ margin: 20px 0; }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f0f0f0; border-radius: 5px; }}
            .high-risk {{ color: #f44336; font-weight: bold; }}
            .medium-risk {{ color: #ff9800; font-weight: bold; }}
            .low-risk {{ color: #4caf50; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="header">Student Risk Analysis Report</div>
        <div>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="metric">Total Students: {statistics['total_students']}</div>
            <div class="metric">High Risk: <span class="high-risk">{statistics['high_risk_count']}</span></div>
            <div class="metric">Medium Risk: <span class="medium-risk">{statistics['medium_risk_count']}</span></div>
            <div class="metric">Low Risk: <span class="low-risk">{statistics['low_risk_count']}</span></div>
        </div>
        
        <div class="section">
            <h2>Risk Distribution</h2>
            <p>High Risk: {statistics['risk_percentages'].get('High', 0)}%</p>
            <p>Medium Risk: {statistics['risk_percentages'].get('Medium', 0)}%</p>
            <p>Low Risk: {statistics['risk_percentages'].get('Low', 0)}%</p>
        </div>
        
        <div class="section">
            <h2>High Risk Students Requiring Immediate Attention</h2>
    """
    
    # Add high risk students
    high_risk_students = predictions_df[predictions_df['risk_level'] == 'High'].sort_values('risk_probability', ascending=False)
    for _, student in high_risk_students.head(10).iterrows():
        report += f"<p>• {student['student_id']}: {student['risk_probability']:.1%} risk probability</p>"
    
    report += """
        </div>
        
        <div class="section">
            <h2>Recommendations</h2>
            <ul>
                <li>Implement immediate intervention for high-risk students</li>
                <li>Schedule proactive support meetings for medium-risk students</li>
                <li>Continue monitoring all students with regular check-ins</li>
                <li>Review and update intervention strategies based on outcomes</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    return report

def main():
    """Test utility functions"""
    from train_model import StudentRiskPredictor
    
    # Generate sample data
    predictor = StudentRiskPredictor()
    df = predictor.generate_sample_data(100)
    
    # Test validation
    is_valid, errors = validate_student_data(df)
    print(f"Data validation: {is_valid}")
    if errors:
        print(f"Errors: {errors}")
    
    # Test statistics calculation
    predictions = predictor.predict_risk(df)
    predictions_df = pd.DataFrame({
        'student_id': df['student_id'],
        'risk_level': predictions['predictions'],
        'risk_probability': [predictions['probabilities'][pred][i] for i, pred in enumerate(predictions['predictions'])]
    })
    
    stats = calculate_risk_statistics(predictions_df)
    print(f"Risk statistics: {stats}")

if __name__ == "__main__":
    main()

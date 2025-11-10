# 🎓 Agentic AI Early Warning System

A comprehensive web application for identifying at-risk students and providing proactive intervention strategies for educators using machine learning and AI agents.

---

## 🌟 Features

### 🤖 AI-Powered Risk Assessment
- **Machine Learning Model**: XGBoost classifier for accurate risk prediction  
- **Risk Categories**: Low, Medium, and High risk classification  
- **Feature Analysis**: Comprehensive analysis of academic, behavioral, and support factors  
- **Real-time Predictions**: Instant risk assessment for uploaded student data  

### 🧠 LangGraph Agent Workflow
- **Intelligent Analysis**: AI agent analyzes predictions and generates personalized recommendations  
- **Natural Language Queries**: Ask questions about student risk in plain English  
- **Intervention Strategies**: Automated generation of tailored intervention plans  
- **Proactive Alerts**: Immediate flagging of high-risk students  

### 📊 Interactive Dashboard
- **Risk Distribution**: Visual representation of student risk levels  
- **Performance Trends**: Historical analysis and trend visualization  
- **Individual Student Analysis**: Detailed breakdown of each student's risk factors  
- **Feature Importance**: Understanding which factors most influence risk predictions  

### 🎯 Intervention Management
- **Personalized Recommendations**: AI-generated intervention strategies  
- **Priority Classification**: High, Medium, and Low priority interventions  
- **Resource Allocation**: Suggested resources and support services  
- **Timeline Planning**: Structured intervention timelines  

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher  
- pip package manager  

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd agentic-ai-early-warning
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Access the dashboard**  
   Open your browser and navigate to `http://localhost:8501`

---

## 📁 Project Structure

```bash
agentic-ai-early-warning/
├── app.py                 # Main Streamlit application
├── train_model.py         # ML model training and prediction
├── agent.py               # LangGraph agent workflow
├── utils.py               # Utility functions and visualizations
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## 🔧 Usage Guide

### 1. Dashboard Overview
- **Risk Metrics**: View total students and risk distribution  
- **Visualizations**: Interactive charts showing risk trends  
- **Feature Importance**: Understanding model decision factors  

### 2. Upload Student Data
- **CSV Format**: Upload student data in CSV format  
- **Required Columns**: See data format requirements below  
- **Sample Data**: Download sample CSV for testing  

### 3. Student Analysis
- **Individual Assessment**: Detailed analysis of specific students  
- **Risk Factors**: Identification of key risk indicators  
- **Performance Profile**: Visual representation of student performance  

### 4. AI Agent Queries
- **Natural Language**: Ask questions about student risk  
- **Smart Responses**: AI-powered insights and recommendations  
- **Risk Filtering**: Find students by risk level  

### 5. Model Information
- **Model Details**: Information about the ML model  
- **Performance Metrics**: Model accuracy and validation  
- **Feature Importance**: Understanding model decisions  

---

## 📊 Data Format Requirements

Your CSV file should contain the following columns:

| Column | Type | Description | Range/Values |
|--------|------|-------------|--------------|
| `student_id` | String | Unique student identifier | Any unique string |
| `gpa` | Float | Current GPA | 0.0 - 4.0 |
| `attendance_rate` | Float | Attendance rate | 0.0 - 1.0 |
| `assignment_completion` | Float | Assignment completion rate | 0.0 - 1.0 |
| `participation_score` | Integer | Participation score | 0 - 100 |
| `lms_login_frequency` | Integer | LMS login frequency | 0+ |
| `late_submissions` | Integer | Number of late submissions | 0+ |
| `quiz_average` | Float | Average quiz score | 0 - 100 |
| `study_hours_per_week` | Float | Study hours per week | 0+ |
| `previous_semester_gpa` | Float | Previous semester GPA | 0.0 - 4.0 |
| `extracurricular_activities` | Integer | Number of activities | 0 - 3 |
| `family_support` | String | Family support level | High / Medium / Low |
| `financial_stress` | String | Financial stress level | Low / Medium / High |
| `mental_health_concerns` | Integer | Mental health concerns | 0 / 1 |
| `peer_support` | String | Peer support level | Strong / Moderate / Weak |

---

## 🤖 AI Agent Capabilities

### Natural Language Queries
- "Which students are most at risk this semester?"  
- "Show me all medium-risk students"  
- "What are the main risk factors for high-risk students?"  

### Intervention Recommendations
- **Academic Support**: Tutoring, study groups, academic planning  
- **Social Support**: Peer mentoring, counseling services  
- **Administrative**: Early intervention teams, advisor notifications  
- **Resource Connections**: Financial aid, mental health services  

### Risk Analysis
- **Feature Contributions**: Understanding which factors influence risk  
- **Confidence Scores**: Reliability of risk predictions  
- **Trend Analysis**: Historical risk progression  

---

## 📈 Model Performance

The system uses an XGBoost classifier with the following performance characteristics:

- **Accuracy**: ~85–90% on test data  
- **Cross-validation**: 5-fold CV with consistent performance  
- **Feature Engineering**: Automated feature selection and scaling  
- **Real-time Prediction**: Fast inference for new student data  

---

## 🔍 Key Features Explained

### Risk Classification
- **Low Risk**: Students performing well with minimal intervention needed  
- **Medium Risk**: Students showing concerning patterns requiring proactive support  
- **High Risk**: Students at immediate risk of academic failure requiring urgent intervention  

### Intervention Strategies
- **Immediate Actions**: For high-risk students requiring urgent attention  
- **Proactive Support**: For medium-risk students to prevent escalation  
- **Monitoring**: For low-risk students to maintain performance  

### AI Agent Workflow
1. **Data Analysis**: Comprehensive analysis of student characteristics  
2. **Risk Assessment**: ML model prediction with confidence scores  
3. **Intervention Generation**: Personalized recommendation strategies  
4. **Summary Creation**: Natural language analysis and next steps  

---

## 🛠️ Customization

### Adding New Features
1. Update the feature list in `train_model.py`  
2. Modify the data validation in `utils.py`  
3. Update the agent analysis in `agent.py`  

### Model Retraining
- Use the "Model Info" page to retrain with new data  
- The system automatically saves and loads the latest model  
- Cross-validation ensures model reliability  

### Custom Interventions
- Modify intervention templates in `agent.py`  
- Add new intervention types and resources  
- Customize priority levels and timelines  

---

## 📊 Visualization Features

- **Risk Distribution**: Pie charts and bar charts  
- **Trend Analysis**: Time series visualizations  
- **Correlation Matrix**: Feature relationship analysis  
- **Performance Profiles**: Individual student radar charts  
- **Demographic Analysis**: Risk distribution by student characteristics  

---

## 🔒 Privacy and Security

- **Local Processing**: All data processing happens locally  
- **No External APIs**: No data sent to external services (except optional OpenAI)  
- **Data Validation**: Comprehensive input validation and error handling  
- **Secure Storage**: Model and data stored locally  

---

## 🚨 Troubleshooting

### Common Issues

1. **Model Loading Error**  
   - Solution: The system will automatically train a new model  
   - Check that all dependencies are installed  

2. **Data Format Error**  
   - Solution: Use the sample CSV format provided  
   - Validate column names and data types  

3. **Performance Issues**  
   - Solution: Reduce dataset size for testing  
   - Use the sample data generator for demonstration  

### Getting Help
- Check the console output for error messages  
- Verify all required columns are present in your CSV  
- Ensure data values are within expected ranges  

---

## 📝 License

This project is licensed under the **MIT License** — see the LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📞 Support

For support and questions:
- Check the troubleshooting section above  
- Review the data format requirements  
- Ensure all dependencies are properly installed  

---

**Built with ❤️ for educators and students**

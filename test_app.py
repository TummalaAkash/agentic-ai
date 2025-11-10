"""
Test script to verify the application functionality
"""
import sys
import os

def test_imports():
    """Test that all modules can be imported"""
    try:
        from train_model import StudentRiskPredictor
        print("✓ train_model.py imports successfully")
        
        from agent import StudentRiskAgent
        print("✓ agent.py imports successfully")
        
        import streamlit as st
        print("✓ streamlit imports successfully")
        
        import pandas as pd
        import numpy as np
        import plotly.express as px
        print("✓ All visualization libraries import successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_model_training():
    """Test model training functionality"""
    try:
        from train_model import StudentRiskPredictor
        
        predictor = StudentRiskPredictor()
        df = predictor.generate_sample_data(100)
        print(f"✓ Generated {len(df)} sample records")
        
        results = predictor.train_model(df, model_type='xgboost')
        print(f"✓ Model trained with accuracy: {results['accuracy']:.3f}")
        
        return True
    except Exception as e:
        print(f"✗ Model training error: {e}")
        return False

def test_agent_functionality():
    """Test agent functionality"""
    try:
        from agent import StudentRiskAgent
        
        agent = StudentRiskAgent()
        
        # Test with sample student data
        sample_student = {
            'student_id': 'TEST_001',
            'gpa': 2.5,
            'attendance_rate': 0.7,
            'assignment_completion': 0.6,
            'participation_score': 65,
            'lms_login_frequency': 5,
            'late_submissions': 3,
            'mental_health_concerns': 0,
            'family_support': 'Medium',
            'financial_stress': 'Medium'
        }
        
        result = agent.analyze_student_risk(
            student_data=sample_student,
            risk_prediction='Medium',
            risk_probability=0.65,
            feature_contributions={'gpa': 0.3, 'attendance_rate': 0.25}
        )
        
        print("✓ Agent analysis completed successfully")
        print(f"  - Risk level: {result['risk_analysis']['risk_level']}")
        print(f"  - Interventions: {len(result['interventions'])}")
        
        return True
    except Exception as e:
        print(f"✗ Agent functionality error: {e}")
        return False

def test_streamlit_app():
    """Test Streamlit app structure"""
    try:
        # Test that the app can be imported without errors
        import app
        print("✓ Streamlit app imports successfully")
        
        # Check if main function exists
        if hasattr(app, 'main'):
            print("✓ Main function exists")
        else:
            print("✗ Main function not found")
            return False
            
        return True
    except Exception as e:
        print(f"✗ Streamlit app error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Agentic AI Early Warning System")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Model Training", test_model_training),
        ("Agent Functionality", test_agent_functionality),
        ("Streamlit App", test_streamlit_app)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}:")
        if test_func():
            passed += 1
            print(f"✓ {test_name} PASSED")
        else:
            print(f"✗ {test_name} FAILED")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The application is ready to use.")
        print("\n🚀 To run the application:")
        print("   py -m streamlit run app.py")
        print("\n📖 Then open your browser to: http://localhost:8501")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)





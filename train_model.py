"""
Machine Learning Model Training for Student Risk Prediction
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import joblib
import os
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

class StudentRiskPredictor:
    """Student risk prediction model using ensemble methods"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_importance = None
        self.feature_names = None
        
    def generate_sample_data(self, n_students: int = 1000) -> pd.DataFrame:
        """Generate sample student data for demonstration"""
        np.random.seed(42)
        
        # Define subjects
        subjects = ['Mathematics', 'Physics', 'Chemistry', 'Biology', 'English', 'History', 'Computer Science', 'Economics']
        
        data = {
            'student_id': [f'STU_{i:04d}' for i in range(n_students)],
            'branch': np.random.choice(['Computer Science', 'Electronics', 'Mechanical', 'Civil', 'Biotechnology'], n_students, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
            'gpa': np.random.normal(3.2, 0.8, n_students),
            'attendance_rate': np.random.beta(8, 2, n_students),
            'assignment_completion': np.random.beta(7, 3, n_students),
            'participation_score': np.random.normal(75, 15, n_students),
            'lms_login_frequency': np.random.poisson(15, n_students),
            'late_submissions': np.random.poisson(2, n_students),
            'quiz_average': np.random.normal(78, 12, n_students),
            'study_hours_per_week': np.random.normal(15, 5, n_students),
            'previous_semester_gpa': np.random.normal(3.1, 0.9, n_students),
            'extracurricular_activities': np.random.choice([0, 1, 2, 3], n_students, p=[0.3, 0.4, 0.2, 0.1]),
            'family_support': np.random.choice(['High', 'Medium', 'Low'], n_students, p=[0.4, 0.4, 0.2]),
            'financial_stress': np.random.choice(['Low', 'Medium', 'High'], n_students, p=[0.5, 0.3, 0.2]),
            'mental_health_concerns': np.random.choice([0, 1], n_students, p=[0.8, 0.2]),
            'peer_support': np.random.choice(['Strong', 'Moderate', 'Weak'], n_students, p=[0.3, 0.5, 0.2])
        }
        
        # Add subject-wise attendance
        for subject in subjects:
            # Generate subject-specific attendance with some correlation to overall attendance
            base_attendance = data['attendance_rate']
            subject_variation = np.random.normal(0, 0.15, n_students)  # ±15% variation
            subject_attendance = np.clip(base_attendance + subject_variation, 0, 1)
            data[f'{subject.lower().replace(" ", "_")}_attendance'] = subject_attendance
        
        df = pd.DataFrame(data)
        
        # Create risk labels based on multiple factors including subject-wise attendance
        risk_scores = []
        subject_risk_factors = []
        
        for _, row in df.iterrows():
            score = 0
            subject_risks = {}
            
            # GPA factors
            if row['gpa'] < 2.5:
                score += 3
            elif row['gpa'] < 3.0:
                score += 2
            elif row['gpa'] < 3.5:
                score += 1
                
            # Overall attendance factors
            if row['attendance_rate'] < 0.7:
                score += 3
            elif row['attendance_rate'] < 0.85:
                score += 2
            elif row['attendance_rate'] < 0.95:
                score += 1
            
            # Subject-wise attendance analysis
            for subject in subjects:
                subject_att_col = f'{subject.lower().replace(" ", "_")}_attendance'
                subject_att = row[subject_att_col]
                
                # Calculate subject-specific risk
                subject_score = 0
                if subject_att < 0.6:  # Very low attendance
                    subject_score += 3
                    subject_risks[subject] = 'Critical'
                elif subject_att < 0.75:  # Low attendance
                    subject_score += 2
                    subject_risks[subject] = 'High'
                elif subject_att < 0.85:  # Below average
                    subject_score += 1
                    subject_risks[subject] = 'Medium'
                else:
                    subject_risks[subject] = 'Low'
                
                # Add to overall score (weighted less than overall attendance)
                score += subject_score * 0.3
                
            # Assignment completion
            if row['assignment_completion'] < 0.6:
                score += 2
            elif row['assignment_completion'] < 0.8:
                score += 1
                
            # Other factors
            if row['participation_score'] < 60:
                score += 2
            elif row['participation_score'] < 75:
                score += 1
                
            if row['lms_login_frequency'] < 5:
                score += 2
            elif row['lms_login_frequency'] < 10:
                score += 1
                
            if row['late_submissions'] > 5:
                score += 2
            elif row['late_submissions'] > 2:
                score += 1
                
            if row['mental_health_concerns'] == 1:
                score += 2
                
            if row['family_support'] == 'Low':
                score += 1
                
            if row['financial_stress'] == 'High':
                score += 1
                
            if row['peer_support'] == 'Weak':
                score += 1
                
            # Assign risk category
            if score >= 8:
                risk_scores.append('High')
            elif score >= 4:
                risk_scores.append('Medium')
            else:
                risk_scores.append('Low')
            
            subject_risk_factors.append(subject_risks)
                
        df['risk_category'] = risk_scores
        df['subject_risk_factors'] = subject_risk_factors
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features for model training"""
        # Select numerical features
        numerical_features = [
            'gpa', 'attendance_rate', 'assignment_completion', 'participation_score',
            'lms_login_frequency', 'late_submissions', 'quiz_average', 'study_hours_per_week',
            'previous_semester_gpa', 'extracurricular_activities'
        ]
        
        # Encode categorical features
        categorical_features = ['family_support', 'financial_stress', 'peer_support']
        # Include branch as a categorical feature if present in the data
        if 'branch' in df.columns:
            categorical_features.append('branch')
        
        # Create dummy variables for categorical features
        df_encoded = df.copy()
        for feature in categorical_features:
            dummies = pd.get_dummies(df[feature], prefix=feature)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
        
        # Add mental health as binary feature
        df_encoded['mental_health_concerns'] = df['mental_health_concerns']
        
        # Combine all features - only include numerical and encoded features
        all_features = numerical_features + ['mental_health_concerns']
        
        # Add dummy variables for categorical features
        for feature in categorical_features:
            dummy_cols = [col for col in df_encoded.columns if col.startswith(feature + '_')]
            all_features.extend(dummy_cols)
        
        # Ensure all features exist and are numeric
        available_features = []
        for feature in all_features:
            if feature in df_encoded.columns:
                if df_encoded[feature].dtype in ['object', 'string']:
                    # Skip non-numeric features
                    continue
                available_features.append(feature)
        
        X = df_encoded[available_features].values
        y = self.label_encoder.fit_transform(df['risk_category'])
        
        return X, y, available_features

    def transform_features_for_prediction(self, df: pd.DataFrame) -> np.ndarray:
        """Transform incoming data to the exact training feature space/order.
        Ensures compatibility with stored scaler and model regardless of extra/missing columns.
        """
        if self.feature_names is None:
            # Fallback to dynamic prepare if training features are not available
            X, _, _ = self.prepare_features(df)
            return X

        # Recreate encoded dataframe similar to prepare_features
        df_encoded = df.copy()
        categorical_features = ['family_support', 'financial_stress', 'peer_support']
        if 'branch' in df.columns:
            categorical_features.append('branch')

        for feature in categorical_features:
            if feature in df.columns:
                dummies = pd.get_dummies(df[feature], prefix=feature)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
        if 'mental_health_concerns' in df.columns:
            df_encoded['mental_health_concerns'] = df['mental_health_concerns']

        # Add any missing training columns with zeros
        for col in self.feature_names:
            if col not in df_encoded.columns:
                df_encoded[col] = 0.0

        # Ensure numeric dtype and correct column order
        X = df_encoded[self.feature_names].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
        return X
    
    def train_model(self, df: pd.DataFrame, model_type: str = 'xgboost') -> Dict:
        """Train the risk prediction model"""
        X, y, feature_names = self.prepare_features(df)
        self.feature_names = feature_names
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Get feature importance
        self.feature_importance = dict(zip(feature_names, self.model.feature_importances_))
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        
        results = {
            'accuracy': self.model.score(X_test_scaled, y_test),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred, 
                                                         target_names=self.label_encoder.classes_),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': self.feature_importance,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        return results
    
    def predict_risk(self, student_data: pd.DataFrame) -> Dict:
        """Predict risk for new student data"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Prepare features for prediction
        X = self.transform_features_for_prediction(student_data)
        X_scaled = self.scaler.transform(X)
        
        # Make predictions with compatibility fallback
        try:
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
        except AttributeError:
            # Fallback to Booster to avoid sklearn get_params attribute lookups
            if hasattr(self.model, 'get_booster'):
                booster = self.model.get_booster()
                dmat = xgb.DMatrix(X_scaled)
                proba = booster.predict(dmat)
                # Ensure 2D probabilities
                if proba.ndim == 1:
                    # Binary case: proba is for positive class; create 2-column
                    proba = np.vstack([1.0 - proba, proba]).T
                probabilities = proba
                # Choose class with highest probability index
                class_indices = np.argmax(probabilities, axis=1)
                # Map indices to label_encoder classes length-safe
                classes = list(self.label_encoder.classes_)
                # If mismatch in dims, pad/trim
                if probabilities.shape[1] != len(classes):
                    if probabilities.shape[1] < len(classes):
                        pad = len(classes) - probabilities.shape[1]
                        probabilities = np.hstack([probabilities, np.zeros((probabilities.shape[0], pad))])
                    else:
                        probabilities = probabilities[:, :len(classes)]
                        class_indices = np.argmax(probabilities, axis=1)
                predictions = np.array([classes[i] for i in class_indices])
            else:
                raise
        
        # Get risk categories
        if isinstance(predictions[0], str):
            risk_categories = predictions
        else:
            risk_categories = self.label_encoder.inverse_transform(predictions)
        
        # Get probability for each class
        class_probabilities = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            # Guard against shape mismatches
            if i < probabilities.shape[1]:
                class_probabilities[class_name] = probabilities[:, i]
            else:
                class_probabilities[class_name] = np.zeros(probabilities.shape[0])
        
        return {
            'predictions': risk_categories,
            'probabilities': class_probabilities,
            'feature_importance': self.feature_importance
        }
    
    def save_model(self, filepath: str = 'student_risk_model.joblib'):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_importance': self.feature_importance,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = 'student_risk_model.joblib'):
        """Load a trained model"""
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.feature_importance = model_data['feature_importance']
            self.feature_names = model_data['feature_names']
            # Backward-compat fix: older saved XGB models may reference removed param
            try:
                if isinstance(self.model, xgb.XGBModel):
                    # Provide dummy attributes so sklearn BaseEstimator.get_params doesn't fail on older pickles
                    if not hasattr(self.model, 'use_label_encoder'):
                        setattr(self.model, 'use_label_encoder', False)
                    if not hasattr(self.model, 'gpu_id'):
                        setattr(self.model, 'gpu_id', None)
            except Exception:
                pass
            print(f"Model loaded from {filepath}")
        else:
            print(f"Model file {filepath} not found")

def main():
    """Main function to train and save the model"""
    predictor = StudentRiskPredictor()
    
    # Generate sample data
    print("Generating sample data...")
    df = predictor.generate_sample_data(1000)
    print(f"Generated {len(df)} student records")
    
    # Train model
    print("Training model...")
    results = predictor.train_model(df, model_type='xgboost')
    
    print(f"Model Accuracy: {results['accuracy']:.3f}")
    print(f"Cross-validation Score: {results['cv_mean']:.3f} (+/- {results['cv_std']*2:.3f})")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Save model
    predictor.save_model()
    
    # Show feature importance
    print("\nTop 10 Most Important Features:")
    sorted_features = sorted(results['feature_importance'].items(), 
                           key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_features[:10]:
        print(f"{feature}: {importance:.3f}")

if __name__ == "__main__":
    main()

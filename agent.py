"""
LangGraph Agent Workflow for Student Risk Analysis and Intervention Recommendations
"""
from typing import Dict, List, Any, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import pandas as pd
import numpy as np
from datetime import datetime
import json

class AgentState(TypedDict):
    """State for the LangGraph agent"""
    student_data: Dict[str, Any]
    risk_prediction: str
    risk_probability: float
    feature_contributions: Dict[str, float]
    intervention_recommendations: List[Dict[str, Any]]
    analysis_summary: str
    next_steps: List[str]
    confidence_score: float

class StudentRiskAgent:
    """LangGraph agent for analyzing student risk and generating interventions"""
    
    def __init__(self, openai_api_key: str = None):
        """Initialize the agent with OpenAI API key"""
        if openai_api_key:
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.3,
                openai_api_key=openai_api_key
            )
        else:
            # Use a mock LLM for demonstration
            self.llm = None
        
        self.intervention_templates = {
            'High': {
                'academic': [
                    "Immediate academic support meeting with advisor",
                    "Mandatory tutoring sessions (3x per week)",
                    "Create personalized study plan with learning specialist",
                    "Implement peer mentoring program",
                    "Schedule regular check-ins with faculty"
                ],
                'social': [
                    "Connect with counseling services",
                    "Facilitate study group formation",
                    "Introduce to student support groups",
                    "Arrange peer mentor assignment"
                ],
                'administrative': [
                    "Flag for early intervention team",
                    "Notify academic advisor immediately",
                    "Schedule comprehensive support meeting",
                    "Create intervention timeline"
                ]
            },
            'Medium': {
                'academic': [
                    "Schedule academic planning session",
                    "Recommend tutoring services",
                    "Create study schedule with advisor",
                    "Implement progress monitoring"
                ],
                'social': [
                    "Connect with study groups",
                    "Introduce to campus resources",
                    "Schedule peer support meeting"
                ],
                'administrative': [
                    "Monitor progress bi-weekly",
                    "Schedule advisor check-in",
                    "Create support plan"
                ]
            },
            'Low': {
                'academic': [
                    "Maintain current academic plan",
                    "Optional tutoring if requested",
                    "Regular advisor meetings"
                ],
                'social': [
                    "Encourage participation in activities",
                    "Maintain peer connections"
                ],
                'administrative': [
                    "Standard monitoring",
                    "Regular advisor check-ins"
                ]
            }
        }
    
    def analyze_student_data(self, state: AgentState) -> AgentState:
        """Analyze student data and identify key risk factors"""
        student = state["student_data"]
        
        # Identify key risk factors
        risk_factors = []
        if student.get('gpa', 0) < 2.5:
            risk_factors.append("Very low GPA")
        elif student.get('gpa', 0) < 3.0:
            risk_factors.append("Below average GPA")
            
        if student.get('attendance_rate', 1) < 0.7:
            risk_factors.append("Poor attendance")
        elif student.get('attendance_rate', 1) < 0.85:
            risk_factors.append("Below average attendance")
            
        if student.get('assignment_completion', 1) < 0.6:
            risk_factors.append("Low assignment completion")
            
        if student.get('participation_score', 100) < 60:
            risk_factors.append("Low class participation")
            
        if student.get('lms_login_frequency', 0) < 5:
            risk_factors.append("Minimal LMS engagement")
            
        if student.get('late_submissions', 0) > 5:
            risk_factors.append("Frequent late submissions")
            
        if student.get('mental_health_concerns', 0) == 1:
            risk_factors.append("Mental health concerns")
            
        if student.get('family_support') == 'Low':
            risk_factors.append("Limited family support")
            
        if student.get('financial_stress') == 'High':
            risk_factors.append("High financial stress")
            
        if student.get('peer_support') == 'Weak':
            risk_factors.append("Weak peer support network")
        
        state["risk_factors"] = risk_factors
        return state
    
    def generate_interventions(self, state: AgentState) -> AgentState:
        """Generate personalized intervention recommendations"""
        risk_level = state["risk_prediction"]
        risk_factors = state.get("risk_factors", [])
        student = state["student_data"]
        
        # Get base interventions for risk level
        interventions = self.intervention_templates.get(risk_level, {})
        
        # Customize based on specific risk factors
        personalized_interventions = []
        
        # Academic interventions
        academic_interventions = interventions.get('academic', [])
        if "Very low GPA" in risk_factors or "Below average GPA" in risk_factors:
            personalized_interventions.extend([
                {
                    "type": "Academic Support",
                    "title": "Intensive Academic Recovery Plan",
                    "description": "Develop comprehensive academic recovery strategy with learning specialist",
                    "priority": "High",
                    "timeline": "Immediate",
                    "resources": ["Learning Specialist", "Academic Advisor", "Tutoring Center"]
                }
            ])
        
        if "Poor attendance" in risk_factors or "Below average attendance" in risk_factors:
            personalized_interventions.append({
                "type": "Attendance Support",
                "title": "Attendance Improvement Plan",
                "description": "Address barriers to attendance and create accountability system",
                "priority": "High",
                "timeline": "1 week",
                "resources": ["Academic Advisor", "Student Success Center"]
            })
        
        if "Low assignment completion" in risk_factors:
            personalized_interventions.append({
                "type": "Academic Support",
                "title": "Assignment Completion Strategy",
                "description": "Develop time management and organizational skills for assignment completion",
                "priority": "Medium",
                "timeline": "2 weeks",
                "resources": ["Learning Specialist", "Academic Coach"]
            })
        
        # Social/Emotional interventions
        if "Mental health concerns" in risk_factors:
            personalized_interventions.append({
                "type": "Mental Health Support",
                "title": "Mental Health Support Connection",
                "description": "Connect with counseling services and mental health resources",
                "priority": "High",
                "timeline": "Immediate",
                "resources": ["Counseling Center", "Mental Health Services", "Crisis Support"]
            })
        
        if "Limited family support" in risk_factors or "Weak peer support network" in risk_factors:
            personalized_interventions.append({
                "type": "Social Support",
                "title": "Social Support Network Development",
                "description": "Build connections with peers, mentors, and support systems",
                "priority": "Medium",
                "timeline": "2-3 weeks",
                "resources": ["Peer Mentors", "Student Organizations", "Support Groups"]
            })
        
        # Financial interventions
        if "High financial stress" in risk_factors:
            personalized_interventions.append({
                "type": "Financial Support",
                "title": "Financial Aid and Support Services",
                "description": "Connect with financial aid office and support services",
                "priority": "High",
                "timeline": "1 week",
                "resources": ["Financial Aid Office", "Student Emergency Fund", "Food Pantry"]
            })
        
        # Add base interventions if no specific ones were added
        if not personalized_interventions:
            for category, base_interventions in interventions.items():
                for intervention in base_interventions:
                    personalized_interventions.append({
                        "type": category.title(),
                        "title": intervention,
                        "description": f"Standard {risk_level.lower()} risk intervention",
                        "priority": "Medium",
                        "timeline": "2-4 weeks",
                        "resources": ["Academic Advisor"]
                    })
        
        state["intervention_recommendations"] = personalized_interventions
        return state
    
    def generate_analysis_summary(self, state: AgentState) -> AgentState:
        """Generate a comprehensive analysis summary"""
        student = state["student_data"]
        risk_level = state["risk_prediction"]
        risk_probability = state["risk_probability"]
        risk_factors = state.get("risk_factors", [])
        
        # Calculate confidence score based on feature contributions
        confidence_score = min(0.95, max(0.6, risk_probability))
        state["confidence_score"] = confidence_score
        
        # Generate summary
        summary_parts = [
            f"Student {student.get('student_id', 'Unknown')} has been classified as {risk_level} risk "
            f"with {risk_probability:.1%} probability."
        ]
        
        if risk_factors:
            summary_parts.append(f"Key risk factors identified: {', '.join(risk_factors[:5])}")
        
        if risk_level == "High":
            summary_parts.append("Immediate intervention is recommended to prevent academic failure.")
        elif risk_level == "Medium":
            summary_parts.append("Proactive support is recommended to prevent risk escalation.")
        else:
            summary_parts.append("Student appears to be on track with minimal intervention needed.")
        
        # Add specific recommendations
        interventions = state.get("intervention_recommendations", [])
        if interventions:
            high_priority = [i for i in interventions if i.get("priority") == "High"]
            if high_priority:
                summary_parts.append(f"High priority interventions: {', '.join([i['title'] for i in high_priority[:3]])}")
        
        state["analysis_summary"] = " ".join(summary_parts)
        
        # Generate next steps
        next_steps = []
        if risk_level == "High":
            next_steps.extend([
                "Schedule immediate meeting with student",
                "Notify academic advisor and support team",
                "Implement high-priority interventions within 48 hours"
            ])
        elif risk_level == "Medium":
            next_steps.extend([
                "Schedule support meeting within 1 week",
                "Begin implementing recommended interventions",
                "Monitor progress bi-weekly"
            ])
        else:
            next_steps.extend([
                "Continue regular monitoring",
                "Maintain current support level",
                "Check in monthly"
            ])
        
        state["next_steps"] = next_steps
        return state
    
    def create_agent_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_data", self.analyze_student_data)
        workflow.add_node("generate_interventions", self.generate_interventions)
        workflow.add_node("generate_summary", self.generate_analysis_summary)
        
        # Add edges
        workflow.add_edge("analyze_data", "generate_interventions")
        workflow.add_edge("generate_interventions", "generate_summary")
        workflow.add_edge("generate_summary", END)
        
        # Set entry point
        workflow.set_entry_point("analyze_data")
        
        return workflow.compile()
    
    def analyze_student_risk(self, student_data: Dict[str, Any], 
                           risk_prediction: str, 
                           risk_probability: float,
                           feature_contributions: Dict[str, float]) -> Dict[str, Any]:
        """Main method to analyze student risk and generate recommendations"""
        
        # Initialize state
        initial_state = AgentState(
            student_data=student_data,
            risk_prediction=risk_prediction,
            risk_probability=risk_probability,
            feature_contributions=feature_contributions,
            intervention_recommendations=[],
            analysis_summary="",
            next_steps=[],
            confidence_score=0.0
        )
        
        # Create and run workflow
        workflow = self.create_agent_workflow()
        final_state = workflow.invoke(initial_state)
        
        return {
            "risk_analysis": {
                "risk_level": final_state["risk_prediction"],
                "risk_probability": final_state["risk_probability"],
                "confidence_score": final_state["confidence_score"],
                "risk_factors": final_state.get("risk_factors", []),
                "feature_contributions": final_state["feature_contributions"]
            },
            "interventions": final_state["intervention_recommendations"],
            "summary": final_state["analysis_summary"],
            "next_steps": final_state["next_steps"]
        }
    
    def query_students(self, query: str, student_data: List[Dict[str, Any]], 
                      predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle natural language queries about students"""
        
        # Simple query processing (in a real implementation, this would use LLM)
        query_lower = query.lower()
        
        if "most at risk" in query_lower or "highest risk" in query_lower:
            # Find students with highest risk
            high_risk_students = []
            for i, pred in enumerate(predictions):
                if pred.get('risk_level') == 'High':
                    high_risk_students.append({
                        'student_id': student_data[i].get('student_id'),
                        'risk_level': pred.get('risk_level'),
                        'risk_probability': pred.get('risk_probability', 0)
                    })
            
            return {
                "query": query,
                "response": f"Found {len(high_risk_students)} high-risk students",
                "students": sorted(high_risk_students, key=lambda x: x['risk_probability'], reverse=True)
            }
        
        elif "medium risk" in query_lower:
            medium_risk_students = []
            for i, pred in enumerate(predictions):
                if pred.get('risk_level') == 'Medium':
                    medium_risk_students.append({
                        'student_id': student_data[i].get('student_id'),
                        'risk_level': pred.get('risk_level'),
                        'risk_probability': pred.get('risk_probability', 0)
                    })
            
            return {
                "query": query,
                "response": f"Found {len(medium_risk_students)} medium-risk students",
                "students": sorted(medium_risk_students, key=lambda x: x['risk_probability'], reverse=True)
            }
        
        elif "low risk" in query_lower:
            low_risk_students = []
            for i, pred in enumerate(predictions):
                if pred.get('risk_level') == 'Low':
                    low_risk_students.append({
                        'student_id': student_data[i].get('student_id'),
                        'risk_level': pred.get('risk_level'),
                        'risk_probability': pred.get('risk_probability', 0)
                    })
            
            return {
                "query": query,
                "response": f"Found {len(low_risk_students)} low-risk students",
                "students": sorted(low_risk_students, key=lambda x: x['risk_probability'], reverse=True)
            }
        
        else:
            return {
                "query": query,
                "response": "I can help you find students by risk level. Try asking about 'most at risk', 'medium risk', or 'low risk' students.",
                "students": []
            }

def main():
    """Test the agent functionality"""
    agent = StudentRiskAgent()
    
    # Sample student data
    sample_student = {
        'student_id': 'STU_0001',
        'gpa': 2.1,
        'attendance_rate': 0.65,
        'assignment_completion': 0.45,
        'participation_score': 55,
        'lms_login_frequency': 3,
        'late_submissions': 8,
        'mental_health_concerns': 1,
        'family_support': 'Low',
        'financial_stress': 'High'
    }
    
    # Test analysis
    result = agent.analyze_student_risk(
        student_data=sample_student,
        risk_prediction='High',
        risk_probability=0.85,
        feature_contributions={'gpa': 0.3, 'attendance_rate': 0.25, 'assignment_completion': 0.2}
    )
    
    print("Agent Analysis Result:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()

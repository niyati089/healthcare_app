"""
Healthcare AI - Symptom Analysis & Medicine Recommendation System
Author: Senior Healthcare AI Engineer
Purpose: Clinical Decision Support System (NOT autonomous prescriber)
"""

import streamlit as st
import pandas as pd
from logic import (
    predict_disease,
    get_medicine_recommendations,
    extract_symptoms_from_text,
    AVAILABLE_SYMPTOMS,
    DISEASE_INFO
)

# Page configuration
st.set_page_config(
    page_title="Healthcare AI Assistant",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E40AF;
        text-align: center;
        margin-bottom: 1rem;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .warning-box {
        background-color: #FEE2E2;
        border-left: 5px solid #DC2626;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
        font-weight: bold;
    }
    .result-box {
        background-color: #DBEAFE;
        border-left: 5px solid #2563EB;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Healthcare AI Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #6B7280;">AI-Powered Symptom Analysis & Medicine Recommendation</p>', unsafe_allow_html=True)
    
    # Critical Disclaimer
    st.markdown("""
        <div class="disclaimer-box">
            <h3>⚠️ MEDICAL DISCLAIMER</h3>
            <p><strong>This system provides decision support only and does NOT replace a qualified healthcare professional.</strong></p>
            <ul>
                <li>Always consult a licensed doctor before taking any medication</li>
                <li>This tool is for educational and assistive purposes only</li>
                <li>In case of emergency, call your local emergency number immediately</li>
                <li>Do not use this system as a substitute for professional medical advice</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Two-column layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Patient Information")
        
        # Patient details
        patient_name = st.text_input("Patient Name", placeholder="Enter full name")
        patient_age = st.number_input("Patient Age", min_value=1, max_value=120, value=30, step=1)
        
        # Age category display
        if patient_age < 12:
            age_category = "Child"
            st.info("👶 Patient Category: Child (Age-appropriate medications will be recommended)")
        elif patient_age < 18:
            age_category = "Adolescent"
            st.info("🧒 Patient Category: Adolescent")
        elif patient_age < 65:
            age_category = "Adult"
            st.info("👤 Patient Category: Adult")
        else:
            age_category = "Senior"
            st.info("👴 Patient Category: Senior (Special care medications)")
        
        st.markdown("---")
        
        # Allergies
        st.subheader("🚫 Known Drug Allergies")
        allergies_input = st.text_input(
            "Enter allergies (comma-separated)",
            placeholder="e.g., Penicillin, Aspirin, Ibuprofen",
            help="List any known drug allergies to filter recommendations"
        )
        allergies = [a.strip().lower() for a in allergies_input.split(",") if a.strip()]
        
        if allergies:
            st.warning(f"⚠️ Patient has {len(allergies)} known allergy(ies): {', '.join(allergies)}")
    
    with col2:
        st.subheader("🩺 Symptom Selection")
        
        # Symptom input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Select from list", "Describe in text (NLP)"],
            horizontal=True
        )
        
        selected_symptoms = []
        
        if input_method == "Select from list":
            st.write("Select all symptoms that apply:")
            
            # Organize symptoms in columns
            symptom_cols = st.columns(3)
            for idx, symptom in enumerate(AVAILABLE_SYMPTOMS):
                col_idx = idx % 3
                with symptom_cols[col_idx]:
                    if st.checkbox(symptom, key=f"symptom_{symptom}"):
                        selected_symptoms.append(symptom)
        
        else:  # NLP-based input
            st.write("Describe your symptoms in natural language:")
            symptom_text = st.text_area(
                "Symptom Description",
                placeholder="Example: I have a high fever, headache, and feel very tired. My body aches and I'm coughing.",
                height=150
            )
            
            if symptom_text:
                with st.spinner("Analyzing symptoms..."):
                    selected_symptoms = extract_symptoms_from_text(symptom_text)
                    if selected_symptoms:
                        st.success(f"✅ Extracted symptoms: {', '.join(selected_symptoms)}")
                    else:
                        st.warning("⚠️ No recognized symptoms found. Try selecting from the list.")
        
        # Display selected symptoms
        if selected_symptoms:
            st.markdown("**Selected Symptoms:**")
            for symptom in selected_symptoms:
                st.markdown(f"- {symptom}")
    
    st.markdown("---")
    
    # Analysis button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        analyze_button = st.button("🔍 Analyze Symptoms", use_container_width=True, type="primary")
    
    # Results section
    if analyze_button:
        if not patient_name:
            st.error("❌ Please enter patient name")
            return
        
        if not selected_symptoms:
            st.error("❌ Please select at least one symptom")
            return
        
        with st.spinner("Analyzing symptoms and generating recommendations..."):
            # Predict disease
            predicted_disease, confidence = predict_disease(selected_symptoms)
            
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            # Display results in columns
            result_col1, result_col2 = st.columns([1, 1])
            
            with result_col1:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown(f"**Patient:** {patient_name}")
                st.markdown(f"**Age:** {patient_age} years ({age_category})")
                st.markdown(f"**Symptoms Count:** {len(selected_symptoms)}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with result_col2:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown(f"**Predicted Condition:** {predicted_disease}")
                st.markdown(f"**Confidence:** {confidence}%")
                
                # Display disease severity
                disease_data = DISEASE_INFO.get(predicted_disease, {})
                severity = disease_data.get("severity", "Unknown")
                
                if severity == "High":
                    st.markdown("**Severity:** 🔴 High")
                elif severity == "Medium":
                    st.markdown("**Severity:** 🟡 Medium")
                else:
                    st.markdown("**Severity:** 🟢 Low")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Disease information
            st.markdown("### 📖 Condition Information")
            description = disease_data.get("description", "No description available.")
            st.info(description)
            
            # Critical check for cardiac risk
            if predicted_disease == "Cardiac Risk":
                st.markdown("""
                    <div class="warning-box">
                        <h3>🚨 URGENT: IMMEDIATE MEDICAL ATTENTION REQUIRED</h3>
                        <p style="font-size: 1.1rem;">Cardiac symptoms detected. This is a medical emergency.</p>
                        <ul>
                            <li><strong>DO NOT</strong> attempt self-medication</li>
                            <li><strong>SEEK IMMEDIATE</strong> emergency medical care</li>
                            <li><strong>CALL</strong> emergency services (108/102 in India, 911 in USA)</li>
                            <li><strong>GO TO</strong> the nearest hospital emergency department</li>
                        </ul>
                        <p><strong>No medicine recommendations will be provided for cardiac conditions.</strong></p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Emergency contacts
                st.error("⚠️ If you're experiencing chest pain, difficulty breathing, or other severe symptoms, seek help immediately!")
                
            else:
                # Get medicine recommendations
                recommendations = get_medicine_recommendations(
                    predicted_disease,
                    patient_age,
                    allergies
                )
                
                if recommendations:
                    st.markdown("### 💊 Recommended Medications (For Reference Only)")
                    
                    st.warning("⚠️ These are suggestions only. Consult a doctor before taking any medication.")
                    
                    # Display recommendations in a table
                    rec_df = pd.DataFrame(recommendations)
                    st.dataframe(rec_df, use_container_width=True, hide_index=True)
                    
                    # Additional precautions
                    st.markdown("### ⚕️ Precautions & Next Steps")
                    precautions = disease_data.get("precautions", [])
                    if precautions:
                        for precaution in precautions:
                            st.markdown(f"- {precaution}")
                    
                    st.info("📌 **Remember:** Always verify medication with a healthcare professional before use.")
                    
                else:
                    st.warning("⚠️ No safe medication recommendations available. Please consult a doctor.")
            
            # Generate summary report
            st.markdown("---")
            st.markdown("### 📄 Summary Report")
            
            report = f"""
**SYMPTOM ANALYSIS REPORT**
**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

**Patient Information:**
- Name: {patient_name}
- Age: {patient_age} years ({age_category})
- Known Allergies: {', '.join(allergies) if allergies else 'None reported'}

**Symptoms Reported:**
{chr(10).join([f'- {s}' for s in selected_symptoms])}

**Analysis:**
- Predicted Condition: {predicted_disease}
- Confidence Level: {confidence}%
- Severity: {severity}

**Recommendations:**
"""
            if predicted_disease == "Cardiac Risk":
                report += "\n⚠️ IMMEDIATE MEDICAL ATTENTION REQUIRED - NO SELF-MEDICATION\n"
            elif recommendations:
                report += "\n" + "\n".join([f"- {r['Medicine']} ({r['Type']})" for r in recommendations])
                report += "\n\n⚠️ Consult doctor before taking any medication"
            
            report += """

**Disclaimer:**
This report is generated by an AI system for decision support purposes only.
It does NOT constitute medical advice, diagnosis, or treatment.
Always consult a qualified healthcare professional before taking any action.
"""
            
            st.text_area("Report (Copy for your records)", report, height=300)
            
            # Download button
            st.download_button(
                label="📥 Download Report",
                data=report,
                file_name=f"symptom_analysis_{patient_name.replace(' ', '_')}_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #6B7280; padding: 2rem;">
            <p><strong>Healthcare AI Assistant v1.0</strong></p>
            <p>Built for clinical decision support | Not a replacement for medical professionals</p>
            <p>In case of emergency, contact your local emergency services immediately</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
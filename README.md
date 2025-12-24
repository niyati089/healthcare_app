# 🏥 Healthcare AI - Symptom Analysis & Medicine Recommendation System

## 📋 Overview

A comprehensive **Clinical Decision Support System** built with Python and Streamlit that analyzes patient symptoms and provides medicine recommendations while maintaining strict safety and ethical guidelines.

**⚠️ IMPORTANT:** This system provides decision support only and does NOT replace professional medical advice.

---

## 🎯 Features

### Core Functionality
1. **Patient Information Management**
   - Patient name and age capture
   - Age-based categorization (Child/Adolescent/Adult/Senior)
   - Known drug allergy tracking

2. **Symptom Input Methods**
   - ✅ Checkbox-based symptom selection (20 symptoms)
   - ✅ Natural Language Processing (NLP) text input
   - Real-time symptom extraction from free text

3. **AI-Powered Disease Prediction**
   - Machine Learning: Naive Bayes Classifier
   - Trained on symptom-disease patterns
   - Confidence score calculation
   - Supports 4 conditions:
     - Flu
     - Viral Infection
     - Acidity
     - Cardiac Risk (Critical)

4. **Safe Medicine Recommendations**
   - Age-appropriate filtering
   - Allergy cross-checking
   - No dosage information (safety measure)
   - Medicine type and notes included

5. **Critical Safety Features**
   - ⚠️ **Cardiac Risk Detection**: Immediate referral, NO medicine recommendations
   - Prominent medical disclaimers
   - Emergency contact information
   - Downloadable consultation reports

---

## 🏗️ Project Structure

```
healthcare_app/
│
├── app.py                 # Streamlit web interface
├── logic.py               # ML model, medicine logic, NLP
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step-by-Step Installation

1. **Clone or download the project**
```bash
mkdir healthcare_app
cd healthcare_app
```

2. **Create virtual environment (recommended)**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Access the application**
   - Local URL: `http://localhost:8501`
   - Network URL: Will be displayed in terminal

---

## 💻 Usage Guide

### For Internship Demo

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Demo Scenario 1: Flu Case**
   - Enter patient name: "John Doe"
   - Age: 28
   - Allergies: Leave blank
   - Select symptoms: fever, cough, fatigue, headache, body_ache
   - Click "Analyze Symptoms"
   - ✅ Expected: Flu diagnosis with medicine recommendations

3. **Demo Scenario 2: Cardiac Emergency**
   - Enter patient name: "Emergency Patient"
   - Age: 55
   - Select symptoms: chest_pain, shortness_of_breath, dizziness, sweating
   - Click "Analyze Symptoms"
   - ⚠️ Expected: Emergency warning, NO medicine recommendations

4. **Demo Scenario 3: NLP Text Input**
   - Choose "Describe in text (NLP)"
   - Enter: "I have a bad headache, fever, and my body hurts all over"
   - System extracts: headache, fever, body_ache
   - Analyzes and provides recommendations

5. **Demo Scenario 4: Allergy Filtering**
   - Patient age: 10 (Child)
   - Allergies: "paracetamol"
   - Symptoms: fever, cough
   - ✅ Expected: Medicines filtered, no paracetamol shown

---

## 🔧 Technical Components

### 1. Machine Learning (logic.py)

**Algorithm:** Multinomial Naive Bayes
- **Training Data:** Symptom-disease mappings
- **Features:** 20 binary symptom indicators
- **Output:** Disease prediction + confidence score

```python
# Example prediction flow
symptoms = ["fever", "cough", "fatigue"]
disease, confidence = predict_disease(symptoms)
# Returns: ("Flu", 87)
```

### 2. Medicine Recommendation Engine

**Safety Filters:**
- Age restrictions (e.g., Omeprazole: 18+ only)
- Allergy contraindications
- Disease-specific contraindications

```python
recommendations = get_medicine_recommendations(
    disease="Flu",
    patient_age=25,
    allergies=["aspirin"]
)
```

### 3. NLP Symptom Extraction

**Method:** Keyword-based pattern matching
- 20 symptom categories
- 3-5 keywords per symptom
- Word boundary detection (avoids partial matches)

```python
text = "I have a high fever and headache"
symptoms = extract_symptoms_from_text(text)
# Returns: ["fever", "headache"]
```

---

## 🛡️ Safety & Ethics Implementation

### Multi-Layer Safety System

1. **UI Level**
   - Prominent disclaimer on every page
   - Color-coded warning boxes
   - Emergency contact information

2. **Logic Level**
   - Cardiac risk override (3+ cardiac symptoms = automatic emergency referral)
   - No medicine recommendations for high-risk conditions
   - Age-based medicine filtering
   - Allergy cross-checking

3. **Data Level**
   - No dosage information stored
   - Generic medicine names only
   - Clear contraindication lists

### Ethical Considerations

- ✅ Transparent limitations
- ✅ No autonomous prescribing
- ✅ Encourages professional consultation
- ✅ Emergency escalation protocols
- ✅ Age-appropriate content

---

## 📊 Supported Conditions

| Condition | Severity | Symptoms Count | Medicine Recommendations |
|-----------|----------|----------------|--------------------------|
| Flu | Medium | 8 | ✅ Yes (4 options) |
| Viral Infection | Low | 7 | ✅ Yes (4 options) |
| Acidity | Low | 5 | ✅ Yes (4 options) |
| Cardiac Risk | **HIGH** | 6 | ❌ **NO - Emergency Only** |

---

## 🧪 Testing Checklist

### Functional Tests
- [ ] Patient information input
- [ ] Symptom selection (checkbox)
- [ ] Symptom extraction (NLP)
- [ ] Disease prediction accuracy
- [ ] Medicine recommendations
- [ ] Age filtering
- [ ] Allergy filtering
- [ ] Cardiac risk detection
- [ ] Report generation
- [ ] Report download

### Safety Tests
- [ ] Cardiac symptoms → No medicines
- [ ] Child age → Age-appropriate medicines only
- [ ] Known allergies → Contraindicated medicines excluded
- [ ] Disclaimer visibility
- [ ] Emergency instructions clarity

---

## 📈 Future Enhancements

### Potential Additions
1. **More Conditions**
   - Diabetes symptoms
   - Hypertension indicators
   - Allergic reactions

2. **Advanced ML**
   - Deep learning models
   - Symptom severity weighting
   - Multi-disease detection

3. **Integration Options**
   - EHR/EMR systems
   - Telemedicine platforms
   - Pharmacy databases

4. **Enhanced NLP**
   - Transformer-based models
   - Multilingual support
   - Severity extraction

5. **User Features**
   - Patient history tracking
   - Symptom timeline
   - Medicine interaction checker
   - Doctor appointment booking

---

## ⚠️ Limitations & Disclaimers

### System Limitations
- Limited to 4 conditions
- Binary symptom representation (present/absent)
- No symptom severity assessment
- No medication dosage information
- Single disease prediction (no comorbidities)

### Medical Disclaimers
1. **Not a Medical Device**: This system is educational and assistive only
2. **Not for Diagnosis**: Cannot replace clinical examination
3. **Not for Prescription**: All medicines require doctor approval
4. **Emergency Cases**: Always call emergency services for serious symptoms

---

## 🎓 Internship Presentation Tips

### Key Talking Points
1. **Problem Statement**
   - Rising need for preliminary health assessment tools
   - Doctor accessibility challenges
   - Self-medication risks

2. **Solution Approach**
   - AI-assisted decision support
   - Safety-first design
   - User-friendly interface

3. **Technical Highlights**
   - Machine Learning (Naive Bayes)
   - NLP for text processing
   - Multi-layer safety filters
   - Responsive web design

4. **Safety Emphasis**
   - Cardiac risk detection
   - Age-based recommendations
   - Allergy checking
   - Clear disclaimers

### Demo Flow
1. Show normal case (Flu) → Full recommendations
2. Show critical case (Cardiac) → Emergency protocol
3. Show safety features (Allergy filtering)
4. Show NLP capability (Text input)
5. Show report generation

---

## 📞 Support & Contact

### For Issues
- Check console for error messages
- Verify all dependencies installed
- Ensure Python 3.8+ is being used

### For Questions
- Review code comments in `app.py` and `logic.py`
- Check this README thoroughly
- Test with provided demo scenarios

---

## 📜 License & Credits

**Educational Project**
- Built for internship demonstration
- Not for commercial use
- Not cleared for clinical deployment

**Technologies Used**
- Python 3.8+
- Streamlit (Web Framework)
- scikit-learn (Machine Learning)
- Pandas (Data Handling)
- NumPy (Numerical Computing)

---

## ✅ Checklist for Internship Submission

- [ ] Code is well-commented
- [ ] All files included (app.py, logic.py, requirements.txt, README.md)
- [ ] Application runs without errors
- [ ] All 4 demo scenarios tested
- [ ] Safety features verified
- [ ] README is comprehensive
- [ ] Presentation slides prepared (optional)
- [ ] Video demo recorded (optional)

---

**Version:** 1.0  
**Last Updated:** December 2024  
**Status:** Internship-Ready ✅

---

## 🏁 Quick Start Command

```bash
# One-line setup and run
pip install -r requirements.txt && streamlit run app.py
```

**Access the app at:** `http://localhost:8501`

---

**Remember: This is a decision support tool, not a replacement for medical professionals!** 🏥
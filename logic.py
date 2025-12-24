"""
Healthcare Logic Module
Contains disease prediction, medicine recommendation, and NLP extraction logic
"""

import numpy as np
from sklearn.naive_bayes import MultinomialNB
import re

# ========================
# SYMPTOM DATABASE
# ========================

AVAILABLE_SYMPTOMS = [
    "fever",
    "cough",
    "fatigue",
    "headache",
    "body_ache",
    "sore_throat",
    "runny_nose",
    "nausea",
    "vomiting",
    "diarrhea",
    "stomach_pain",
    "heartburn",
    "chest_pain",
    "shortness_of_breath",
    "dizziness",
    "rapid_heartbeat",
    "sweating",
    "loss_of_appetite",
    "weakness",
    "chills"
]

# ========================
# DISEASE DATABASE
# ========================

DISEASE_SYMPTOMS = {
    "Flu": [
        "fever", "cough", "fatigue", "headache", "body_ache",
        "sore_throat", "chills", "weakness"
    ],
    "Viral Infection": [
        "fever", "cough", "fatigue", "sore_throat", "runny_nose",
        "headache", "body_ache"
    ],
    "Acidity": [
        "heartburn", "stomach_pain", "nausea", "vomiting",
        "loss_of_appetite"
    ],
    "Cardiac Risk": [
        "chest_pain", "shortness_of_breath", "dizziness",
        "rapid_heartbeat", "sweating", "nausea"
    ]
}

DISEASE_INFO = {
    "Flu": {
        "description": "Influenza (flu) is a viral infection that attacks the respiratory system. Common during seasonal changes.",
        "severity": "Medium",
        "precautions": [
            "Get adequate rest (7-9 hours sleep)",
            "Stay hydrated (drink plenty of water)",
            "Avoid contact with others to prevent spread",
            "Monitor temperature regularly",
            "Consult doctor if symptoms worsen after 3-4 days"
        ]
    },
    "Viral Infection": {
        "description": "Common viral infections affecting the upper respiratory tract. Usually self-limiting.",
        "severity": "Low",
        "precautions": [
            "Rest and allow your body to recover",
            "Drink warm fluids",
            "Avoid cold foods and beverages",
            "Practice good hygiene",
            "Isolate to prevent transmission"
        ]
    },
    "Acidity": {
        "description": "Excess acid production in the stomach causing discomfort and heartburn.",
        "severity": "Low",
        "precautions": [
            "Avoid spicy and oily foods",
            "Eat smaller, frequent meals",
            "Avoid lying down immediately after eating",
            "Reduce caffeine and alcohol intake",
            "Maintain healthy weight"
        ]
    },
    "Cardiac Risk": {
        "description": "⚠️ CRITICAL: Symptoms suggesting potential cardiac/heart-related issues requiring immediate medical evaluation.",
        "severity": "High",
        "precautions": [
            "SEEK IMMEDIATE EMERGENCY MEDICAL CARE",
            "Do NOT attempt self-medication",
            "Call emergency services (108/102 in India, 911 in USA)",
            "If conscious, sit upright and stay calm",
            "Have someone stay with the patient"
        ]
    }
}

# ========================
# MEDICINE DATABASE
# ========================

MEDICINE_DATABASE = {
    "Flu": [
        {
            "name": "Paracetamol",
            "type": "Fever reducer & Pain reliever",
            "age_restrictions": None,  # Safe for all ages (dose-adjusted)
            "contraindications": ["paracetamol", "acetaminophen"],
            "notes": "Standard fever and pain management"
        },
        {
            "name": "Cetirizine",
            "type": "Antihistamine",
            "age_restrictions": 2,  # Minimum age
            "contraindications": ["cetirizine"],
            "notes": "For runny nose and allergic symptoms"
        },
        {
            "name": "Vitamin C",
            "type": "Supplement",
            "age_restrictions": None,
            "contraindications": [],
            "notes": "Immune support"
        },
        {
            "name": "Zinc supplements",
            "type": "Supplement",
            "age_restrictions": None,
            "contraindications": [],
            "notes": "May reduce duration of symptoms"
        }
    ],
    "Viral Infection": [
        {
            "name": "Paracetamol",
            "type": "Fever reducer",
            "age_restrictions": None,
            "contraindications": ["paracetamol", "acetaminophen"],
            "notes": "For fever management"
        },
        {
            "name": "Cetirizine",
            "type": "Antihistamine",
            "age_restrictions": 2,
            "contraindications": ["cetirizine"],
            "notes": "For cold symptoms"
        },
        {
            "name": "Honey (warm water)",
            "type": "Natural remedy",
            "age_restrictions": 1,  # Not for infants under 1 year
            "contraindications": [],
            "notes": "Soothes throat, natural antimicrobial"
        },
        {
            "name": "Steam inhalation",
            "type": "Home remedy",
            "age_restrictions": None,
            "contraindications": [],
            "notes": "Helps clear nasal congestion"
        }
    ],
    "Acidity": [
        {
            "name": "Omeprazole",
            "type": "Proton pump inhibitor",
            "age_restrictions": 18,
            "contraindications": ["omeprazole"],
            "notes": "Reduces stomach acid production"
        },
        {
            "name": "Ranitidine alternatives (Famotidine)",
            "type": "H2 blocker",
            "age_restrictions": 12,
            "contraindications": ["famotidine"],
            "notes": "Alternative acid reducer"
        },
        {
            "name": "Antacid (Digene/ENO)",
            "type": "Antacid",
            "age_restrictions": None,
            "contraindications": [],
            "notes": "Quick relief from heartburn"
        },
        {
            "name": "Probiotics",
            "type": "Supplement",
            "age_restrictions": None,
            "contraindications": [],
            "notes": "Supports digestive health"
        }
    ],
    "Cardiac Risk": []  # NO MEDICATIONS - Emergency referral only
}

# ========================
# ML MODEL INITIALIZATION
# ========================

# Create training data for Naive Bayes
def create_training_data():
    """Generate training data from disease-symptom mappings"""
    X_train = []
    y_train = []
    
    for disease, symptoms in DISEASE_SYMPTOMS.items():
        # Create binary symptom vector
        symptom_vector = [1 if s in symptoms else 0 for s in AVAILABLE_SYMPTOMS]
        X_train.append(symptom_vector)
        y_train.append(disease)
        
        # Add variations with fewer symptoms (for robustness)
        if len(symptoms) > 3:
            # Random subsets of symptoms
            for i in range(3):
                subset_size = max(2, len(symptoms) - i - 1)
                subset = np.random.choice(symptoms, subset_size, replace=False)
                symptom_vector = [1 if s in subset else 0 for s in AVAILABLE_SYMPTOMS]
                X_train.append(symptom_vector)
                y_train.append(disease)
    
    return np.array(X_train), np.array(y_train)

# Initialize and train model
X_train, y_train = create_training_data()
disease_model = MultinomialNB()
disease_model.fit(X_train, y_train)

# ========================
# PREDICTION FUNCTIONS
# ========================

def predict_disease(symptoms):
    """
    Predict disease based on symptoms using ML model
    
    Args:
        symptoms: List of symptom names
        
    Returns:
        tuple: (predicted_disease, confidence_percentage)
    """
    # Create symptom vector
    symptom_vector = [1 if s in symptoms else 0 for s in AVAILABLE_SYMPTOMS]
    
    # Predict using Naive Bayes
    prediction = disease_model.predict([symptom_vector])[0]
    
    # Get confidence (probability)
    probabilities = disease_model.predict_proba([symptom_vector])[0]
    confidence = int(max(probabilities) * 100)
    
    # Rule-based override for cardiac symptoms (safety critical)
    cardiac_symptoms = set(DISEASE_SYMPTOMS["Cardiac Risk"])
    patient_symptoms = set(symptoms)
    
    # If 3+ cardiac symptoms present, override to Cardiac Risk
    if len(cardiac_symptoms.intersection(patient_symptoms)) >= 3:
        prediction = "Cardiac Risk"
        confidence = 95
    
    return prediction, confidence

# ========================
# MEDICINE RECOMMENDATION
# ========================

def get_medicine_recommendations(disease, patient_age, allergies):
    """
    Get safe medicine recommendations based on disease, age, and allergies
    
    Args:
        disease: Predicted disease name
        patient_age: Patient's age in years
        allergies: List of known drug allergies
        
    Returns:
        list: List of recommended medicines with details
    """
    # CRITICAL: No medicines for cardiac risk
    if disease == "Cardiac Risk":
        return []
    
    # Get disease-specific medicines
    all_medicines = MEDICINE_DATABASE.get(disease, [])
    
    # Filter based on age and allergies
    safe_medicines = []
    
    for med in all_medicines:
        # Check age restrictions
        if med["age_restrictions"] is not None:
            if patient_age < med["age_restrictions"]:
                continue  # Skip if patient too young
        
        # Check allergies
        if any(allergy in med["contraindications"] for allergy in allergies):
            continue  # Skip if patient allergic
        
        # Add to safe list
        safe_medicines.append({
            "Medicine": med["name"],
            "Type": med["type"],
            "Notes": med["notes"],
            "Age Appropriate": "Yes" if med["age_restrictions"] is None or patient_age >= med["age_restrictions"] else "No"
        })
    
    return safe_medicines

# ========================
# NLP SYMPTOM EXTRACTION
# ========================

# Symptom keyword mapping for NLP
SYMPTOM_KEYWORDS = {
    "fever": ["fever", "temperature", "hot", "burning up", "pyrexia"],
    "cough": ["cough", "coughing", "throat clearing"],
    "fatigue": ["tired", "fatigue", "exhausted", "weakness", "weak", "low energy"],
    "headache": ["headache", "head pain", "migraine"],
    "body_ache": ["body ache", "muscle pain", "body pain", "aching", "soreness"],
    "sore_throat": ["sore throat", "throat pain", "throat hurts"],
    "runny_nose": ["runny nose", "nasal discharge", "stuffy nose", "congestion"],
    "nausea": ["nausea", "nauseated", "feel sick", "queasy"],
    "vomiting": ["vomit", "vomiting", "throwing up", "puking"],
    "diarrhea": ["diarrhea", "loose motions", "loose stools"],
    "stomach_pain": ["stomach pain", "abdominal pain", "belly pain", "stomach ache"],
    "heartburn": ["heartburn", "acid reflux", "burning chest", "indigestion"],
    "chest_pain": ["chest pain", "chest discomfort", "chest pressure", "chest tightness"],
    "shortness_of_breath": ["shortness of breath", "breathing difficulty", "can't breathe", "breathless", "dyspnea"],
    "dizziness": ["dizzy", "dizziness", "lightheaded", "vertigo"],
    "rapid_heartbeat": ["rapid heartbeat", "racing heart", "palpitations", "heart racing"],
    "sweating": ["sweating", "perspiring", "cold sweat"],
    "loss_of_appetite": ["loss of appetite", "no appetite", "not hungry"],
    "weakness": ["weakness", "weak", "feeble"],
    "chills": ["chills", "shivering", "shaking", "cold"]
}

def extract_symptoms_from_text(text):
    """
    Extract symptoms from natural language text using keyword matching
    
    Args:
        text: Patient's symptom description
        
    Returns:
        list: Extracted symptom names
    """
    text_lower = text.lower()
    extracted_symptoms = []
    
    for symptom, keywords in SYMPTOM_KEYWORDS.items():
        for keyword in keywords:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_lower):
                if symptom not in extracted_symptoms:
                    extracted_symptoms.append(symptom)
                break
    
    return extracted_symptoms

# ========================
# UTILITY FUNCTIONS
# ========================

def get_disease_summary():
    """Get summary of all supported diseases"""
    return {
        disease: {
            "symptoms": symptoms,
            "info": DISEASE_INFO.get(disease, {})
        }
        for disease, symptoms in DISEASE_SYMPTOMS.items()
    }

def validate_symptoms(symptoms):
    """Validate if symptoms are in the available list"""
    return all(s in AVAILABLE_SYMPTOMS for s in symptoms)
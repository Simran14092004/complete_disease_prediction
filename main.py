# main.py
from flask import Flask, render_template, request, redirect, url_for, flash, session, g, jsonify
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3  # For persistent storage
from flask_mail import Mail, Message  # For sending emails
from dotenv import load_dotenv
from collections import Counter  # Import Counter
import os
import logging  # Import the logging module
import pandas as pd
#below three especially for depression model
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, template_folder='templates', static_folder='templates')
# Set the secret key
app.secret_key = os.getenv('SECRET_KEY', 'fallback_super_secret_key')  # Fallback ensures the app runs even without .env

# --- Configuration ---
# Database Configuration
DATABASE_PATH = 'health_database.db'  # SQLite database file

# Email Configuration (for contact form)
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT') or 587)
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER')

mail = Mail(app)

# --- Logging Setup ---
# Configure logging to write to a file (e.g., app.log) and the console
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),  # Log to a file
        logging.StreamHandler()         # Log to the console
    ]
)
logger = logging.getLogger(__name__)  # Get a logger instance

# --- Database Setup ---
def get_db():
    """Get database connection."""
    db = getattr(g, '_database', None)
    if db is None:
        db = sqlite3.connect(DATABASE_PATH)
        db.row_factory = sqlite3.Row  # Return rows as dictionaries
        g._database = db
    return db

@app.teardown_appcontext
def close_db(exception):
    """Close database connection."""
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    """Initialize the database (create tables)."""
    with app.app_context():
        db = get_db()
        with app.open_resource('schema.sql', mode='r') as f:
            db.cursor().executescript(f.read())
        db.commit()

def query_db(query, args=(), one=False):
    """Execute a database query."""
    db = get_db()
    cursor = db.execute(query, args)
    results = cursor.fetchall()
    cursor.close()
    return (results[0] if results else None) if one else results

def modify_db(query, args=()):
    """Modify the database (INSERT, UPDATE, DELETE)."""
    db = get_db()
    cursor = db.execute(query, args)
    db.commit()
    cursor.close()

# --- Model Loading ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

try:
    diabetes_model = joblib.load(os.path.join(BASE_DIR, 'Trained_sav_models/diabetes_model.sav'))
    logger.info("Diabetes model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading diabetes model: {e}")
    # exit(1)  # Don't exit here, handle more gracefully if other models might load
    diabetes_model = None  # Set to None to prevent errors later

try:
    parkinsons_model = joblib.load(os.path.join(BASE_DIR, 'Trained_sav_models/parkinsons_model.sav'))
    logger.info("Parkinson's model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading parkinson's model: {e}")
    # exit(1)
    parkinsons_model = None

try:
    heart_disease_model = joblib.load(os.path.join(BASE_DIR, 'Trained_sav_models/heart_disease_model.sav'))
    logger.info("Heart disease model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading heart disease model: {e}")
    # exit(1)
    heart_disease_model = None


try:
    depression_trained_model = joblib.load(os.path.join(BASE_DIR, 'Trained_sav_models\depression_model_pipeline (1).sav'))
    logger.info("Depression model pipeline loaded successfully.")
except FileNotFoundError:
    logger.error("Depression model pipeline file not found. Please check the file path.")
    depression_trained_model = None
except Exception as e:
    logger.error(f"Error loading depression model pipeline: {e}")
    depression_trained_model = None

try:
    # Load the generic disease model
    generic_trained_model = joblib.load(os.path.join(BASE_DIR, 'Trained_sav_models/random_forest_model.sav'))
    logger.info("Generic disease model loaded successfully.")
except FileNotFoundError:
    logger.error("Generic disease model file not found. Please check the file path.")
    generic_trained_model = None
except Exception as e:
    logger.error(f"Error loading generic model: {e}")
    generic_trained_model = None


#generic label encoder
try:
    with open(os.path.join(BASE_DIR, 'label_encoder_models/label_encoder.sav'), 'rb') as f:
        generic_label_encoder = joblib.load(f)
    logger.info("Generic disease label encoder loaded successfully.")
except FileNotFoundError:
    logger.error("Generic disease label encoder file not found. Please check the file path.")
    generic_label_encoder = None

#chart breast cancer
# Load the breast cancer dataset
try:
    file_path = os.path.join(BASE_DIR, r'increasing_disease_india\Breast_cancer_india(2016-2021).csv')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    breast_cancer_data = pd.read_csv(file_path)
    logger.info("Breast cancer data loaded successfully.")
except FileNotFoundError as e:
    logger.error(f"File not found: {e}")
    raise
except Exception as e:
    logger.error(f"Error loading breast cancer data: {e}")
    raise

#chart heart attack
# Load the heart attack dataset
# Load the heart attack dataset
try:
    file_path = os.path.join(BASE_DIR, r'increasing_disease_india/heart_attack_youngsters_india.csv')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    heart_attack_data = pd.read_csv(file_path)
    logger.info("Heart attack data loaded successfully.")
except FileNotFoundError as e:
    logger.error(f"File not found: {e}")
    raise
except Exception as e:
    logger.error(f"Error loading heart attack data: {e}")
    raise





# --- Helper Functions ---
def predict_disease(model, data):
    """Make predictions using the specified model."""
    if model is None:
        logger.error("predict_disease called with a None model.")
        return None
    try:
        logger.debug(f"Data shape for prediction: {data.shape}")
        logger.debug(f"Data content for prediction: {data}")
        prediction = model.predict(data)
        logger.debug(f"Raw prediction: {prediction}")
        return prediction[0]  # Return single prediction
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None  # Handle errors gracefully

def get_recommendations(disease, prediction):
    """Get recommendations based on disease and prediction."""
    recommendations = {
        "Diabetes": {
            "Positive": ["Continue regular checkups. Maintain a healthy lifestyle.", "Consult your doctor for personalized advice."],
            "Negative": ["Adopt a healthy diet. Increase physical activity.", "Monitor your blood sugar levels regularly."],
        },
        "Parkinson's Disease": {
            "Positive": ["Follow your treatment plan. Consider support groups.", "Engage in regular exercise and therapy."],
            "Negative": ["Consult a neurologist for diagnosis and treatment options.", "Learn about Parkinson's and available resources."],
        },
        "Heart Disease": {
            "Positive": ["Maintain a heart-healthy lifestyle. Follow your doctor's advice.", "Continue prescribed medications and attend follow-up appointments."],
            "Negative": ["Adopt a low-sodium, low-fat diet. Quit smoking.", "Engage in regular cardiovascular exercise."],
        },
        "Depression": {
            "Positive": ["Continue therapy and/or medication. Practice self-care.", "Stay connected with supportive people."],
            "Negative": ["Seek professional help from a therapist or counselor.", "Engage in activities you used to enjoy."],
        },
        "Generic Disease": {
            "Positive": ["Follow up with a healthcare professional.", "Maintain a healthy lifestyle."],
            "Negative": ["Consult a doctor for proper diagnosis.", "Get enough rest and manage stress."],
        }
    }
    return recommendations.get(disease, {}).get(prediction, [])

def validate_input(form, expected_fields, types):
    """Validate form input."""
    validated_data = {}
    for field in expected_fields:
        if field not in form:
            flash(f"Missing field: {field}", "error")
            return None
        try:
            validated_data[field] = types[field](form[field])
        except ValueError:
            flash(f"Invalid data type for {field}. Expected {types[field].__name__}.", "error")
            return None
    return validated_data

# --- Routes ---
@app.route('/')
def index():
    """Home page."""
    articles = [
        {"title": "Understanding Depression", "description": "Learn about the causes and symptoms of depression.",
         "image": "static/images/article1.jpg", "link": "#"},
        {"title": "Heart Health Tips", "description": "Essential tips for maintaining a healthy heart.",
         "image": "static/images/article2.jpg", "link": "#"},
        {"title": "Managing Diabetes", "description": "Effective strategies for managing diabetes.",
         "image": "static/images/article3.jpg", "link": "#"},
        {"title": "Parkinson's Disease", "description": "Information and resources about Parkinson's.",
         "image": "static/images/article4.jpg", "link": "#"},
    ]
    courses = [
        {"title": "Mental Health First Aid",
         "description": "Learn to provide initial support to someone experiencing a mental health crisis.",
         "link": "https://www.mentalhealth.org/get-help/immediate-help"},
        {"title": "Heart Disease Prevention",
         "description": "Learn about lifestyle changes and medical treatments to prevent heart disease.",
         "link": "https://www.heart.org/en/health-topics/consumer-healthcare/what-is-cardiovascular-disease/how-to-help-prevent-heart-disease-at-any-age"},
        {"title": "Diabetes Management", "description": "Learn how to manage your diabetes effectively.",
         "link": "https://www.niddk.nih.gov/health-information/diabetes/overview/managing-diabetes"},
        {"title": "Living with Parkinson's",
         "description": "Resources and support for individuals and families facing Parkinson's disease.",
         "link": "https://www.parkinson.org/"},
    ]
    return render_template('index.html', articles=articles, courses=courses)

@app.route('/mental', methods=['GET', 'POST'])
def mental():
    """Mental disease prediction page (Depression)."""
    # Expected fields and their data types
    expected_fields = [
        'Age', 'Gender', 'Profession', 'Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction',
        'Job Satisfaction', 'Sleep Duration', 'Dietary Habits', 'Degree', 'Have you ever had suicidal thoughts ?',
        'Work/Study Hours', 'Financial Stress', 'Family History of Mental Illness'
    ]

    # Field types for validation
    types = {
        'Age': float, 'Gender': str, 'Profession': str, 'Academic Pressure': float, 'Work Pressure': float,
        'CGPA': float, 'Study Satisfaction': float, 'Job Satisfaction': float, 'Sleep Duration': str,
        'Dietary Habits': str, 'Degree': str, 'Have you ever had suicidal thoughts ?': str,
        'Work/Study Hours': float, 'Financial Stress': float, 'Family History of Mental Illness': str
    }

    if request.method == 'POST' and depression_trained_model:
        # Validate input data
        validated_data = {}
        for field in expected_fields:
            if field not in request.form:
                flash(f"Missing field: {field}", "error")
                return render_template('mental.html', disease_type='depression')
            try:
                validated_data[field] = types[field](request.form[field])
            except ValueError:
                flash(f"Invalid data type for {field}. Expected {types[field].__name__}.", "error")
                return render_template('mental.html', disease_type='depression')

        try:
            # Convert input data to DataFrame
            input_df = pd.DataFrame([validated_data])

            # Make prediction using the trained pipeline
            prediction = depression_trained_model.predict(input_df)[0]
            prediction_proba = depression_trained_model.predict_proba(input_df)[0]

            # Interpret prediction
            if prediction == 0:
                prediction_text = "Not Depressed"
                recommendations = ["Maintain a healthy lifestyle", "Continue good habits"]
            else:
                prediction_text = "Depressed"
                recommendations = ["Seek professional help", "Consider therapy or counseling", "Follow a balanced lifestyle"]

            logger.info(f"Prediction: {prediction_text}, Probabilities: {prediction_proba}")

            return render_template(
                'mental.html',
                disease_type='depression',
                prediction=prediction_text,
                recommendations=recommendations
            )
        except Exception as e:
            logger.error(f"Error during depression prediction: {e}")
            flash("An unexpected error occurred. Please try again.", "error")

    # Render the page for GET method
    return render_template('mental.html', disease_type='depression')


@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    """Diabetes prediction page."""
    if request.method == 'POST':
        # Dataset terms for diabetes
        expected_fields = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        types = {field: float for field in expected_fields}
        types['Pregnancies'] = int
        types['Age'] = int

        validated_data = validate_input(request.form, expected_fields, types)
        if validated_data and diabetes_model:  # Check if the model is loaded
            try:
                # Prepare data in the correct order
                data = np.array([[
                    validated_data['Pregnancies'], validated_data['Glucose'], validated_data['BloodPressure'],
                    validated_data['SkinThickness'], validated_data['Insulin'], validated_data['BMI'],
                    validated_data['DiabetesPedigreeFunction'], validated_data['Age']
                ]])
                logger.debug(f"Input data for diabetes prediction: {data}")

                # Make prediction
                prediction = predict_disease(diabetes_model, data)
                prediction_text = "Positive" if prediction == 1 else "Negative"

                # Get recommendations
                recommendations_positive = get_recommendations("Diabetes", "Positive")
                recommendations_negative = get_recommendations("Diabetes", "Negative")
                recommendations = recommendations_positive if prediction == 1 else recommendations_negative

                return render_template('diabetes.html', disease_type='diabetes', prediction=prediction_text,
                                       recommendations=recommendations)
            except Exception as e:
                logger.error(f"Error during diabetes prediction: {e}")
                flash(f"An unexpected error occurred: {e}", "error")
        return render_template('diabetes.html', disease_type='diabetes')
    return render_template('diabetes.html', disease_type='diabetes')


@app.route('/heart', methods=['GET', 'POST'])
def heart():
    """Heart disease prediction page."""
    if request.method == 'POST':
        # Dataset terms for heart disease
        expected_fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        types = {'age': int, 'sex': int, 'cp': int, 'trestbps': int, 'chol': int, 
                 'fbs': int, 'restecg': int, 'thalach': int, 'exang': int, 
                 'oldpeak': float, 'slope': int, 'ca': int, 'thal': int}

        validated_data = validate_input(request.form, expected_fields, types)
        if validated_data and heart_disease_model:  # Check if the model is loaded
            try:
                # Prepare data in the correct order
                data = np.array([[
                    validated_data['age'], validated_data['sex'], validated_data['cp'], validated_data['trestbps'],
                    validated_data['chol'], validated_data['fbs'], validated_data['restecg'], validated_data['thalach'],
                    validated_data['exang'], validated_data['oldpeak'], validated_data['slope'], validated_data['ca'],
                    validated_data['thal']
                ]])
                logger.debug(f"Input data for heart prediction: {data}")

                # Make prediction
                prediction = predict_disease(heart_disease_model, data)
                prediction_text = "Positive" if prediction == 1 else "Negative"

                # Get recommendations
                recommendations_positive = get_recommendations("Heart Disease", "Positive")
                recommendations_negative = get_recommendations("Heart Disease", "Negative")
                recommendations = recommendations_positive if prediction == 1 else recommendations_negative

                return render_template('heart.html', prediction=prediction_text, recommendations=recommendations)
            except Exception as e:
                logger.error(f"Error during heart disease prediction: {e}")
                flash(f"An unexpected error occurred: {e}", "error")
        return render_template('heart.html')
    return render_template('heart.html')

@app.route('/parkinson', methods=['GET', 'POST'])
def parkinson():
    """Parkinson's disease prediction page."""
    if request.method == 'POST':
        # Dataset terms for Parkinson's disease
        expected_fields = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 
                           'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 
                           'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 
                           'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']
        types = {field: float for field in expected_fields}

        validated_data = validate_input(request.form, expected_fields, types)
        if validated_data and parkinsons_model:  # Check if the model is loaded
            try:
                # Prepare data in the correct order
                data = np.array([[
                    validated_data[field] for field in expected_fields
                ]])
                logger.debug(f"Input data for Parkinson's prediction: {data}")

                # Make prediction
                prediction = predict_disease(parkinsons_model, data)
                prediction_text = "Positive" if prediction == 1 else "Negative"

                # Get recommendations
                recommendations_positive = get_recommendations("Parkinson's Disease", "Positive")
                recommendations_negative = get_recommendations("Parkinson's Disease", "Negative")
                recommendations = recommendations_positive if prediction == 1 else recommendations_negative

                return render_template('chronic.html', disease_type='parkinson', prediction=prediction_text,
                                       recommendations=recommendations)
            except Exception as e:
                logger.error(f"Error during Parkinson's prediction: {e}")
                flash(f"An unexpected error occurred: {e}", "error")
        return render_template('parkinson.html', disease_type='parkinson')
    return render_template('parkinson.html', disease_type='parkinson')


@app.route('/generic', methods=['GET', 'POST'])
def generic():
    """Generic disease prediction page."""
    all_symptoms_list = [
        'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills',
        'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting',
        'burning_micturition', 'spotting_urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets',
        'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level',
        'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion',
        'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes',
        'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes',
        'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise',
        'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure',
        'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements',
        'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising',
        'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
        'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech',
        'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness',
        'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell',
        'bladder_discomfort', 'foul_smell_of_urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching',
        'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body',
        'belly_pain', 'abnormal_menstruation', 'dischromic_patches', 'watering_from_eyes', 'increased_appetite',
        'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
        'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
        'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations',
        'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting',
        'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze'
        # ... (remaining symptoms)
    ]

    if request.method == 'POST' and generic_trained_model and generic_label_encoder:
        try:
            # Get selected symptoms
            selected_symptoms = request.form.getlist('symptoms')
            if not selected_symptoms:
                flash("Please select at least one symptom.", "error")
                return render_template('generic.html', all_symptoms=all_symptoms_list)

            # Create input vector
            input_vector = [1 if symptom in selected_symptoms else 0 for symptom in all_symptoms_list]
            data = np.array([input_vector])  # Ensure 2D array for prediction
            logger.debug(f"Input vector: {input_vector}")
            logger.debug(f"Input data for prediction: {data}")

            # Make prediction
            prediction_encoded = generic_trained_model.predict(data)[0]
            prediction = generic_label_encoder.inverse_transform([prediction_encoded])[0]
            logger.debug(f"Encoded prediction: {prediction_encoded}")
            logger.debug(f"Decoded prediction: {prediction}")

            # Top 3 predictions
            probabilities = generic_trained_model.predict_proba(data)[0]
            top_3_indices = np.argsort(probabilities)[-3:][::-1]
            top_3_predictions = [
                (generic_label_encoder.inverse_transform([i])[0], probabilities[i] * 100)
                for i in top_3_indices
            ]
            logger.debug(f"Top 3 predictions: {top_3_predictions}")

            # # Recommendations
            # recommendations = get_recommendations("Generic Disease", prediction)
             # Recommendations
            # Determine whether the prediction is positive or negative (logic can be adapted)
            prediction_label = "Positive" if prediction_encoded == 1 else "Negative"
            recommendations = get_recommendations("Generic Disease", prediction_label)


            return render_template(
                'generic.html',
                all_symptoms=all_symptoms_list,
                prediction=prediction,
                recommendations=recommendations,
                top_3_diseases=top_3_predictions
            )
        except Exception as e:
            logger.error("Error during generic disease prediction", exc_info=True)
            flash("An unexpected error occurred. Please try again.", "error")
    return render_template('generic.html', all_symptoms=all_symptoms_list)




@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page."""
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = query_db('SELECT * FROM users WHERE email = ?', [email], one=True)

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['name']
            flash('Logged in successfully!', 'success')
            return redirect(url_for('index'))  # Redirect to home or profile
        else:
            flash('Invalid login credentials. Please try again.', 'error')
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Signup page."""
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        # Basic validation (you should add more robust validation)
        if not name or not email or not password:
            flash('All fields are required.', 'error')
            return render_template('signup.html')

        if query_db('SELECT * FROM users WHERE email = ?', [email], one=True):
            flash('Email already exists. Please use a different email.', 'error')
            return render_template('signup.html')

        # Hash the password
        hashed_password = generate_password_hash(password)

        # Insert the new user into the database
        modify_db('INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
                  [name, email, hashed_password])

        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')


@app.route('/logout')
def logout():
    """Logout route."""
    session.pop('user_id', None)  # Clear the user session
    session.pop('username', None)
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))  # Redirect to the home page



@app.route('/contact', methods=['GET', 'POST'])
def contact():
    """Contact Us page."""
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        # Send email (using Flask-Mail)
        try:
            msg = Message("Contact Form Submission", recipients=[app.config['MAIL_USERNAME']])  # Send to yourself
            msg.body = f"Name: {name}\nEmail: {email}\n\nMessage:\n{message}"
            mail.send(msg)
            flash('Your message has been sent. We will get back to you soon!', 'success')
            return redirect(url_for('contact'))  # Redirect to a thank you page or back to contact
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            flash(f"Failed to send message: {e}", 'error')
            return render_template('contacts.html')

    return render_template('contacts.html')


@app.route('/about', methods=['GET'])
def about():
    """About Us page."""
    return render_template('aboutus.html')



# route for  breast cancer dataset chart


@app.route('/charts/breast-cancer')
def breast_cancer_chart():
    return render_template('breast_chart.html')

@app.route('/api/breast-cancer-data')
def breast_cancer_data_api():
    # Transform the data to JSON format for visualization
    data = breast_cancer_data.to_dict(orient='list')
    return jsonify(data)


#route for heart attack dataset chart
@app.route('/charts/heart-attack')
def heart_attack_chart():
    return render_template('heart_attack.html')

@app.route('/api/heart-attack-data')
def heart_attack_data_api():
    # Transform the heart attack dataset into JSON format
    data = heart_attack_data.to_dict(orient='list')
    return jsonify(data)



if __name__ == '__main__':
    # Initialize the database before running the app
    with app.app_context():
        init_db()  # Create tables if they don't exist
    app.run(debug=True)

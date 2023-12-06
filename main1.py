import streamlit as st
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import speech_recognition as sr
import pyttsx3

# Set page title and icon FIRST
st.set_page_config(page_title='Diabetes Checkup', page_icon=':hospital:')

# Load your data
df = pd.read_csv(r'C:\Users\ADMIN\Desktop\AIML_Project\diabetes.csv')

# Custom CSS for styling
st.markdown(
    """
    <style>
        body {
            background-color: #F8F9FA;
            color: #1E1E1E;
            font-family: 'Arial', sans-serif;
        }
        .main-container {
            padding: 2rem;
            max-width: 800px;
            margin: auto;
        }
        .sidebar .sidebar-content {
            background-color: #343A40;
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-top: 2rem;
        }
        .footer {
            text-align: center;
            padding: 10px;
            font-size: 12px;
            color: #6C757D;
        }
        .check-button {
            background-color: #007BFF;
            color: white;
            font-weight: bold;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.title('Diabetes Checkup')
st.write("Welcome to the Diabetes Checkup app. Enter your health data below and click 'Check' to get a quick prediction.")

# Main container for a cleaner layout
main_container = st.container()

# Create input fields for user data
with main_container:
    st.header('Enter Your Health Data:')
    pregnancies = st.number_input('Pregnancies', 0, 17, 3)
    glucose = st.number_input('Glucose', 0, 200, 120)
    blood_pressure = st.number_input('Blood Pressure', 0, 122, 70)
    skin_thickness = st.number_input('Skin Thickness', 0, 100, 20)
    insulin = st.number_input('Insulin', 0, 846, 79)
    bmi = st.number_input('BMI', 0, 67, 20)
    dpf = st.number_input('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
    age = st.number_input('Age', 21, 88, 33)

# Check button to trigger prediction update
if main_container.button('Check'):
    # Create DataFrame from user input
    user_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    user_data_df = pd.DataFrame(user_data, index=[0])

    # Train the model
    x = df.drop(['Outcome'], axis=1)
    y = df['Outcome']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0) 
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)

    # Make predictions
    user_result = rf.predict(user_data_df)

    # Display result and provide health tip
    main_container.subheader('Your Report:')
    result_message = 'You are healthy.' if user_result[0] == 0 else 'You are not healthy.'
    if user_result[0] == 0:
        main_container.success(result_message)
    else:
        main_container.error(result_message)

    # Initialize text-to-speech engine
    engine = pyttsx3.init()

    # Display health tip
    main_container.subheader('Health Tip:')
    if user_result[0] == 0:
        healthy_tip = 'Great news! Keep maintaining a healthy lifestyle to prevent diabetes.'
        main_container.write(healthy_tip)
        # Convert text to speech for healthy tip
        engine.say(healthy_tip)
        engine.runAndWait()
    else:
        main_container.write('Please consult with a healthcare professional for further guidance.')
        main_container.write('In the meantime, consider making lifestyle changes such as regular exercise and a balanced diet to manage diabetes.')
        # Unhealthy tip for diabetic patients
        unhealthy_tip = 'It is essential to monitor your blood sugar levels regularly, take prescribed medications, and follow a diabetic-friendly diet.'
        main_container.error(unhealthy_tip)
        # Convert text to speech for unhealthy tip
        engine.say(unhealthy_tip)
        engine.runAndWait()

    # Model Accuracy
    main_container.subheader('Model Accuracy:')
    model_accuracy = accuracy_score(y_test, rf.predict(x_test)) * 100
    main_container.write(f'The model accuracy on the test data is {model_accuracy:.2f}%.')

# Voice command button and processing with suggestions
if st.button("Voice Command"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Say a command...")
        recognizer.adjust_for_ambient_noise(source, duration=5)
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio).lower()
        st.write(f"Command recognized: {command}")
        if "check" in command:
            # Trigger the 'Check' button action
            main_container.button('Check').click()
        elif "suggestion" in command:
            # Provide suggestions based on the predicted outcome
            if user_result[0] == 0:
                st.write("Healthy tip suggestion: Keep a balanced diet and engage in regular physical activity.")
                # Convert text to speech for healthy suggestion
                engine.say("Healthy tip suggestion: Keep a balanced diet and engage in regular physical activity.")
                engine.runAndWait()
            else:
                st.write("Unhealthy tip suggestion: Consult with a healthcare professional and consider lifestyle changes.")
                # Convert text to speech for unhealthy suggestion
                engine.say("Unhealthy tip suggestion: Consult with a healthcare professional and consider lifestyle changes.")
                engine.runAndWait()
        else:
            st.warning("Invalid command. Try saying 'Check' or 'Suggestion'.")

    except sr.UnknownValueError:
        st.warning("Sorry, I could not understand the audio.")
    except sr.RequestError as e:
        st.warning(f"Could not request results from Google Speech Recognition service; {e}")

# Add a footer with an image
footer = """
---
<div class="footer">
    <p>© 2023 Diabetes Checkup App</p>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)

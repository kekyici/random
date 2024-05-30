import streamlit as st
from constant import *
from data_processor import DataProcessor
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
import pickle
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

hide_st_style = """
                <style>
                #MainMenu {visibility : hidden;}
                footer {visibility : hidden;}
                header {visibility : hidden;}
                </style>
                """

# default data
def load_and_process_data():
    processor = DataProcessor(DATA_URL, COLUMN_NAMES, Z_SCORE_THRESHOLD)
    data = processor.data
    numerical_variables = ['age', 'education_num', 'hours_per_week']
    data = processor.remove_outliers(numerical_variables)
    data = processor.transform_data()
    return data

data = load_and_process_data()   

st.set_page_config(
    page_title="SMART",
    page_icon="memo",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.linkedin.com/in/abhiiiman',
        'Report a bug': "https://www.github.com/abhiiiman",
        'About': "## A 'Student Performance and Placement Prediction Tool' by Abhijit Mandal & Divyanshi"
    }
)

#remove all the default streamlit configs here
st.markdown(hide_st_style, unsafe_allow_html=True)

with st.sidebar:
    selected = option_menu(
        menu_title = "S.M.A.R.T",
        options = ["Home", "Data Exploration", "Analysis", "Predictions", "Model", "Download CSV", "Settings", "Contributors"],
        icons = ["house", "upload", "graph-up", "magic", "speedometer", "file-earmark-arrow-down", "gear", "people"],
        menu_icon= "robot",
        default_index = 0,
    )

# ========= HOME TAB =========
if selected == "Home":
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title('S.M.A.R.T :rocket:')
        st.header("Student Management 🏠 And")
        st.header("Recruitment Tool 🔮")
        st.divider()
        st.header("About :memo:")
        st.markdown('''
        ####
        We are thrilled to introduce you to SMART, your all-in-one solution for student
        management and recruitment needs
        providing a comprehensive platform for tracking academic progress, 
        facilitating career development, and predicting placement opportunities.
        With SMART, educational institutions can efficiently manage student data,
        track their academic performance, and streamline the recruitment process. 
        Meanwhile, students can access personalized career guidance, 
        explore job opportunities, and receive tailored recommendations
        for enhancing their employability. 
        Join us on this exciting journey as we revolutionize
        student management and recruitment with SMART!
        ''')
        
        st.markdown("#### `Get Started Now!`")

    with col2:
        st.image("Images\\home_tab_pic.png")

# ========= UPLOAD TAB =========
if selected == "Data Exploration":

    st.title("Census Income Prediction Web App")
    st.header("Explore the dataset and predict income levels.")
    # Data Exploration Page
    # Sidebar for data exploration options

    st.header("Data Exploration ")
    st.subheader("Summary Statistics")
    st.write(data.describe())

    st.subheader("Data Visualization")
    selected_feature = st.selectbox("Select Feature for Visualization", data.columns)
    selected_chart = st.selectbox("Select Chart Type", ["Histogram", "Bar Chart", "Scatter Plot"])
    filtered_data = data[data["income"] == 1] if st.checkbox("Filter High Income") else data



    # Data visualization
    if selected_chart == "Histogram":
        st.bar_chart(filtered_data[selected_feature].value_counts())
    elif selected_chart == "Bar Chart":
        st.bar_chart(filtered_data[selected_feature].value_counts())
    else:
        st.scatter_chart(filtered_data.sample(100), x="age", y="hours_per_week")


# ========= ANALYSIS TAB =========
if selected == "Analysis":
   
        st.balloons()
        st.title("Data Analysis 📊")
        st.title("Census Income Prediction Web App")
        st.header("Explore the dataset and predict income levels.")
        # Data Exploration Page
        # Sidebar for data exploration options

        st.header("Data Exploration ")
        st.subheader("Summary Statistics")
        st.write(data.describe())

        st.subheader("Data Visualization")
        selected_feature = st.selectbox("Select Feature for Visualization", data.columns)
        selected_chart = st.selectbox("Select Chart Type", ["Histogram", "Bar Chart", "Scatter Plot"])
        filtered_data = data[data["income"] == 1] if st.checkbox("Filter High Income") else data



        # Data visualization
        if selected_chart == "Histogram":
            st.bar_chart(filtered_data[selected_feature].value_counts())
        elif selected_chart == "Bar Chart":
            st.bar_chart(filtered_data[selected_feature].value_counts())
        else:
            st.scatter_chart(filtered_data.sample(100), x="age", y="hours_per_week")

# ========= PREDICTION TAB =======
def transform_input(age, workclass, fnlwgt, education, education_num, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country):
    # Define transformation mappings
    workclass_mapping = {
        'Private': 0, 'Self-emp-not-inc': 1, 'Self-emp-inc': 2, 'Federal-gov': 3, 'Local-gov': 4,
        'State-gov': 5, 'Without-pay': 6, 'Never-worked': 7
    }
    education_mapping = {
        'Bachelors': 0, 'Some-college': 1, '11th': 2, 'HS-grad': 3, 'Prof-school': 4, 'Assoc-acdm': 5,
        'Assoc-voc': 6, '9th': 7, '7th-8th': 8, '12th': 9, 'Masters': 10, '1st-4th': 11, '10th': 12,
        'Doctorate': 13, '5th-6th': 14, 'Preschool': 15
    }
    marital_status_mapping = {
        'Single': 0, 'Married': 1
    }
    occupation_mapping = {
        'Blue_collar': 0, 'White_collar': 1, 'Brown_collar/Protective_service': 2, 'Pink_collar/Service_and_sales': 3
    }
    relationship_mapping = {
        'Married': 0, 'Single': 1, 'Separated': 2
    }
    race_mapping = {
        'White': 0, 'Asian-Pac-Islander': 1, 'Amer-Indian-Eskimo': 2, 'Other': 3, 'Black': 4
    }
    sex_mapping = {
        'Male': 0, 'Female': 1
    }
    native_country_mapping = {
        'United-States': 0, 'Cambodia': 1, 'England': 2, 'Puerto-Rico': 3, 'Canada': 4, 'Germany': 5,
        'Outlying-US(Guam-USVI-etc)': 6, 'India': 7, 'Japan': 8, 'Greece': 9, 'South': 10, 'China': 11,
        'Cuba': 12, 'Iran': 13, 'Honduras': 14, 'Philippines': 15, 'Italy': 16, 'Poland': 17, 'Jamaica': 18,
        'Vietnam': 19, 'Mexico': 20, 'Portugal': 21, 'Ireland': 22, 'France': 23, 'Dominican-Republic': 24,
        'Laos': 25, 'Ecuador': 26, 'Taiwan': 27, 'Haiti': 28, 'Columbia': 29, 'Hungary': 30, 'Guatemala': 31,
        'Nicaragua': 32, 'Scotland': 33, 'Thailand': 34, 'Yugoslavia': 35, 'El-Salvador': 36, 'Trinadad&Tobago': 37,
        'Peru': 38, 'Hong': 39, 'Holand-Netherlands': 40
    }

    # Apply transformations
    workclass_encoded = workclass_mapping.get(workclass, -1)
    education_encoded = education_mapping.get(education, -1)
    marital_status_encoded = marital_status_mapping.get(marital_status, -1)
    occupation_encoded = occupation_mapping.get(occupation, -1)
    relationship_encoded = relationship_mapping.get(relationship, -1)
    race_encoded = race_mapping.get(race, -1)
    sex_encoded = sex_mapping.get(sex, -1)
    native_country_encoded = native_country_mapping.get(native_country, -1)

    return [age, workclass_encoded, fnlwgt, education_encoded, education_num, marital_status_encoded, occupation_encoded, relationship_encoded, race_encoded, sex_encoded, capital_gain, capital_loss, hours_per_week, native_country_encoded]

if selected == "Predictions":
    col1, col2 = st.columns(2)
    with col1:
        st.title("Income Level Prediction ⚡")
        st.subheader("Provide the inputs below 👇🏻")
        st.divider()
        st.markdown("##### _Here we will use <span style='color:yellow'>Random Forest 🤖</span> Machine Learning Algorithm to create our Model to predict the Income Level of Individuals_.", unsafe_allow_html=True)
        st.markdown("##### _You just need to provide the individual's data to get started and predict their income level using our <span style='color:yellow'>well trained Model right here</span>_.", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        # Get user input for Age
        age = st.slider('Enter the Age 👇🏻', min_value=18, max_value=90, step=1)
        # Get user input for fnlwgt
        fnlwgt = st.number_input('Enter the fnlwgt 👇🏻', min_value=0)
        # Get user input for education_num
        education_num = st.slider('Enter the Education Num 👇🏻', min_value=1, max_value=16, step=1)
        # Get user input for capital_gain
        capital_gain = st.number_input('Enter the Capital Gain 👇🏻', min_value=0)
        # Get user input for capital_loss
        capital_loss = st.number_input('Enter the Capital Loss 👇🏻', min_value=0)
        # Get user input for hours_per_week
        hours_per_week = st.slider('Enter the Hours Per Week 👇🏻', min_value=1, max_value=100, step=1)
        predict_button = st.button("Predict the Income Level ⚡")
        model_choice = st.selectbox("Select Model", ["Random Forest", "Decision Tree", "SVM", "Naive Bayes", "Logistic Regression"])
    

    with col2:
        # Get user input for Workclass
        workclass = st.selectbox('Choose Workclass 🏢', ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
        # Get user input for Education
        education = st.selectbox('Choose Education Level 🎓', ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'])
        # Get user input for Marital Status
        marital_status = st.selectbox('Choose Marital Status 💍', ['Single', 'Married'])
        # Get user input for Occupation
        occupation = st.selectbox('Choose Occupation 💼', ['Blue_collar', 'White_collar', 'Brown_collar/Protective_service', 'Pink_collar/Service_and_sales'])
        # Get user input for Relationship
        relationship = st.selectbox('Choose Relationship Status 💑', ['Married', 'Single', 'Separated'])
        # Get user input for Race
        race = st.selectbox('Choose Race 🌍', ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
        # Get user input for Sex
        sex = st.selectbox('Choose Sex 🧑🏻‍🦱👧🏻', ['Male', 'Female'])
        # Get user input for Native Country
        native_country = st.selectbox('Choose Native Country 🌎', ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'])

    # Check if the Predict Income button is clicked
    if predict_button:
        st.balloons()
        
        # Transform the input features
        user_data = transform_input(age, workclass, fnlwgt, education, education_num, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country)
     # Prepare the user input as a dataframe
        user_df = pd.DataFrame([user_data], columns=[
            'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
            'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country'
        ])
        
        st.divider()
        st.markdown("* ## Input Dataframe ⬇️")
        st.write(user_df)

        X = data.drop("income", axis=1)
        y = data["income"]

        # Label Encoding
        label_encoders = {}
        for column in X.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column])
            label_encoders[column] = le

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_choice == "Random Forest":
            clf = RandomForestClassifier()
        elif model_choice == "Decision Tree":
            clf = DecisionTreeClassifier()
        elif model_choice == "SVM":
            clf = SVC()
        elif model_choice == "Naive Bayes":
            clf = GaussianNB()
        elif model_choice == "Logistic Regression":
            clf = LogisticRegression()

        clf.fit(X_train, y_train)
        
            # Predict on user input
        prediction = clf.predict([user_data])
        prediction_proba = clf.predict_proba([user_data])

        # Display the prediction result
        st.markdown("* ## Prediction Result ✅")
        if prediction == 1:
            st.markdown("### <span style='color:lightgreen'>Income >50K 🎉</span>", unsafe_allow_html=True)
        else:
            st.markdown("### <span style='color:red'>Income <=50K 😢</span>", unsafe_allow_html=True)

# ========= PERFORMANCE TAB ======
if selected == "Model":
    le = LabelEncoder()
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = le.fit_transform(data[column])


    st.title("Census Income Prediction Web App")
    st.header("Income Prediction")

    # Sidebar for income prediction
    model = st.selectbox("Select Model", ["Random Forest", "Decision Tree", "SVM", "Naive Bayes", "Logistic Regression"])
    selected_features = st.multiselect("Select Features for Prediction", data.columns[:-1])
    test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

    # Income prediction
    if selected_features and st.button("Train and Predict"):
        # Data preprocessing
        X = data[selected_features]
        y = data["income"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        if model == "Random Forest":
            clf = RandomForestClassifier()
        elif model == "Decision Tree":
            clf = DecisionTreeClassifier()
        elif model == "SVM":
            clf = SVC()
        elif model == "Naive Bayes":
            clf = GaussianNB()
        elif model == "Logistic Regression":
            clf = LogisticRegression()

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Display results
        st.subheader("Model Evaluation")
        st.write(f"Selected Model: {model}")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        st.text("Classification Report:")
        st.code(classification_report(y_test, y_pred, target_names=["<=50K", ">50K"]))

# ========= CONTRIBUTORS =========
if selected == "Contributors":
    st.title("About Us ⚡")
    st.header("Team HardCoders 🦾")

    col1, col2 = st.columns(2)
    with col1:
        st.image("Images\\A-pfp.png")
        st.subheader("1️⃣ Abhijit Mandal")
        st.markdown('''
            * **`Github`** ⭐  
                https://github.com/abhiiiman
            * **`Linkedin`**  🔗 
                https://linkedin.com/in/abhiiiman
            * **`Portfolio`** 🌐
                https://abhiiiman.github.io/Abhijit-Mandal
        ''')

    with col2:
        st.image("Images\\D-pfp.png")
        st.subheader("2️⃣ Divyanshi")
        st.markdown('''
            * **`Github`** ⭐
                https://github.com/Divvyanshiii
            * **`Linkedin`**  🔗 
                https://linkedin.com/in/divyanshi-shrivastav
        ''')
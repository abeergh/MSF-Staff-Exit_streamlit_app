import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
#%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from streamlit_option_menu import option_menu
from streamlit_option_menu import option_menu
from scipy.stats import chi2_contingency
from sklearn.feature_selection import chi2
from sklearn.metrics import mean_squared_error, roc_curve, roc_auc_score

# Initialize a session state variable to store the data
if "data" not in st.session_state:
    st.session_state.data = None
    
with st.sidebar:
    st.image("logo.png", use_column_width=True)
    selected_tab = option_menu("Main Menu", ['Home', 'Upload Data', 'Statistical Analysis', 'Modeling', 'Predict Staff Exit'], 
        icons=['house', 'upload', 'kanban', 'sliders', 'person'], menu_icon="cast", default_index=4)
    selected_tab

if selected_tab == "Home":
    st.markdown(
        """
        <style>
        .text-justify {
            text-align: justify;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    # Title of the app
    st.header("Welcome to our Staff Exit Prediction App!")
    # Add your image
    image_path = 'retention.png'
    st.image(image_path)
    # Add the app summary
    st.markdown("""<div class="text-justify">
                <b>This Streamlit application</b> is a tool for predicting the likelihood of staff members leaving or staying working for Medecins Sans Frontieres (MSF). The application has different sections, the 'Upload Data' tab allows user to input the staff assignments data. In the 'Statistical Analysis' tab, the app leverages chi-squared tests and t-tests to determine associations between various variables and staff exit. In 'Modeling,' a logistic regression model is trained to make predictions. The 'Predict Staff Exit' tab is the central feature, enabling users to input staff information and receive a prediction regarding whether the staff member will stay or leave the organization.</div>    """,
    unsafe_allow_html=True)

        
if selected_tab == "Upload Data":
    st.title("Assignment Data Upload")

    # Upload assignment data
    uploaded_file = st.file_uploader("Upload assignment data (CSV)", type=["csv"])


    # Load Data Button
    if st.button("Load Data"):
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            # Transform the turnover variable(dependent variable) into 0 if turnover = No and 1 if turnover = Yes
            data['turnover'] = data['turnover'].apply(lambda x: 1 if x == 'Yes' else 0)
            # Remove rows with contract_type_category = "Detachment". Detachment are short assignments and doesn't require engagement, someone could have one detachment only
            data = data[data['contract_type_category'] != 'Detachment']
            
            st.session_state.data = data
            st.success("Data loaded successfully.")
        else:
            st.warning("No data to load. Upload a CSV file first.")

    # Perform data preprocessing and model training
    if st.session_state.data is not None:
        st.subheader("Data Preview:")
        st.write(st.session_state.data.head())
        
if selected_tab == "Statistical Analysis":
    st.title("Statistical Analysis")
    st.markdown('<div style="text-align: justify;">The Statistical Analysis tab is designed to provide insights into the assignments dataset by evaluating associations between different variables. The Chi-Squared test is used to examine associations between categorical variables and staff exit, and the T-test to analyze numerical variables in relation to staff exit. These tests help us uncover valuable insights that contribute to a better understanding of the dataset.</div>', unsafe_allow_html=True)
    if st.session_state.data is not None:
        data = st.session_state.data
        duration_turnover_yes = data[data['turnover'] == 1]['assignment_length_months']
        duration_turnover_no = data[data['turnover'] == 0]['assignment_length_months']
        assignmentcount_turnover_yes = data[data['turnover'] == 1]['assignments_count']
        assignmentcount_turnover_no = data[data['turnover'] == 0]['assignments_count']
        
        st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

        # Check association between categorical variables and turnover
        categorical_vars = ['gender', 'age_group', 'job_family', 'employee_continent',
                            'contract_type_category', 'function_name', 'job_category_level','assignment_country_name']

        var_names = []
        chi2_values = []
        p_values = []

        # Iterate through each categorical variable
        for var in categorical_vars:
            # Calculate a contingency table for the variable and 'turnover'
            contingency_table = pd.crosstab(data[var], data['turnover'])
            # Perform the chi-squared test
            chi2, p_val, dof, expected = chi2_contingency(contingency_table)
            # Append the results to the lists
            var_names.append(var)
            chi2_values.append(chi2)
            p_values.append(p_val)

        # Create a DataFrame for the results
        results_df = pd.DataFrame({
            'Variable': var_names,
            'Chi-Squared Statistic': chi2_values,
            'P-value': p_values
        })

        # Display the results 
        st.markdown("### Chi-Squared Test Results:")
        st.markdown(
            results_df.to_html(index=False, escape=False),
            unsafe_allow_html=True
            )
        # Add spacing
        st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

        t_stat, p_value = stats.ttest_ind(duration_turnover_yes, duration_turnover_no)
        test_results = {
        'Metric': ['T-test', 'p-value'],
        'Value': [t_stat, p_value]}

        # Perform T-tests and create scatter plots
        t_stat_duration, p_value_duration = stats.ttest_ind(duration_turnover_yes, duration_turnover_no)
        t_stat_assignmentcount, p_value_assignmentcount = stats.ttest_ind(assignmentcount_turnover_yes, assignmentcount_turnover_no)

        test_results = {
            'Metric': ['T-test', 'p-value'],
            'Value (Duration)': [t_stat_duration, p_value_duration],
            'Value (Assignment Count)': [t_stat_assignmentcount, p_value_assignmentcount]
        }

        test_results_df = pd.DataFrame(test_results)

        # Add a section for T-test results
        st.markdown("### T-test Results")
        st.table(test_results_df)

        # Add spacing
        st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

        # Create a section for boxplots
        st.markdown("### Scatter plots")

        col1, col2= st.columns(2)

        def create_duration_scatterplot(data_yes, data_no, title):
            fig, ax = plt.subplots()
            ax.scatter(range(len(data_no)), data_no, label='Exit: No', alpha=0.5, color='blue')
            ax.scatter(range(len(data_yes)), data_yes, label='Exit: Yes', alpha=0.5, color='red')
            ax.set_xlabel('Individuals')
            ax.set_ylabel('Assignment Duration')
            ax.set_title(title)
            ax.legend()
            return fig

        def create_assignmentcount_scatterplot(data_yes, data_no, title):
            fig, ax = plt.subplots()
            ax.scatter(range(len(data_no)), data_no, label='Exit: No', alpha=0.5, color='blue')
            ax.scatter(range(len(data_yes)), data_yes, label='Exit: Yes', alpha=0.5, color='red')
            ax.set_xlabel('Individuals')
            ax.set_ylabel('# of Assignments')
            ax.set_title(title)
            ax.legend()
            return fig

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("###### Assignment Duration vs Staff Exit")
            fig1 = create_duration_scatterplot(duration_turnover_yes, duration_turnover_no, 'Assignment Duration')
            st.pyplot(fig1)

        with col2:
            st.markdown("###### Number of Assignments vs Staff Exit")
            fig2 = create_assignmentcount_scatterplot(assignmentcount_turnover_yes, assignmentcount_turnover_no, '# of Assignments')
            st.pyplot(fig2)

    else:
        st.warning("Upload data in the 'Upload Data' tab first.")


if selected_tab == "Modeling":
    
    st.title ("Logistic Regression Model")
    if st.session_state.data is not None:
        data = st.session_state.data

        X = data[['gender', 'age_group', 'job_family', 'employee_continent', 
                            'contract_type_category', 'function_name', 'job_category_level','assignment_country_name','assignment_length_months','assignments_count']]
        y = data['turnover']
        
        X = pd.get_dummies(X, columns=['gender', 'age_group', 'job_family', 'employee_continent', 'assignment_country_name',
                    'contract_type_category', 'function_name', 'job_category_level'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Create a Logistic Regression model
        logistic_model = LogisticRegression()
        # Train the model on the training data
        logistic_model.fit(X_train, y_train)
        # Make predictions on the test data
        y_pred = logistic_model.predict(X_test)
        # Calculate the confusion matrix
        confusion = confusion_matrix(y_test, y_pred)
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)*100
        precision = precision_score(y_test, y_pred)*100
        recall = recall_score(y_test, y_pred)*100
        f1 = f1_score(y_test, y_pred)*100
        # Format percentage values with two decimal places
        accuracy_formatted = f"{accuracy:.2f}%"
        precision_formatted = f"{precision:.2f}%"
        recall_formatted = f"{recall:.2f}%"
        f1_formatted = f"{f1:.2f}%"

        metrics_data = {
        'Metric': ['Accuracy', 'Specificity', 'Sensitivity', 'F1 Score'],
        'Value': [accuracy_formatted, precision_formatted, recall_formatted, f1_formatted]
        }
        metrics_df = pd.DataFrame(metrics_data)
        # Create a layout with two columns
        col1, col2 = st.columns([1,2]   )

        with col1:
            st.markdown("Evaluation Metrics:")
            st.markdown(
                metrics_df.to_html(index=False, escape=False),
                unsafe_allow_html=True
            )
        with col2:
            # Display the confusion matrix as a heatmap
            st.markdown("<div style='text-align: center;'>Confusion Matrix Heatmap</div>", unsafe_allow_html=True)            
            plt.figure(figsize=(4, 2))
            sns.heatmap(confusion, annot=True, cmap="Blues", fmt="d", xticklabels=["Actual Exit", "Actual Retention"], yticklabels=["Predicted Exit", "Predicted Retention"])
            st.pyplot(plt) 
        
        # Get predicted probabilities for the positive class
        y_pred_prob = logistic_model.predict_proba(X_test)[:, 1]

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
       
        # Calculate AUC
        auc = roc_auc_score(y_test, y_pred_prob)
        st.write(f"AUC (Area Under the Curve): {auc:.2f}")

        # Plot ROC curve
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
        ax.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guessing
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Upload data in the 'Upload Data' tab first.")

if selected_tab == "Predict Staff Exit":
    st.title ("MSF Staff Exit Prediction App")    
    if st.session_state.data is not None:
        import pickle
        # Load your trained logistic regression model
        with open('logistic_regression_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        # Function to get user inputs
        def get_user_inputs():
        # Add CSS for styling
            st.markdown(
                """
                <style>
                .input-container {
                    display: flex;
                    flex-wrap: wrap;
                }
                .input-item {
                    margin: 10px;
                }
                .prediction-button {
                    margin-top: 10px;
                }
                .big-title {
                    font-size: 26px;
                    font-weight: bold;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            st.markdown('<div class="input-container">', unsafe_allow_html=True)
            st.subheader("Select Variables:")
            function_names = [
                'Head of Human Resources',
                'Head of Logistics',
                'accountant',
                'admin transit',
                'advocacy manager',
                'anaesthesia nurse/ technician',
                'anaesthetist nurse ',
                'anthropologist',
                'base and facilities officer',
                'base responsible',
                'biomed manager',
                'biomed specialist ',
                'community health worker/ mobilizer',
                'construction manager',
                'counselor educator',
                'cultural mediator',
                'cultural mediator supervisor ',
                'data analyst activity manager',
                'deputy coordinator in charge of logistics',
                'deputy coordinator in charge of supply chain',
                'deputy coordinator in charge of technical logistics',
                'deputy coordinator in charge of watsan',
                'deputy finance coordinator',
                'deputy finance/ hr coordinator',
                'deputy head nurse',
                'deputy head of mission',
                'deputy hr coordinator',
                'deputy medical coordinator',
                'deputy pharmacy coordinator',
                'deputy project coordinator',
                'deputy project medical referent ',
                'deputy supply chain coordinator assistant',
                'doctor anaesthetist',
                'driver',
                'e-health manager',
                'electrician ',
                'electricity manager',
                'entomologist',
                'epidemiologist',
                'epidemiologist op. researcher',
                'epidemiology activity manager',
                'er doctor',
                'field communications manager',
                'field communications officer',
                'field grant reporting manager',
                'field hr support',
                'field legal support',
                'finance and accountancy manager',
                'finance and hr coordinator',
                'finance coordinator',
                'finance coordinator assistant',
                'fleet manager',
                'flight manager',
                'gis activity manager',
                'gis specialist',
                'head nurse',
                'head of human resources ',
                'head of ialo ',
                'head of mission',
                'head of mission advisor ',
                'head of mission assistant',
                'head of mission support',
                'health economist',
                'health promoter iec officer',
                'health promoter supervisor',
                'hospital clinical lead',
                'hospital director',
                'hospital facilities manager',
                'hospital nursing manager',
                'hr coordinator ',
                'hr coordinator assistant',
                'humanitarian affairs officer',
                'ict supervisor',
                'iechp  manager',
                'industrial hygienist ',
                'infection prevention and control ',
                'infection prevention and control  ',
                'information systems specialist',
                'internal controller',
                'intersectional flight coordinator',
                'intersectional legal advisor',
                'intersectional pharmacist',
                'intersectional vaccination focal point',
                'intersectional workshop manager',
                'laboratory manager',
                'laboratory supervisor',
                'lay counselor educator',
                'learning & development specialist',
                'legal assistant',
                'logistics  fin hr manager',
                'logistics coordinator',
                'logistics coordinator ',
                'logistics coordinator assistant',
                'logistics coordinator workshop',
                'logistics manager',
                'logistics specialist',
                'logistics supervisor',
                'logistics team leader',
                'mechanic',
                'mechanic specialist',
                'medical activity manager',
                'medical coordinator ',
                'medical coordinator assistant ',
                'medical coordinator support',
                'medical doctor',
                'medical m&e coordinator ',
                'medical research coordinator',
                'medical research manager',
                'mental health activity manager',
                'mental health supervisor',
                'microbiologist',
                'midwife',
                'midwife activity manager',
                'midwife supervisor',
                'mission  specialized activity manager',
                'mission fin hr manager',
                'mission logistics manager',
                'mission mentalhealth manager ',
                'mission pharmacy manager',
                'mission supply chain manager',
                'mission technical referent',
                'mission technical referent watsan',
                'mobile health activity manager',
                'nurse',
                'nurse aid',
                'nurse mentor ',
                'nurse specialist supervisor',
                'nursing activity manager',
                'nursing team supervisor',
                'nutrition nurse',
                'nutrition supervisor',
                'nutritional activity manager',
                'obstetrician gynaecologist',
                'operational deputy head of mission',
                'operational research activity manager',
                'operational research coordinator ',
                'orthopedic surgeon',
                'ot nurse',
                'other specialist/ supervisor  ',
                'paediatrician',
                'partner-ngo liaison officer ',
                'patient support activity manager',
                'patient support supervisor',
                'personnel administration manager',
                'personnel development manager',
                'pharmacy coordinator',
                'pharmacy supervisor',
                'physiotherapist',
                'physiotherapist supervisor',
                'physiotherapy activity manager',
                'procurement manager',
                'project coordinator',
                'project fin/ hr manager',
                'project finance manager',
                'project hr manager',
                'project medical referent',
                'project medical responsible ',
                'project operations responsible ',
                'project pharmacy manager',
                'project supply chain manager',
                'psychiatrist',
                'psychologist',
                'public health activity manager ',
                'qualified translator',
                'qualitative researcher ',
                'radiologist',
                'reasearcher m&e ',
                'record management officer',
                'referent ncd ',
                'regional advocacy representative',
                'regional communications coordinator',
                'regional finance support ',
                'regional hr support ',
                'regional logistics support ',
                'regional medical referent',
                'regional medical responsible ',
                'regional operational responsible ',
                'regional technical referent',
                'remuneration analyst',
                'resources manager',
                'sar boat pilot ',
                'secretary',
                'security advisor',
                'sexual violence program activity manager',
                'social worker',
                'specialized medical doctor',
                'specialized nurse',
                'specialized technician ',
                'supply activity manager',
                'supply chain coordinator',
                'supply chain officer',
                'supply chain supervisor',
                'supply chain team leader',
                'supply fin hr manager ',
                'supply fin manager',
                'surgeon',
                'surgical activity manager',
                'technical activity manager',
                'training facilitator medical',
                'training officer',
                'training supervisor',
                'training supervisor medical',
                'transport and customs manager',
                'warehouse manager',
                'watsan coordinator',
                'watsan manager',
                'watsan supervisor',
                'watsan team leader',
                'workshop manager',
                'workshop supervisor',
                'x-ray supervisor',
            ]
            assignment_countries = [
                'Afghanistan',
                'American Samoa',
                'Angola',
                'Argentina',
                'Armenia',
                'Australia',
                'Austria',
                'Azerbaijan',
                'Bangladesh',
                'Belgium',
                'Benin',
                'Bolivia',
                'Brazil',
                'Burkina Faso',
                'Burundi',
                'Cambodia',
                'Cameroon',
                'Canada',
                'Central African Republic',
                'Chad',
                'China',
                'Colombia',
                'Congo',
                'Costa Rica',
                'Czech Republic',
                'Democratic Republic of  Congo',
                'Denmark',
                'Dominica',
                'Ecuador',
                'Egypt',
                'El Salvador',
                'Eritrea',
                'Eswatini',
                'Ethiopia',
                'France',
                'Gabon',
                'Georgia',
                'Germany',
                'Greece',
                'Guatemala',
                'Guinea',
                'Guinea-Bissau',
                'Haiti',
                'Honduras',
                'Hong Kong',
                'Hungary',
                'India',
                'Indonesia',
                'Iran',
                'Iraq',
                'Ireland',
                'Italy',
                'Ivory Coast',
                'Japan',
                'Jordan',
                'Kenya',
                'Kiribati',
                'Kyrgyzstan',
                'Lebanon',
                'Liberia',
                'Libya',
                'Lithuania',
                'Luxembourg',
                'Madagascar',
                'Malawi',
                'Malaysia',
                'Mali',
                'Mauritania',
                'Mediterranean sea rescue',
                'Mexico',
                'Moldova',
                'Montenegro',
                'Mozambique',
                'Myanmar',
                'Nauru',
                'Nepal',
                'Nicaragua',
                'Niger',
                'Nigeria',
                'North Korea',
                'Norway',
                'Pakistan',
                'Palestinian Territory',
                'Papua New Guinea',
                'Peru',
                'Philippines',
                'Poland',
                'Portugal',
                'Russia',
                'Rwanda',
                'Senegal',
                'Serbia',
                'Sierra Leone',
                'Slovakia',
                'Slovenia',
                'Somalia',
                'South Africa',
                'South Korea',
                'South Sudan',
                'Spain',
                'Sudan',
                'Suriname',
                'Sweden',
                'Switzerland',
                'Syria Arab Republic',
                'Tajikistan',
                'Tanzania',
                'Thailand',
                'The Netherlands',
                'Togo',
                'Tunisia',
                'Turkey',
                'Uganda',
                'Ukraine',
                'United Arab Emirates',
                'United Kingdom',
                'United States of America',
                'Unknown',
                'Uzbekistan',
                'Venezuela (Bolivarian Republic)',
                'Yemen',
                'Zimbabwe',

            ]
            # Create two columns for organizing variables
            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="input-column">', unsafe_allow_html=True)
                gender = 'gender_' + st.selectbox('Gender:', ['Male', 'Female'])
                age_group = 'age_group_' + st.selectbox('Age Group:', ['16-24', '25-34', '35-44', '45-54', '55-64', '65-85'])
                employee_continent = 'employee_continent_' + st.selectbox('Employee Continent:', ['Africa', 'Australia & Oceania', 'Asia', 'Europe', 'North America', 'South America'])
                function_name = 'function_name_' + st.selectbox('Function Name:', function_names)
                job_family = 'job_family_' + st.selectbox('Job Family:', ['Medical & Paramedical', 'Logistics & Supply', 'HR & FIN', 'Operations'])
                job_category_level = 'job_category_level_' + st.selectbox('Job Level:', ['Basic Skilled Positions', 'Skilled Positions', 'Supervisors & Specialists', 'Activity Managers', 'Coordinators'])
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="input-column">', unsafe_allow_html=True)
                assignment_country= 'assignment_country_name_' + st.selectbox('Assignment Country:', assignment_countries)
                contract_type_category = 'contract_type_category_' + st.selectbox('Contract Type:', ['Intermissioner', 'LTA12', 'LTA24', 'Vocationer', 'Emergency team', 'TANDEM', 'MALTA', 'Other special contracts'])
                assignments_count = st.number_input("Number of Assignments")
                assignment_length_months = st.number_input("Assignment Length (in months)")
                st.markdown('</div>', unsafe_allow_html=True)
                
            user_input_gender = pd.DataFrame({
                'gender_Female': 0,
                'gender_Male': 0,
                'gender_Unknown': 0
                }, index=[0])
            user_input_age_group = pd.DataFrame({
                'age_group_16-24': 0,
                'age_group_25-34': 0,
                'age_group_35-44': 0,
                'age_group_45-54': 0,
                'age_group_55-64': 0,
                'age_group_65-85': 0,
                'age_group_unknown/outliers': 0
            }, index=[0])
            user_input_job_family = pd.DataFrame({
                'job_family_HR & FIN': 0,
                'job_family_Logistics & Supply': 0,
                'job_family_Medical & Paramedical': 0,
                'job_family_Operations': 0,
                'job_family_Unknown': 0
            }, index=[0])    
            user_input_employee_continent = pd.DataFrame({
                'employee_continent_Africa': 0,
                'employee_continent_Asia': 0,
                'employee_continent_Australia & Oceania': 0,
                'employee_continent_Europe': 0,
                'employee_continent_North America': 0,
                'employee_continent_South America': 0,
                'employee_continent_Unknown': 0
            }, index=[0])    
            user_input_assignment_country = pd.DataFrame({
                'assignment_country_name_Afghanistan':0,
                'assignment_country_name_American Samoa':0,
                'assignment_country_name_Angola':0,
                'assignment_country_name_Argentina':0,
                'assignment_country_name_Armenia':0,
                'assignment_country_name_Australia':0,
                'assignment_country_name_Austria':0,
                'assignment_country_name_Azerbaijan':0,
                'assignment_country_name_Bangladesh':0,
                'assignment_country_name_Belgium':0,
                'assignment_country_name_Benin':0,
                'assignment_country_name_Bolivia':0,
                'assignment_country_name_Brazil':0,
                'assignment_country_name_Burkina Faso':0,
                'assignment_country_name_Burundi':0,
                'assignment_country_name_Cambodia':0,
                'assignment_country_name_Cameroon':0,
                'assignment_country_name_Canada':0,
                'assignment_country_name_Central African Republic':0,
                'assignment_country_name_Chad':0,
                'assignment_country_name_China':0,
                'assignment_country_name_Colombia':0,
                'assignment_country_name_Congo':0,
                'assignment_country_name_Costa Rica':0,
                'assignment_country_name_Czech Republic':0,
                'assignment_country_name_Democratic Republic of  Congo':0,
                'assignment_country_name_Denmark':0,
                'assignment_country_name_Dominica':0,
                'assignment_country_name_Ecuador':0,
                'assignment_country_name_Egypt':0,
                'assignment_country_name_El Salvador':0,
                'assignment_country_name_Eritrea':0,
                'assignment_country_name_Eswatini':0,
                'assignment_country_name_Ethiopia':0,
                'assignment_country_name_France':0,
                'assignment_country_name_Gabon':0,
                'assignment_country_name_Georgia':0,
                'assignment_country_name_Germany':0,
                'assignment_country_name_Greece':0,
                'assignment_country_name_Guatemala':0,
                'assignment_country_name_Guinea':0,
                'assignment_country_name_Guinea-Bissau':0,
                'assignment_country_name_Haiti':0,
                'assignment_country_name_Honduras':0,
                'assignment_country_name_Hong Kong':0,
                'assignment_country_name_Hungary':0,
                'assignment_country_name_India':0,
                'assignment_country_name_Indonesia':0,
                'assignment_country_name_Iran':0,
                'assignment_country_name_Iraq':0,
                'assignment_country_name_Ireland':0,
                'assignment_country_name_Italy':0,
                'assignment_country_name_Ivory Coast':0,
                'assignment_country_name_Japan':0,
                'assignment_country_name_Jordan':0,
                'assignment_country_name_Kenya':0,
                'assignment_country_name_Kiribati':0,
                'assignment_country_name_Kyrgyzstan':0,
                'assignment_country_name_Lebanon':0,
                'assignment_country_name_Liberia':0,
                'assignment_country_name_Libya':0,
                'assignment_country_name_Lithuania':0,
                'assignment_country_name_Luxembourg':0,
                'assignment_country_name_Madagascar':0,
                'assignment_country_name_Malawi':0,
                'assignment_country_name_Malaysia':0,
                'assignment_country_name_Mali':0,
                'assignment_country_name_Mauritania':0,
                'assignment_country_name_Mediterranean sea rescue':0,
                'assignment_country_name_Mexico':0,
                'assignment_country_name_Moldova':0,
                'assignment_country_name_Montenegro':0,
                'assignment_country_name_Mozambique':0,
                'assignment_country_name_Myanmar':0,
                'assignment_country_name_Nauru':0,
                'assignment_country_name_Nepal':0,
                'assignment_country_name_Nicaragua':0,
                'assignment_country_name_Niger':0,
                'assignment_country_name_Nigeria':0,
                'assignment_country_name_North Korea':0,
                'assignment_country_name_Norway':0,
                'assignment_country_name_Pakistan':0,
                'assignment_country_name_Palestinian Territory':0,
                'assignment_country_name_Papua New Guinea':0,
                'assignment_country_name_Peru':0,
                'assignment_country_name_Philippines':0,
                'assignment_country_name_Poland':0,
                'assignment_country_name_Portugal':0,
                'assignment_country_name_Russia':0,
                'assignment_country_name_Rwanda':0,
                'assignment_country_name_Senegal':0,
                'assignment_country_name_Serbia':0,
                'assignment_country_name_Sierra Leone':0,
                'assignment_country_name_Slovakia':0,
                'assignment_country_name_Slovenia':0,
                'assignment_country_name_Somalia':0,
                'assignment_country_name_South Africa':0,
                'assignment_country_name_South Korea':0,
                'assignment_country_name_South Sudan':0,
                'assignment_country_name_Spain':0,
                'assignment_country_name_Sudan':0,
                'assignment_country_name_Suriname':0,
                'assignment_country_name_Sweden':0,
                'assignment_country_name_Switzerland':0,
                'assignment_country_name_Syria Arab Republic':0,
                'assignment_country_name_Tajikistan':0,
                'assignment_country_name_Tanzania':0,
                'assignment_country_name_Thailand':0,
                'assignment_country_name_The Netherlands':0,
                'assignment_country_name_Togo':0,
                'assignment_country_name_Tunisia':0,
                'assignment_country_name_Turkey':0,
                'assignment_country_name_Uganda':0,
                'assignment_country_name_Ukraine':0,
                'assignment_country_name_United Arab Emirates':0,
                'assignment_country_name_United Kingdom':0,
                'assignment_country_name_United States of America':0,
                'assignment_country_name_Unknown':0,
                'assignment_country_name_Uzbekistan':0,
                'assignment_country_name_Venezuela (Bolivarian Republic)':0,
                'assignment_country_name_Yemen':0,
                'assignment_country_name_Zimbabwe':0,

            }, index=[0])    
            user_input_contract_type_category = pd.DataFrame({
                'contract_type_category_Emergency team': 0,
                'contract_type_category_Intermissioner': 0,
                'contract_type_category_LTA12': 0,
                'contract_type_category_LTA24': 0,
                'contract_type_category_MALTA': 0,
                'contract_type_category_Other special contracts': 0,
                'contract_type_category_TANDEM': 0,
                'contract_type_category_Unknown': 0,
                'contract_type_category_Vocationer': 0      
            }, index=[0])    
            user_input_function_name = pd.DataFrame({
                'function_name_Head of Human Resources': 0,
                'function_name_Head of Logistics': 0,
                'function_name_Unknown': 0,
                'function_name_accountant': 0,
                'function_name_admin transit': 0,
                'function_name_advocacy manager': 0,
                'function_name_anaesthesia nurse/ technician': 0,
                'function_name_anaesthetist nurse ': 0,
                'function_name_anthropologist': 0,
                'function_name_base and facilities officer': 0,
                'function_name_base responsible': 0,
                'function_name_biomed manager': 0,
                'function_name_biomed specialist ': 0,
                'function_name_community health worker/ mobilizer': 0,
                'function_name_construction manager': 0,
                'function_name_counselor educator': 0,
                'function_name_cultural mediator': 0,
                'function_name_cultural mediator supervisor ': 0,
                'function_name_data analyst activity manager': 0,
                'function_name_deputy coordinator in charge of logistics': 0,
                'function_name_deputy coordinator in charge of supply chain': 0,
                'function_name_deputy coordinator in charge of technical logistics': 0,
                'function_name_deputy coordinator in charge of watsan': 0,
                'function_name_deputy finance coordinator': 0,
                'function_name_deputy finance/ hr coordinator': 0,
                'function_name_deputy head nurse': 0,
                'function_name_deputy head of mission': 0,
                'function_name_deputy hr coordinator': 0,
                'function_name_deputy medical coordinator': 0,
                'function_name_deputy pharmacy coordinator': 0,
                'function_name_deputy project coordinator': 0,
                'function_name_deputy project medical referent ': 0,
                'function_name_deputy supply chain coordinator assistant': 0,
                'function_name_doctor anaesthetist': 0,
                'function_name_driver': 0,
                'function_name_e-health manager': 0,
                'function_name_electrician ': 0,
                'function_name_electricity manager': 0,
                'function_name_entomologist': 0,
                'function_name_epidemiologist': 0,
                'function_name_epidemiologist op. researcher': 0,
                'function_name_epidemiology activity manager': 0,
                'function_name_er doctor': 0,
                'function_name_field communications manager': 0,
                'function_name_field communications officer': 0,
                'function_name_field grant reporting manager': 0,
                'function_name_field hr support': 0,
                'function_name_field legal support': 0,
                'function_name_finance and accountancy manager': 0,
                'function_name_finance and hr coordinator': 0,
                'function_name_finance coordinator': 0,
                'function_name_finance coordinator assistant': 0,
                'function_name_fleet manager': 0,
                'function_name_flight manager': 0,
                'function_name_gis activity manager': 0,
                'function_name_gis specialist': 0,
                'function_name_head nurse': 0,
                'function_name_head of human resources ': 0,
                'function_name_head of ialo ': 0,
                'function_name_head of mission': 0,
                'function_name_head of mission advisor ': 0,
                'function_name_head of mission assistant': 0,
                'function_name_head of mission support': 0,
                'function_name_health economist': 0,
                'function_name_health promoter iec officer': 0,
                'function_name_health promoter supervisor': 0,
                'function_name_hospital clinical lead': 0,
                'function_name_hospital director': 0,
                'function_name_hospital facilities manager': 0,
                'function_name_hospital nursing manager': 0,
                'function_name_hr coordinator ': 0,
                'function_name_hr coordinator assistant': 0,
                'function_name_humanitarian affairs officer': 0,
                'function_name_ict supervisor': 0,
                'function_name_iechp  manager': 0,
                'function_name_industrial hygienist ': 0,
                'function_name_infection prevention and control ': 0,
                'function_name_infection prevention and control  ': 0,
                'function_name_information systems specialist': 0,
                'function_name_internal controller': 0,
                'function_name_intersectional flight coordinator': 0,
                'function_name_intersectional legal advisor': 0,
                'function_name_intersectional pharmacist': 0,
                'function_name_intersectional vaccination focal point': 0,
                'function_name_intersectional workshop manager': 0,
                'function_name_laboratory manager': 0,
                'function_name_laboratory supervisor': 0,
                'function_name_lay counselor educator': 0,
                'function_name_learning & development specialist': 0,
                'function_name_legal assistant': 0,
                'function_name_logistics  fin hr manager': 0,
                'function_name_logistics coordinator': 0,
                'function_name_logistics coordinator ': 0,
                'function_name_logistics coordinator assistant': 0,
                'function_name_logistics coordinator workshop': 0,
                'function_name_logistics manager': 0,
                'function_name_logistics specialist': 0,
                'function_name_logistics supervisor': 0,
                'function_name_logistics team leader': 0,
                'function_name_mechanic': 0,
                'function_name_mechanic specialist': 0,
                'function_name_medical activity manager': 0,
                'function_name_medical coordinator ': 0,
                'function_name_medical coordinator assistant ': 0,
                'function_name_medical coordinator support': 0,
                'function_name_medical doctor': 0,
                'function_name_medical m&e coordinator ': 0,
                'function_name_medical research coordinator': 0,
                'function_name_medical research manager': 0,
                'function_name_mental health activity manager': 0,
                'function_name_mental health supervisor': 0,
                'function_name_microbiologist': 0,
                'function_name_midwife': 0,
                'function_name_midwife activity manager': 0,
                'function_name_midwife supervisor': 0,
                'function_name_mission  specialized activity manager': 0,
                'function_name_mission fin hr manager': 0,
                'function_name_mission logistics manager': 0,
                'function_name_mission mentalhealth manager ': 0,
                'function_name_mission pharmacy manager': 0,
                'function_name_mission supply chain manager': 0,
                'function_name_mission technical referent': 0,
                'function_name_mission technical referent watsan': 0,
                'function_name_mobile health activity manager': 0,
                'function_name_nurse': 0,
                'function_name_nurse aid': 0,
                'function_name_nurse mentor ': 0,
                'function_name_nurse specialist supervisor': 0,
                'function_name_nursing activity manager': 0,
                'function_name_nursing team supervisor': 0,
                'function_name_nutrition nurse': 0,
                'function_name_nutrition supervisor': 0,
                'function_name_nutritional activity manager': 0,
                'function_name_obstetrician gynaecologist': 0,
                'function_name_operational deputy head of mission': 0,
                'function_name_operational research activity manager': 0,
                'function_name_operational research coordinator ': 0,
                'function_name_orthopedic surgeon': 0,
                'function_name_ot nurse': 0,
                'function_name_other specialist/ supervisor  ': 0,
                'function_name_paediatrician': 0,
                'function_name_partner-ngo liaison officer ': 0,
                'function_name_patient support activity manager': 0,
                'function_name_patient support supervisor': 0,
                'function_name_personnel administration manager': 0,
                'function_name_personnel development manager': 0,
                'function_name_pharmacy coordinator': 0,
                'function_name_pharmacy supervisor': 0,
                'function_name_physiotherapist': 0,
                'function_name_physiotherapist supervisor': 0,
                'function_name_physiotherapy activity manager': 0,
                'function_name_procurement manager': 0,
                'function_name_project coordinator': 0,
                'function_name_project fin/ hr manager': 0,
                'function_name_project finance manager': 0,
                'function_name_project hr manager': 0,
                'function_name_project medical referent': 0,
                'function_name_project medical responsible ': 0,
                'function_name_project operations responsible ': 0,
                'function_name_project pharmacy manager': 0,
                'function_name_project supply chain manager': 0,
                'function_name_psychiatrist': 0,
                'function_name_psychologist': 0,
                'function_name_public health activity manager ': 0,
                'function_name_qualified translator': 0,
                'function_name_qualitative researcher ': 0,
                'function_name_radiologist': 0,
                'function_name_reasearcher m&e ': 0,
                'function_name_record management officer': 0,
                'function_name_referent ncd ': 0,
                'function_name_regional advocacy representative': 0,
                'function_name_regional communications coordinator': 0,
                'function_name_regional finance support ': 0,
                'function_name_regional hr support ': 0,
                'function_name_regional logistics support ': 0,
                'function_name_regional medical referent': 0,
                'function_name_regional medical responsible ': 0,
                'function_name_regional operational responsible ': 0,
                'function_name_regional technical referent': 0,
                'function_name_remuneration analyst': 0,
                'function_name_resources manager': 0,
                'function_name_sar boat pilot ': 0,
                'function_name_secretary': 0,
                'function_name_security advisor': 0,
                'function_name_sexual violence program activity manager': 0,
                'function_name_social worker': 0,
                'function_name_specialized medical doctor': 0,
                'function_name_specialized nurse': 0,
                'function_name_specialized technician ': 0,
                'function_name_supply activity manager': 0,
                'function_name_supply chain coordinator': 0,
                'function_name_supply chain officer': 0,
                'function_name_supply chain supervisor': 0,
                'function_name_supply chain team leader': 0,
                'function_name_supply fin hr manager ': 0,
                'function_name_supply fin manager': 0,
                'function_name_surgeon': 0,
                'function_name_surgical activity manager': 0,
                'function_name_technical activity manager': 0,
                'function_name_training facilitator medical': 0,
                'function_name_training officer': 0,
                'function_name_training supervisor': 0,
                'function_name_training supervisor medical': 0,
                'function_name_transport and customs manager': 0,
                'function_name_warehouse manager': 0,
                'function_name_watsan coordinator': 0,
                'function_name_watsan manager': 0,
                'function_name_watsan supervisor': 0,
                'function_name_watsan team leader': 0,
                'function_name_workshop manager': 0,
                'function_name_workshop supervisor': 0,
                'function_name_x-ray supervisor': 0,
                }, index=[0])

            user_input_job_category_level = pd.DataFrame({
                'job_category_level_Activity Managers': 0,
                'job_category_level_Basic Skilled Positions': 0,   
                'job_category_level_Coordinators': 0,
                'job_category_level_Skilled Positions': 0,
                'job_category_level_Supervisors & Specialists': 0,
                'job_category_level_Unknown': 0
            }, index=[0])
            
            # Create user input DataFrames based on user selections
            user_input_gender[gender] = [1]  # Set the selected value to 1
            user_input_age_group[age_group] = [1]  # Set the selected value to 1
            user_input_job_family[job_family] = [1]
            user_input_employee_continent[employee_continent] = [1]
            user_input_contract_type_category[contract_type_category] = [1]
            user_input_function_name[function_name] = [1]
            user_input_job_category_level[job_category_level] = [1]
            user_input_assignment_country[assignment_country] = [1]
            # Create a DataFrame for numerical inputs
            user_input_numerical = pd.DataFrame({
                'assignment_length_months': [assignment_length_months],
                'assignments_count': [assignments_count]
                }, index=[0])
            # Concatenate user input DataFrames (categorical and numerical)

            user_input = pd.concat([user_input_numerical, user_input_gender, user_input_age_group, user_input_job_family,
                                    user_input_employee_continent, user_input_contract_type_category, user_input_function_name,
                                    user_input_job_category_level, user_input_assignment_country], axis=1)    
            
            return user_input

        # Get user inputs
        user_input  = get_user_inputs()
        # Create two columns
        col1, col2 = st.columns(2)
        # Initialize prediction and image_path variables
        prediction = ""
        image_path = ""
        probability_leave = 0.0
        # Check if Function Name is empty
        if col1.button("Predict"):
            probability_leave = model.predict_proba(user_input)[0, 1]  # Probability of leaving

            # Define a threshold to decide whether staff will leave or not
            threshold = 0.5

            # Determine the prediction (leave or stay) and corresponding image
            if probability_leave >= threshold:
                prediction = "The Staff Member will leave"
                image_path = "sad_face.png"
            else:
                prediction = "The Staff Member will stay"
                image_path = "happy_face.png"
        # Display the prediction and probability using HTML formatting
        col2.markdown(f"<b>Prediction:</b> {prediction}", unsafe_allow_html=True)
        col2.markdown(f"<b>Probability of leaving:</b> {probability_leave:.2%}", unsafe_allow_html=True)

        # Display the image
        if image_path:
            col2.image(image_path, caption="")
    else:
        st.warning("Upload data in the 'Upload Data' tab first.")
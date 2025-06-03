import streamlit as st
import time
import os
import io
import joblib
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split


st.title("🤖ML Model Generator🤖")

# Sidebar Section
with st.sidebar:
    # Data Source Section
    st.header("📰Data Source📰")
    data_source = st.radio("Choose data source:", ["Generate Synthetic Data", "Upload Dataset"])

    if data_source == "Generate Synthetic Data":
        st.subheader("Synthetic Data Generation")
        st.write("Define parameters for synthetic data generation below.")
        
        # Data Generation Parameters Section
        st.subheader("Data Generation Parameters")
        
        # Input for feature names
        features_input = st.text_input("Enter feature names (comma-separated)", "length (mm), width (mm), density (g/cm³)")
        features = [f.strip() for f in features_input.split(",")]

        
        # Input for class names
        classes_input = st.text_input("Enter class names (comma-separated)", "Ampalaya, Banana, Cabbage")
        classes = [c.strip() for c in classes_input.split(",")]

        # Class-Specific Settings
        st.subheader("Class-Specific Settings")
        class_data = [] 

        # Store mean and std values for each class
        mean_values_dict = {}
        std_values_dict = {}

        for class_name in classes:
            with st.expander(f"{class_name} Settings", expanded=False):
                st.checkbox(f"Set specific values for {class_name}", value=True)

                col1, col2 = st.columns(2)
                with col1:
                    mean_values_dict[class_name] = [
                        int(st.number_input(f"Mean for {feature}, ({class_name})", value=100.0, min_value=0.0)) for feature in features
                    ]
                with col2:
                    std_values_dict[class_name] = [
                        int(st.number_input(f"Std Dev for {feature}, ({class_name})", value=10.0, min_value=0.1)) for feature in features
                    ]

        st.subheader("Sample Size & Train/Test Split Configuration")   
        
        col1, col2 = st.columns(2)
        with col1:
            total_sample_size = st.slider(
                "Number of samples", 
                max_value = 50000, 
                min_value=500,
                step=500
            )
        with col2:
            train_test_split_percent = st.slider(
                "Train-Test Split (%)",
                min_value=10,
                max_value=50,
                step=5
            )

        samples_per_class = total_sample_size // len(classes)
        remainder = total_sample_size % len(classes)
        # Generate synthetic data for each class
        class_data = []
        for i, class_name in enumerate(classes):
        # Add 1 extra sample to some classes to account for the remainder
            extra_sample = 1 if i < remainder else 0
            num_samples = samples_per_class + extra_sample
            
            # Generate the synthetic data
            mean_values = mean_values_dict[class_name]
            std_values = std_values_dict[class_name]
            data = np.random.normal(
                loc=mean_values,
                scale=std_values,
                size=(num_samples, len(features))
            )
            class_labels = np.full((num_samples, 1), class_name)  # Class label column
            class_data.append(np.hstack([data, class_labels])) 
        
    elif data_source == "Upload Dataset":
        st.subheader("📤Upload Dataset")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            class_df = pd.read_csv(uploaded_file)
            st.write("Uploaded Dataset:")
            st.dataframe(class_df)

            features = class_df.columns[:-1]  # Assuming the last column is the target
            target_column = class_df.columns[-1]  # Assuming the last column is the target
            classes = class_df[target_column].unique()  # Assuming the last column is the target

            st.subheader("Train/Test Split Configuration")
            col1, col2 = st.columns(2)
            with col1:
                train_test_split_percent = st.slider(
                    "Train-Test Split (%)",
                    min_value=10,
                    max_value=50,
                    step=5
                )
    generate_data_button = st.button("Generate Data and Train Model")
            
# Ensure class_df is defined before using it
if generate_data_button or 'generated' not in st.session_state:
    try:
        if data_source == "Generate Synthetic Data":
            all_data = np.vstack(class_data)
            np.random.shuffle(all_data)

            # Split data into train and test
            train_size = train_test_split_percent / 100

            # Extract feature data and labels
            feature_data = all_data[:, :-1].astype(float)
            labels = all_data[:, -1]

            # Create DataFrame for class data
            class_df = pd.DataFrame(feature_data, columns=features)
            class_df['Target'] = labels
            
        if data_source == "Upload Dataset" and uploaded_file is not None:
            try:
                # Read the uploaded file as raw text to check its contents
                raw_file_content = uploaded_file.getvalue().decode("utf-8")
                
                if raw_file_content.strip() == "":
                    st.error("The uploaded file is empty. Please upload a valid CSV file.")
                else:
                    # Attempt to parse the CSV into a DataFrame
                    class_df = pd.read_csv(io.StringIO(raw_file_content))
                    
                    if class_df.empty:
                        st.error("The dataset is empty. Please upload a valid CSV file.")
                    else:
                        st.sidebar.write("Dataset Information:")
                        st.sidebar.write(f"Shape: {class_df.shape}")
                        st.sidebar.write(f"Columns: {list(class_df.columns)}")

                        st.write("Uploaded Dataset:")
                        st.dataframe(class_df)

                        # Clean column names
                        class_df.columns = class_df.columns.str.strip()

                        # Validate 'Target' column
                        if 'Target' not in class_df.columns:
                            st.error("The dataset must include a 'Target' column.")
                        elif class_df['Target'].isnull().all():
                            st.error("The 'Target' column contains only null values.")
                        else:
                            features = class_df.columns[:-1].tolist()  # Exclude the 'Target' column
                            labels = class_df['Target']

                            # Check for numeric features
                            for col in features:
                                class_df[col] = pd.to_numeric(class_df[col], errors='coerce')

                            if class_df[features].isnull().values.any():
                                st.warning("The dataset contains non-numeric or missing feature values.")
                            else:
                                st.success("Dataset validated and ready for processing!")
            except Exception as e:
                st.error(f"Error loading dataset: {e}")



        # Ensure class_df is defined before using it
        if 'class_df' in locals():
            total_samples = len(class_df)
            train_samples = int((train_test_split_percent / 100) * total_samples)
            test_samples = total_samples - train_samples

            st.subheader("Dataset Split Information")
            col1, col2, col3 = st.columns(3)

            # Total Samples Column
            with col1:
                st.markdown("Total Samples")
                st.subheader(total_samples)

            # Training Samples Column
            with col2:
                st.markdown("Training Samples")
                st.subheader(f"{train_samples} ({train_test_split_percent}%)")
            
            # Testing Samples Column
            with col3:
                st.markdown("Testing Samples") 
                st.subheader(f"{test_samples} ({100 - train_test_split_percent}%)")

            # Scale the feature data using MinMaxScaler
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(class_df[features].apply(pd.to_numeric, errors='coerce'))  # Scale only the feature columns
            scaled_df = pd.DataFrame(scaled_data, columns=features)
            scaled_df['Target'] = labels  # Keep the 'Target' column with class labels

            st.subheader("Generated Data Sample")

            # Define the column widths
            col1, col2 = st.columns([4, 4])

            with col1:
                st.write("Original Data (Random samples from each class):")
                # Adjust the dataframe width to fit the container width and take up more space
                st.dataframe(class_df, use_container_width=True)

            with col2:
                st.write("Scaled Data (using best model's scaler):")
                # Adjust the dataframe width to fit the container width and take up more space
                st.dataframe(scaled_df, use_container_width=True)

            # Feature Visualization
            st.subheader("Feature Visualization")
            features = class_df.columns[:-1]  # Exclude 'Target' for plotting

            # Convert all features to numeric, coercing errors
            for feature in features:
                class_df[feature] = pd.to_numeric(class_df[feature], errors='coerce')

            # List of unique class labels
            classes = class_df['Target'].unique()

            # Select visualization type
            visualization_type = st.radio("Select Visualization Type", ["2D", "3D"])

            if visualization_type == "2D":
                # Dropdowns to select features for X and Y axes
                col1, col2 = st.columns(2)
                with col1:
                    x_feature = st.selectbox("Select X-Axis Feature", features)
                    
                with col2:    
                    y_feature = st.selectbox("Select Y-Axis Feature", features)

                # Check if selected features are numeric, or convert if necessary
                if pd.to_numeric(class_df[x_feature], errors='coerce').isnull().any() or pd.to_numeric(class_df[y_feature], errors='coerce').isnull().any():
                    st.error("Selected features should be numeric for scatter plot.")
                else:
                    # Create the 2D scatter plot using Plotly Express
                    fig = px.scatter(
                        class_df, 
                        x=x_feature, 
                        y=y_feature, 
                        color='Target',  # Color points by class
                        title=f"Scatter Plot of {x_feature} vs {y_feature}",
                        labels={x_feature: x_feature, y_feature: y_feature}
                    )
                    # Display the plot in the Streamlit app
                    st.plotly_chart(fig, use_container_width=True)

            elif visualization_type == "3D":
                # Dropdowns to select features for X, Y, and Z axes
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_feature = st.selectbox("Select X-Axis Feature", features, key="x_3d")
                with col2:
                    y_feature = st.selectbox("Select Y-Axis Feature", features, key="y_3d")
                with col3:
                    z_feature = st.selectbox("Select Z-Axis Feature", features, key="z_3d")

                # Ensure the selected features are numeric before plotting
                if (
                    pd.to_numeric(class_df[x_feature], errors='coerce').isnull().any() or
                    pd.to_numeric(class_df[y_feature], errors='coerce').isnull().any() or
                    pd.to_numeric(class_df[z_feature], errors='coerce').isnull().any()
                ):
                    st.error("Selected features should be numeric for 3D scatter plot.")
                else:
                    # Create the 3D scatter plot using Plotly Express
                    fig = px.scatter_3d(
                        class_df,
                        x=x_feature,
                        y=y_feature,
                        z=z_feature,
                        color='Target',  # Color points by class
                        title=f"3D Scatter Plot of {x_feature}, {y_feature}, {z_feature}",
                        labels={x_feature: x_feature, y_feature: y_feature, z_feature: z_feature}
                    )
                    # Display the plot in the Streamlit app
                    st.plotly_chart(fig, use_container_width=True)

            # Download Dataset
            st.subheader("Download Dataset")

            # Function to convert DataFrame to CSV
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            # Generate CSV files for original and scaled datasets
            original_csv = convert_df_to_csv(class_df)
            scaled_csv = convert_df_to_csv(scaled_df)

            # Create download buttons
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download Original Dataset (CSV)",
                    data=original_csv,
                    file_name="original_dataset.csv",
                    mime="text/csv"
                )
            with col2:
                st.download_button(
                    label="Download Scaled Dataset (CSV)",
                    data=scaled_csv,
                    file_name="scaled_dataset.csv",
                    mime="text/csv"
                )


            with st.expander("Dataset Statistics"):
                st.subheader("Dataset Statistics Overview")

                col1, col2 = st.columns(2)

                    # Display statistics for the Original dataset in Column 1 and 2
                with col1:
                    st.write("**Original Dataset**")
                    st.dataframe(class_df.describe())
                with col2:
                    st.write("**Scaled Dataset**")
                    st.dataframe(scaled_df.describe())
                        

            X = class_df.drop(columns=["Target"])
            y = class_df["Target"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - train_test_split_percent) / 100, random_state=42)

                # Models
            models = {
            "Gaussian Naive Bayes": GaussianNB(),
            "AdaBoost Classifier": AdaBoostClassifier(algorithm='SAMME'),
            "Random Forest Classifier": RandomForestClassifier(),
            "Support Vector Classification": SVC(),
            "Multi-layer Perceptron": MLPClassifier(max_iter=500),
            "Extra Trees Classifier": ExtraTreesClassifier(),
                }

            best_model = None
            best_score = 0
            model_results = {}

            for model_name, model in models.items():
                start_time = time.time()  
                status = "Failed" 
                try:
                    model.fit(X_train, y_train) 
                    end_time = time.time() 

                    training_time = end_time - start_time  # Calculate training time

                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')

                    status = "Success"

                except Exception as e:
                        # If there is an error during training, print the error and leave status as "Failed"
                    st.error(f"Error in training {model_name}: {e}")
                    training_time = None

                    # Store model performance and training time
                model_results[model_name] = {
                    "Accuracy": accuracy if status == "Success" else None,
                    "Precision": precision if status == "Success" else None,
                    "Recall": recall if status == "Success" else None,
                    "F1-Score": f1 if status == "Success" else None,
                    "Training Time (s)": round(training_time, 4) if status == "Success" else None,
                    "Status": status
                    }

                    # Update the best model based on accuracy if the model was successful
                if status == "Success" and accuracy > best_score:
                    best_score = accuracy
                    best_model = model

                # If best model is selected, calculate its performance
            if best_model:
                best_model.fit(X_train, y_train)
                y_pred = best_model.predict(X_test)  

                    # Generate classification report
                report = classification_report(y_test, y_pred, output_dict=True)

                    # Show best model performance
                st.subheader("Best Model Performance")
                st.write(f"Best Model: {best_model.__class__.__name__}")
                st.write(f"Accuracy: {best_score:.4f}")

                    # Display classification report
                st.write("Classification Report (Best Model):")
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)

                # Show model comparison
            st.subheader("Model Comparison")
            model_comparison_df = pd.DataFrame(model_results).T
            st.dataframe(model_comparison_df)

            st.subheader("Performance Metrics Summary")
            selected_models = st.multiselect(
            "Select Models for Comparison", 
                options=list(models.keys()), 
                default=list(models.keys())  
                )

                # Display the performance metrics comparison for selected models
            if not selected_models:
                st.warning("Please select at least one model to compare.")
            else:
                    # Prepare data for graphical comparison
                metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                metric_values = {metric: [] for metric in metrics}

                for model_name in selected_models:
                    results = model_results.get(model_name)
                    if results and results['Status'] == 'Success':
                        for metric in metrics:
                            metric_values[metric].append(results[metric])

                    # Create a DataFrame for plotting
                metric_df = pd.DataFrame(metric_values, index=selected_models)

                fig = px.bar(
                metric_df,
                x=metric_df.index,
                y=metrics,
                title="Performance Metrics Comparison",
                labels={"value": "Score", "metric": "Metric", "model": "Model"},
                barmode='group', 
                )

                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Saved Models")

            # Create a DataFrame from the saved models dictionary
            model_accuracy_df = pd.DataFrame(
                [(model_name, results['Accuracy']) for model_name, results in model_results.items()],
                columns=["Model", "Accuracy"]
            )
            st.dataframe(model_accuracy_df)

            saved_models_dir = "saved_models"
            os.makedirs(saved_models_dir, exist_ok=True)

            for model_name, results in model_results.items():
                if results["Status"] == "Success":
                    model_file_path = os.path.join(saved_models_dir, f"{model_name}.pkl")
                    joblib.dump(models[model_name], model_file_path)
                            
            selected_model = st.selectbox("Select Model to Download", options=model_accuracy_df["Model"])

            if selected_model:
                model_file_path = os.path.join(saved_models_dir, f"{selected_model}.pkl")
                if os.path.exists(model_file_path):
                        # Create a download button for the selected model
                    with open(model_file_path, "rb") as file:
                        st.download_button(
                            label=f"Download {selected_model} (.pkl)",
                            data=file,
                            file_name=f"{selected_model}.pkl",
                            mime="application/octet-stream",
                        )
                else:
                    st.error(f"Model file for {selected_model} not found!")

            def plot_learning_curve(estimator, title, X, y, cv=None, train_sizes=np.linspace(0.1, 1.0, 5)):
                plt.figure(figsize=(4, 4))
                plt.title(title)
                plt.xlabel("Training Examples")
                plt.ylabel("Score")
                    
                train_sizes, train_scores, test_scores = learning_curve(
                    estimator, X, y, cv=cv, train_sizes=train_sizes, scoring='accuracy', n_jobs=-1
                )
                    
                train_scores_mean = np.mean(train_scores, axis=1)
                train_scores_std = np.std(train_scores, axis=1)
                test_scores_mean = np.mean(test_scores, axis=1)
                test_scores_std = np.std(test_scores, axis=1)
                    
                    # Plot learning curve
                plt.grid()
                plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                                train_scores_mean + train_scores_std, alpha=0.1, color="r")
                plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                                test_scores_mean + test_scores_std, alpha=0.1, color="g")
                plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
                plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
                    
                plt.legend(loc="best")
                return plt

            st.subheader("Learning Curves for All Models")

            model_names = list(models.keys())
            n_models = len(model_names)
            cols_per_row = 4
            rows = (n_models // cols_per_row) + (1 if n_models % cols_per_row else 0)

                # Iterate through rows and columns
            for row in range(rows):
                cols = st.columns(cols_per_row)
                for col_idx in range(cols_per_row):
                    model_idx = row * cols_per_row + col_idx
                    if model_idx < n_models:
                        model_name = model_names[model_idx]
                        model = models[model_name]
                        model_accuracy = model_results.get(model_name, {}).get("Accuracy", "N/A")
                            
                        with cols[col_idx]:
                                # Title includes the model's accuracy rate
                            fig = plot_learning_curve(
                                model, 
                                f"{model_name} \n Accuracy: {model_accuracy:.2%})", 
                                X_train, 
                                y_train, 
                                cv=5
                                )
                            st.pyplot(fig)

                

            st.subheader("Confusion Matrix for Each Model")

                # Define number of models and layout (3x3 grid)
            n_models = len(models)
            rows = (n_models + 2) // 4
            cols_per_row = 4
            model_names = list(models.keys())

                # Get class names (assumes y_test contains class labels)
            class_names = sorted(y_test.unique()) 

            for row in range(rows):
                cols = st.columns(cols_per_row)
                for col_idx in range(cols_per_row):
                    model_idx = row * cols_per_row + col_idx
                    if model_idx < n_models:
                        model_name = model_names[model_idx]
                        model = models[model_name]
                        model_accuracy = model_results.get(model_name, {}).get("Accuracy", "N/A")
                            
                        with cols[col_idx]:
                            if model_results.get(model_name, {}).get("Status") == "Success":
                                    # Get predictions
                                y_pred = model.predict(X_test)

                                    # Calculate confusion matrix
                                cm = confusion_matrix(y_test, y_pred)

                                    # Plot confusion matrix using seaborn
                                fig, ax = plt.subplots(figsize=(6, 4))
                                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                                xticklabels=class_names, yticklabels=class_names)
                                ax.set_title(f"{model_name} \n  Accuracy: {model_accuracy:.2%}")
                                ax.set_xlabel("Predicted")
                                ax.set_ylabel("Actual")
                                    
                                st.pyplot(fig)
                            else:
                                st.warning(f"{model_name} did not train successfully.")  
                                                                    
    except ValueError as e:
        st.error(f"Error: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

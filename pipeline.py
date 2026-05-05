import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.model_selection import LeaveOneGroupOut
import scipy.stats as stats
import matplotlib.pyplot as plt


data_path = os.path.join("data", 'HR_data_2.csv')
df = pd.read_csv(data_path)

df = df.drop(columns=["Unnamed: 0", "Round", "Puzzler", "original_ID", "raw_data_path", "Team_ID", "Cohort"])

quastinare_columns = ["Frustrated","upset","hostile", "alert", "ashamed", "inspired", "nervous", "attentive", "afraid", "active", "determined"]
data_columns = [col for col in df.columns if col not in quastinare_columns + ["Phase"] + ["Individual"]]

df = df.dropna(subset=data_columns)



scaled_data = []


# loop over experiments (phase 1+3 and only phase 1)
for experiment in [["phase1", "phase3"], ["phase1"]]:
    # Isolate resting data per subject
    for subject_id, subject_data in df.groupby('Individual'):
        resting_data = subject_data[subject_data['Phase'].isin(experiment)]
        
        scaler = StandardScaler()
        scaler.fit(resting_data[data_columns])
        
        subject_scaled = subject_data.copy()
        subject_scaled[data_columns] = scaler.transform(subject_data[data_columns])
        scaled_data.append(subject_scaled)

    df_scaled = pd.concat(scaled_data, ignore_index=True)

    logo = LeaveOneGroupOut()
    groups = df_scaled['Individual']

    results = []

    all_test_evaluations = []

    print("Starting LOSO Cross-Validation...\n")

    for fold_idx, (train_index, test_index) in enumerate(logo.split(df_scaled, groups=groups)):
        
        train_df = df_scaled.iloc[train_index]
        test_df = df_scaled.iloc[test_index]
        
        test_subject = test_df['Individual'].iloc[0]
        
        train_rest = train_df[train_df['Phase'].isin(experiment)][data_columns]

        test_phase1 = test_df[test_df['Phase'] == "phase1"][data_columns]
        test_phase2 = test_df[test_df['Phase'] == "phase2"][data_columns]
        test_phase3 = test_df[test_df['Phase'] == "phase3"][data_columns]

        # reduce dimensionality with PCA
        pca = PCA(n_components=0.95)
        pca.fit(train_rest)
        
        train_rest_pca = pca.transform(train_rest)

        test_phase1_pca = pca.transform(test_phase1)
        test_phase2_pca = pca.transform(test_phase2)
        test_phase3_pca = pca.transform(test_phase3)

        # train the OCSVM
        ocsvm = OneClassSVM(kernel='rbf', nu=0.1, gamma='scale')
        ocsvm.fit(train_rest_pca)
        
        # predict on all phases
        predictions_p1 = ocsvm.predict(test_phase1_pca)
        predictions_p2 = ocsvm.predict(test_phase2_pca)
        predictions_p3 = ocsvm.predict(test_phase3_pca)
        
        total_points = len(predictions_p2)
        anomalies = np.sum(predictions_p2 == -1)

        rate_p1 = (np.sum(predictions_p1 == -1) / len(predictions_p1)) * 100 if len(predictions_p1) > 0 else 0
        rate_p2 = (np.sum(predictions_p2 == -1) / len(predictions_p2)) * 100 if len(predictions_p2) > 0 else 0
        rate_p3 = (np.sum(predictions_p3 == -1) / len(predictions_p3)) * 100 if len(predictions_p3) > 0 else 0
        
        results.append({
            'Test_Subject': test_subject,
            'Total_Puzzle_Samples': total_points,
            'Anomalies_Flagged': anomalies,
            'Rate_p1%': rate_p1,
            'Rate_p2%': rate_p2, 
            'Rate_p3%': rate_p3,
            'PCA_Components_Used': pca.n_components_
        })
        
        test_q_data = test_df[test_df['Phase'] == "phase2"][quastinare_columns].copy()
        
        test_q_data['Anomaly_Flag'] = (predictions_p2 == -1).astype(int) 
        test_q_data['Subject'] = test_subject
        
        all_test_evaluations.append(test_q_data)
        
        print(f"Subject {test_subject}: {rate_p2:.2f}% of puzzle phase flagged as anomalous.")

    results_df = pd.DataFrame(results)
    print("\n--- Summary of Anomaly Detection ---")
    print(results_df.describe())

    print("\n--- Global Correlation with Questionnaires ---")
    print("Interpreting the metric: A positive correlation means that when the model flagged an anomaly,")
    print("the participant generally reported higher levels of that specific emotion.\n")

    eval_df = pd.concat(all_test_evaluations, ignore_index=True)

    for emotion in quastinare_columns:
        # Drop rows where the user might not have answered that specific question (NaN)
        clean_df = eval_df.dropna(subset=[emotion, 'Anomaly_Flag'])
        
        if len(clean_df) > 0:
            # Calculate Point-Biserial correlation (binary flag vs continuous)
            corr, p_value = stats.pointbiserialr(clean_df['Anomaly_Flag'], clean_df[emotion])
            
            # Determine significance
            sig = "*" if p_value < 0.05 else ""
            
            print(f"Emotion: {emotion.ljust(12)} | Correlation: {corr:+.3f} | p-value: {p_value:.3f} {sig}")

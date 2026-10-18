# SIMGuard — SIM-Swapping Fraud Risk Assessment

**IBM Z Datathon 2026 | AI for Good**

## Overview

SIMGuard is a prototype fraud-risk assessment application designed to flag potentially suspicious transactions using behavioral signals associated with possible SIM-swapping activity.

SIM swapping occurs when an attacker fraudulently transfers a victim's phone number to a SIM card or service under the attacker's control. This may help an attacker intercept authentication messages and attempt account takeover.

SIMGuard combines a machine-learning model with an interactive dashboard to help users explore suspicious transaction patterns and prioritize cases for further review.

## Problem Statement

SIM-swapping attacks can undermine phone-based authentication and expose financial accounts to unauthorized activity. Reviewing every transaction manually can be difficult, while simple fixed rules may miss combinations of suspicious behaviors.

## Proposed Solution

SIMGuard provides:

- **Transaction risk scoring:** estimates a model-generated fraud score from transaction and account behavior.
- **Behavioral signals:** considers SIM-change indicators, device changes, failed logins, transaction amounts, transaction frequency, location anomalies, and recent credential changes.
- **Interactive dashboard:** supports individual transaction assessment.
- **Batch analysis:** accepts CSV files, ranks transactions by model score, and allows results to be downloaded.
- **Review support:** helps users identify transactions that may warrant further investigation.

## Technical Architecture

1. Synthetic transaction data is generated using Python, NumPy, and pandas.
2. A Random Forest classifier is trained using scikit-learn.
3. The trained model is saved as a Joblib artifact.
4. A Streamlit application loads the model and accepts transaction inputs.
5. The application displays risk scores and batch-analysis results.

## Technology Stack

- Python
- pandas and NumPy
- scikit-learn
- Joblib
- Streamlit
- Plotly (if used by the application)

## Dataset

The current prototype uses a synthetically generated dataset of 10,000 transactions. Fraud labels are generated from simulated assumptions rather than independently verified real-world incidents.

The model uses nine behavioral features, including SIM-change timing, device changes, failed logins, transaction amounts, transaction frequency, unusual location indicators, and recent credential changes.

## Evaluation

An initial stratified train/test split produced the following results on the synthetic dataset:

- Accuracy: 99.40%
- Fraud precision: 97.02%
- Fraud recall: 97.85%
- ROC-AUC: 0.9997

These figures describe performance on the generated test split only. Because training and test examples come from the same synthetic generation process, they do not establish real-world fraud-detection performance.

## Limitations and Responsible Use

- The dataset is synthetic and may not represent actual attacker behavior or legitimate customer variation.
- Risk scores are experimental model outputs, not calibrated real-world fraud probabilities.
- False positives and false negatives are possible.
- The application is a demonstration prototype, not a production banking security system.
- Any real deployment would require representative, appropriately authorized data, independent validation, privacy safeguards, monitoring, and human review.

## Running the Application

1. Install Python and the listed dependencies.
2. Generate the synthetic dataset using the dataset-generation script.
3. Train the model using `train_model.py`.
4. Start the application with:

   `python -m streamlit run app.py`

5. Open the local URL shown in the terminal.

The trained model artifact must be available at the path expected by the application.

## IBM Z Relevance

The current implementation runs locally on Windows. The application has not been deployed to IBM Z or LinuxONE as part of this implementation.

IBM Z/LinuxONE could be explored as a future hosting environment for secure, scalable transaction-risk assessment, subject to deployment, integration, and performance validation.

## Future Work

- Evaluate on independently sourced and authorized data.
- Test difficult legitimate and fraudulent scenarios.
- Analyze false positives and false negatives.
- Add model explanations and an analyst review workflow.
- Explore deployment on IBM Z/LinuxONE.

## Project Details

- **Team name:** [Add team name]
- **Team members:** [Add names]
- **Repository:** [Add repository URL]
- **Demo video:** [Add link when available]

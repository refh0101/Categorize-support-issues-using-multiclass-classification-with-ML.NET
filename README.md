# Categorize-support-issues-using-multiclass-classification-with-ML.NET
This repository demonstrates a machine learning solution built with ML.NET to classify GitHub issues based on their titles and descriptions. 

## Features
- Data Preprocessing: Converts categorical labels to numeric keys, featurizes text fields, and combines them into a single feature vector.

- Model Training: Utilizes the SDCA Maximum Entropy trainer for efficient multiclass classification.
  
-Evaluation: Provides metrics such as accuracy, log-loss, and log-loss reduction to measure the model's performance.

-Prediction: Supports real-time prediction for individual data points using the trained model.

-Model Persistence: Saves and reloads the trained model for reuse in production environments.

## Technology Stack
- ML.NET: Core machine learning framework for building and training models.
- C#: Programming language used to implement the pipeline and logic.
- .NET Core: Provides cross-platform compatibility for the application.

## How It Works
- Data Preparation:

Input data is read from TSV files containing GitHub issues.
The dataset includes fields such as Title, Description, and Area (classification label).
Pipeline Definition:

Text fields are featurized into numeric representations.
Labels are mapped to numeric keys.
Featurized fields are combined into a single Features column.

- Model Training:

The training dataset is fed through the pipeline to build a model.
The trained model is saved to a .zip file for future use.
Evaluation:

A test dataset is used to measure the model's accuracy and other metrics.
Prediction:

A trained model is loaded to classify a new GitHub issue based on its Title and Description.

## Folder Structure

.
- Data
-- issues_train.tsv    # Training dataset
│   ├── issues_test.tsv     # Test dataset
├── Models
│   ├── model.zip           # Saved trained model
├── Program.cs              # Main application logic
└── README.md               # Project description

## Prerequisites
- .NET Core SDK (version 3.1 or higher)
- Visual Studio or any C# IDE
- ML.NET version 1.5 or higher

## How to Run
Clone the repository:
bash
Copy code
git clone https://github.com/your-username/github-issue-classifier.git


## Use Cases
- Classifying GitHub issues by their area or category for better issue management.
- Adapting the pipeline to classify support tickets, emails, or any text-based data.
- Learning and experimenting with ML.NET's capabilities for multiclass classification.

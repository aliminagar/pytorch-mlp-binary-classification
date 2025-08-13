🧠 MLP Binary Classification – PyTorch Implementation
📌 Project Overview
This project implements a Multilayer Perceptron (MLP) neural network for binary classification using PyTorch.
The work is part of my PyTorch Udemy course learning path, where I extended the base material with my own experimental design and result analysis.
The aim is to train and evaluate a neural network on a two-class dataset, systematically changing one hyperparameter at a time to see its effect on performance.
We use:
•	MLP model implemented from scratch in PyTorch
•	Train/Test split with standardized features
•	Multiple experiments covering activation functions, optimizers, learning rates, and epochs
•	Performance evaluation using multiple metrics
•	Visualization of decision boundaries and ROC curves
•	Comparative analysis of configurations
________________________________________
📂 Dataset
•	File: Dataset.csv
•	Features:
o	Feature 0 (float)
o	Feature 1 (float)
•	Target:
o	target (0 or 1)
•	Size: 1500 total samples
•	Split:
o	Training: 1050 samples (70%)
o	Testing: 450 samples (30%)
Preprocessing:
•	Standardized features using StandardScaler from scikit-learn
•	Encoded labels using LabelEncoder
________________________________________
🧮 Model Architecture
Implemented in PyTorch (torch.nn.Module):
Input Layer:  2 features
Hidden Layer: 32 neurons (variable in code)
Activation:   tanh or logistic (sigmoid)
Output Layer: 1 neuron (binary classification)
Loss:         BCEWithLogitsLoss
Optimizer:    SGD or Adam
________________________________________
⚙️ Experiments
We systematically varied the following parameters:
1.	Activation Functions
o	tanh
o	logistic (sigmoid)
2.	Learning Rates
o	0.001
o	0.01
o	0.1
3.	Optimizers
o	Stochastic Gradient Descent (SGD)
o	Adam
4.	Epochs
o	50
o	100
Baseline Configuration:
{
    "activation": "tanh",
    "lr": 0.01,
    "optimizer": "SGD",
    "epochs": 50
}
________________________________________
📊 Metrics Used
For each run, we recorded:
•	Accuracy
•	Precision
•	Recall
•	F1-score
•	AUC (Area Under ROC Curve)
•	Confusion Matrix (TP, TN, FP, FN)
________________________________________
📈 Results Summary
Experiment	Accuracy	Precision	Recall	F1	AUC	TN	FP	FN	TP
opt=Adam	1.0000	1.0000	1.0000	1.0000	1.0000	225	0	0	225
lr=0.1	0.9356	0.9495	0.9200	0.9345	0.9877	214	11	18	207
act=logistic	0.9022	0.9209	0.8800	0.8997	0.9719	208	17	27	198
epochs=100	0.9000	0.9167	0.8800	0.8986	0.9718	207	18	27	198
baseline	0.9022	0.9209	0.8800	0.8997	0.9715	208	17	27	198
lr=0.001	0.9022	0.9209	0.8800	0.8997	0.9709	208	17	27	198
________________________________________
📉 Visualizations
Decision Boundary (PCA-2D)
We reduced the standardized features to 2D using PCA and plotted decision boundaries for each experiment.
These plots show how different parameter settings influence class separation.
Example:
________________________________________
ROC Curves
We plotted the ROC curves for all experiments on a single chart to compare AUC performance.
________________________________________
Confusion Matrices
For each experiment, a confusion matrix heatmap was generated.
Example:
________________________________________
📦 Installation & Usage
1. Clone the Repository
git clone https://github.com/aliminagar/pytorch-mlp-binary-classification.git
cd pytorch-mlp-binary-classification

2. Install Dependencies
pip install -r requirements.txt

3. Run the Notebook
jupyter notebook mlp_binary_classification.ipynb
________________________________________
📚 Learning Outcomes
•	Built and trained an MLP in PyTorch.
•	Learned how hyperparameters affect classification performance.
•	Practiced data preprocessing and feature scaling.
•	Implemented ROC curve & AUC analysis.
•	Visualized decision boundaries in 2D.
•	Compared multiple optimization strategies.
________________________________________
🚀 Future Improvements
•	Add cross-validation for more robust performance estimation.
•	Try deeper architectures (more hidden layers).
•	Use dropout to improve generalization.
•	Automate hyperparameter search with Optuna or GridSearch.
________________________________________
📜 License
This project is released under the MIT License.
________________________________________
🔗 Acknowledgements
•	Udemy PyTorch Course (inspiration & guidance)
•	PyTorch documentation
•	scikit-learn documentation

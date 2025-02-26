# Timing_Analysis

## Timing Analysis using GNN for RTL Complexity Prediction

#### SOLUTION SUMMARY
A Graph Neural Network (GNN)-based Timing Prediction Tool that extracts RTL netlist features and predicts the logic depth of a circuit.

#### FEATURES 
Parses Verilog Netlists and converts them into circuit graphs.
Extracts node-level features (Fan-In, Fan-Out, Depth).
Implements a Graph Convolutional Network (GCN) to predict logic depth.
Provides real-time timing prediction on new circuits.

### PROJECT STRUCTURE
Timing_Analysis/
│── netlists/                # Verilog netlists directory
│── scripts/                 # Timing extraction scripts
│── Feature_Extraction.py    # Extracts node features from netlists
│── GNN.py                   # Graph Neural Network model
│── Graph_Generation.py      # Converts netlist to graph
│── Predict_Timing.py        # Predicts timing using the trained model
│── Train.py                 # Trains the model
│── trained_model.pth        # Pre-trained model (generated after training)
│── README.md                # Project Documentation
│── requirements.txt         # Dependencies list


#### INSTALLATION AND SETUP
Follow these steps to set up and run the Timing Analysis project on your local machine.
Prerequisites:
Python 3.8+ 
Git 
pip (Usually included with Python)
virtualenv (Recommended)

**Step 1: Clone the Repository** <br>  
git clone https://github.com/your-username/Timing_Analysis.git
cd Timing_Analysis

**Step 2: Set Up a Virtual Environment (Optional but Recommended)** <br>
On Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate
<br>
On macOS/Linux
python3 -m venv venv
source venv/bin/activate

**Step 3: Install Dependencies**
pip install -r requirements.txt

**Step 4: Train the Model**
python Train.py

**Step 5: Run Prediction on a Netlist**
python Predict_Timing.py

## SAMPLE OUTPUT
![image](https://github.com/user-attachments/assets/408839e8-c47b-45ed-955f-4ed5a5788dce)



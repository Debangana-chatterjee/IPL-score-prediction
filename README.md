# IPL-Score-Prediction

**Overview**

This project is an IPL (Indian Premier League) Score Prediction App built using Streamlit and an LSTM model. It allows users to predict the final score of a cricket match based on input parameters such as batting team, bowling team, venue, overs, wickets fallen, and runs scored in the last 5 overs.


---
**Files in This Repository**

- app.py: The main Streamlit application that serves the IPL score prediction interface.

- lstm_model.h5: The trained LSTM model used for predicting the final score.

- IPL_Score_Prediction.ipynb: A Jupyter Notebook that contains the data preprocessing, model training, and evaluation steps.

- encoder.pkl: The encoder used for categorical feature transformation.

- scaler.pkl: The scaler used for numerical feature normalization.


---
**Key Insights**

- The model leverages LSTM, which is well-suited for time-series prediction tasks.

- Feature engineering plays a crucial role in improving the prediction accuracy, particularly encoding categorical variables and scaling numerical values.

- The model's performance was evaluated using RMSE and R² score.

- The app provides an intuitive interface for cricket enthusiasts to estimate match outcomes in real time.


---
**Model Performance**

The LSTM model was evaluated using the test dataset, with the following performance metrics:

- Root Mean Squared Error (RMSE): 10.28
- R² Score: **0.87**

These values indicate the model's ability to predict match scores with reasonable accuracy.

---
**Future Improvements**

- Enhancing model accuracy using additional features.

- Exploring different deep learning architectures.

- Integrating real-time data updates.

---

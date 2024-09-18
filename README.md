# Food-Delivery-Prediction

The food delivery time prediction model is essential in the the food delivery industry, where timely and accurate deliveries are critical for customer satisfaction and overall experience.

To create an effective prediction model, data needs to cleaned meticulously in order to eliminate the errors and inconsistencies, enduring reliability and accuracy of the predictions.

Nest, feature engineering is used to derive valuable insights from the dataset. By considering factors such as delivery person's age, ratings, location coordinates, and time-related variable, we aimed to identify the key elements that affect the delivery time. These engineered features enhance the model's predictive power.

I have built the predictive model using regression algorithms like linear regression, decision tree, random forest and XGBoost. The model was trained on a subset of the dataset using cross validation techniques to ensure robustness. WE evaluated te model's accuracy with metrics such as mean squared error (MSE) and R-squared (R2) score.

The food delivery time prediction model enables business to optimize their operations and enhance the overall delivery experience for their customers.

I have added the following files in the repository
  1. ```Food Delivery Prediction.ipynb``` - The file contains the complete code involving Loading Data, Data Cleaning, Data Visualization, Model creation, Model evaluation
  2. ```predict.py``` - Contains the streamlit code to build a basic webpage to predict the food delivery time
  3. ```std_scaler.pkl``` - It is pickle file contains the Standard Scaler weights and biases used for data normalization
  4. ```xgb_model.pkl``` - It contains the XGB Model's weights and biases which can be reused
  5. ```train.csv``` - It is the data on which the model is trained

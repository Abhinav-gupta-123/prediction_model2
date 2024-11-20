from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import pandas as pd
import joblib  # Import joblib for saving models

# # Load the dataset
# df = pd.read_csv("/Users/sanghvi/Desktop/SIH/dataset/synthetic_aluminium_data.csv")

# # Separate features and target variables
# y = df[['UTS', 'Elongation', 'Conductivity']]
# X = df[['CastingTemp', 'CoolingWaterTemp', 'CastingSpeed', 'EntryTempRollingMill', 
#         'EmulsionTemp', 'EmulsionPressure', 'EmulsionConcentration', 'RodQuenchWaterPressure']]

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardize features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Initialize the Random Forest model
# rf = RandomForestRegressor(random_state=42)

# # Wrap the model in MultiOutputRegressor for multi-output prediction
# multi_target_rf = MultiOutputRegressor(rf)

# # Fit the model on the training data
# multi_target_rf.fit(X_train, y_train)

# # Predict and evaluate on the training and testing set
# y_rf_train_pred = multi_target_rf.predict(X_train)
# y_rf_test_pred = multi_target_rf.predict(X_test)

# print("\nRandom Forest Regressor (Multi-output)")
# print("MSE (test):", mean_squared_error(y_test, y_rf_test_pred))
# print("R^2 Score (test):", r2_score(y_test, y_rf_test_pred))
# print("MSE (train):", mean_squared_error(y_train, y_rf_train_pred))
# print("R^2 Score (train):", r2_score(y_train, y_rf_train_pred))

# # Hyperparameter tuning for Random Forest using GridSearchCV
# param_grid_rf = {
#     'estimator__n_estimators': [50, 100, 150],
#     'estimator__max_depth': [None, 10, 20, 30],
#     'estimator__min_samples_split': [2, 5, 10],
#     'estimator__min_samples_leaf': [1, 2, 4],
#     'estimator__bootstrap': [True, False]
# }

# grid_search_rf = GridSearchCV(estimator=multi_target_rf, param_grid=param_grid_rf, 
#                               cv=5, scoring='r2', n_jobs=-1)
# grid_search_rf.fit(X_train, y_train)

# # Best parameters for Random Forest
# print("\nBest parameters from GridSearchCV for Random Forest:", grid_search_rf.best_params_)

# # Evaluate with best estimator
# best_rf = grid_search_rf.best_estimator_
# y_best_rf_test_pred = best_rf.predict(X_test)

# print("\nOptimized Random Forest Test MSE:", mean_squared_error(y_test, y_best_rf_test_pred))
# print("Optimized Random Forest R^2 Score (test):", r2_score(y_test, y_best_rf_test_pred))

# # Save the scaler and best_rf model
# joblib.dump(scaler, 'scaler_rf.pkl')
# joblib.dump(best_rf, 'best_rf_model.pkl')

# print("\nScaler and Best Random Forest model have been saved.")

# Function to load the saved models
def load_models():
    scaler = joblib.load(r"C:\Users\abhin\Desktop\PREDICTION\PREDICTION\Random Forest Regressor\scaler_rf.pkl")
    best_rf = joblib.load(r"C:\Users\abhin\Desktop\PREDICTION\PREDICTION\Random Forest Regressor\best_rf_model.pkl")
    return scaler, best_rf

# Accept runtime input and make predictions
def get_runtime_input():
    print("\nEnter the input parameters for prediction:")
    casting_temp = float(input("Casting Temperature (°C): "))
    cooling_water_temp = float(input("Cooling Water Temperature (°C): "))
    casting_speed = float(input("Casting Speed (cm/s): "))
    entry_temp_rolling_mill = float(input("Entry Temperature at Rolling Mill (°C): "))
    emulsion_temp = float(input("Emulsion Temperature (°C): "))
    emulsion_pressure = float(input("Emulsion Pressure (bar): "))
    emulsion_concentration = float(input("Emulsion Concentration (%): "))
    rod_quench_water_pressure = float(input("Rod Quench Water Pressure (bar): "))
    
    # Construct input array
    input_features = [[casting_temp, cooling_water_temp, casting_speed, 
                       entry_temp_rolling_mill, emulsion_temp, emulsion_pressure, 
                       emulsion_concentration, rod_quench_water_pressure]]
    
    return input_features

# Main loop for making predictions
while True:
    # Load the saved models (uncomment the next line if you want to load models from disk)
    scaler, best_rf = load_models()
    
    # Get user input for prediction
    user_input = get_runtime_input()
    
    # Scale the user input using the saved scaler
    user_input_scaled = scaler.transform(user_input)
    
    # Make predictions using the loaded best_rf model
    predictions = best_rf.predict(user_input_scaled)
    
    print("\nPredicted Properties:")
    print(f"UTS (Ultimate Tensile Strength): {predictions[0][0]:.2f}")
    print(f"Elongation: {predictions[0][1]:.2f}")
    print(f"Conductivity: {predictions[0][2]:.2f}")
    
    cont = input("\nDo you want to make another prediction? (yes/no): ").strip().lower()
    if cont != 'yes':
        break

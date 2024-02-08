import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn import metrics
from sklearn import linear_model
from scipy.stats import pearsonr

import streamlit as st

# READ INPUT DATA
st.title("Stock Price Prediction")

genre = st.radio("Click To Select Dataset", ('No', 'Yes'))

if genre == 'Yes':
    st.write('You selected Data.')
    st.subheader('Stock Price Prediction')
    dataframe = pd.read_csv("D:/programming/PYTHON/FINALYEARPROJECT/Stock Dataset.csv")
    
    # Display input data
    st.subheader('Input Data')
    st.dataframe(dataframe.head(20))

    # Data Preprocessing
    st.subheader('Data Preprocessing')
    st.text(dataframe.isnull().sum())

    # Drop unwanted columns
    columns = ['Date']
    dataframe.drop(columns, inplace=True, axis=1)
    st.subheader('Drop Unwanted Columns')
    st.dataframe(dataframe)

    # Correlation
    st.subheader('Correlation')

    # Remove commas and convert columns to numeric, handling errors
    dataframe['Close'] = pd.to_numeric(dataframe['Close'].str.replace(',', ''), errors='coerce')
    dataframe['High'] = pd.to_numeric(dataframe['High'].str.replace(',', ''), errors='coerce')

    # Drop NaN values after conversion
    dataframe = dataframe.dropna(subset=['Close', 'High'])

    # Check if there is still enough data for correlation
    if len(dataframe) > 0:
        corr, _ = pearsonr(dataframe['Close'], dataframe['High'])
        st.text('Pearsons correlation: {:.2f}'.format(corr))
    else:
        st.text('Insufficient data for correlation calculation')

    # Data Splitting
    st.subheader('Data Splitting')
    X_train, X_test, Y_train, Y_test = train_test_split(dataframe.drop('Close', axis=1), dataframe['Close'], test_size=0.3, random_state=40)
    st.text('Total number of data in input: {}'.format(dataframe.shape[0]))
    st.text('Total number of data in training part: {}'.format(X_train.shape[0]))
    st.text('Total number of data in testing part: {}'.format(X_test.shape[0]))


        
    # Support Vector Regression
    st.subheader('Support Vector Regression')
    
    # Convert 'Open', 'High', 'Low', and 'Close' columns to numeric values
    for col in ['Open', 'High', 'Low']:
        try:
            X_train[col] = X_train[col].astype(str).str.replace(',', '').astype(float)
            X_test[col] = X_test[col].astype(str).str.replace(',', '').astype(float)
        except ValueError as e:
            st.text(f"Error converting '{col}' column: {e}")
            st.stop()
    
    # Now fit the SVR model
    svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=1)
    try:
        svr_rbf.fit(X_train, Y_train)
    except ValueError as e:
        st.text(f"Error fitting SVR model: {e}")
        st.stop()
    
    # Use the same columns for prediction
    X_test_for_prediction = X_test[['Open', 'High', 'Low']]
    
    y_rbf = svr_rbf.predict(X_test_for_prediction)
    st.text('1. Mean Absolute Error: {:.2f}'.format(metrics.mean_absolute_error(Y_test, y_rbf)))
    st.text('2. Mean Squared Error: {:.2f}'.format(metrics.mean_squared_error(Y_test, y_rbf)))
    st.text('3. Root Mean Squared Error: {:.2f}'.format(np.sqrt(metrics.mean_squared_error(Y_test, y_rbf))))

    # Lasso Regression
    st.subheader('Lasso Regression')
    reg1 = linear_model.Lasso()
    reg1.fit(X_train, Y_train)
    prd_Ir = reg1.predict(X_test)
    st.text('1. Mean Absolute Error: {:.2f}'.format(metrics.mean_absolute_error(Y_test, prd_Ir)))
    st.text('2. Mean Squared Error: {:.2f}'.format(metrics.mean_squared_error(Y_test, prd_Ir)))
    st.text('3. Root Mean Squared Error: {:.2f}'.format(np.sqrt(metrics.mean_squared_error(Y_test, prd_Ir))))

    # Prediction
    # Prediction
    st.subheader('Prediction')
    min_length = min(len(X_test), len(prd_Ir))
    
    for i in range(min_length):
        st.text('Stock Price for ID {}: {:.2f}'.format(X_test.index[i], prd_Ir[i]))
    # Prediction Visualization
    st.subheader('Prediction Visualization')
    
    # Create a DataFrame for predicted values
    prediction_df = pd.DataFrame({
        'Actual': Y_test.values,
        'SVR Prediction': y_rbf,
        'Lasso Prediction': prd_Ir
    }, index=X_test.index)
    
    # Display the predicted values
    st.write(prediction_df.loc[[16, 21, 14, 4, 8, 11, 19]])
    
    # Visualization - Line Chart for Predicted Values
    st.subheader('Line Chart - Predicted Values')
    
    # Plot the line chart for predicted values
    st.line_chart(prediction_df[['Actual', 'SVR Prediction', 'Lasso Prediction']].loc[[16, 21, 14, 4, 8, 11, 19]], use_container_width=True)
    
    # Visualization - Vertical Bar Chart for Error Values
    st.subheader('Comparison Graph -- Error Values')
    
    # Create a DataFrame for error values
    error_df = pd.DataFrame({
        'Method': ['SVR', 'Lasso Regression'],
        'MAE': [metrics.mean_absolute_error(Y_test, y_rbf), metrics.mean_absolute_error(Y_test, prd_Ir)]
    })
    
    # Create a vertical bar chart
    st.bar_chart(error_df.set_index('Method'), use_container_width=True)
    
    
else:
    st.write("You didn't select Data")

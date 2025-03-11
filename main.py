import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler          # MinMaxScaler
from tensorflow.keras.models import Sequential  # Sequential model
from tensorflow.keras.layers import Dense, Dropout, LSTM   # Long Short Term Memory


#load data
Company = 'AAPL'

start = dt.datetime (2010,1,1)
end = dt.datetime(2024,1,1)

data = web.DataReader(Company, 'yahoo', start, end)

# Prepare data

scaler = MinMaxScaler(feature_range=(0,1))  #sk learn preprocessing model
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

Prediction_days = 90 # looks 90 days into the passed to predict the next day closing price

x_train = []
y_train = []

for x in range(Prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-Prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#Building the model

model=Sequential
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1)) # Prediction of the next closing price value (dense layer thingy)

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=50, batch_size=32)

'''DATA TEST'''
#Test the model accuracy on existing data
#load data

test_start = dt.datetime(2020,1,1)
test_end = dt.datetime.now()

test_data = web.DataReader (Company, 'yahoo', test_start, test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - Prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

#Make predictions on Test data

x_test = []

for x in range(Prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-Prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict (x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

#plot the test predictions
plt.plot(actual_prices, color='black', label=f'Actual {Company} price')
plt.plot(predicted_prices, color='green', label=f'Predicted {Company} price')
plt.title(f'{Company} share price')
plt.xlabel('Time')
plt.ylabel(f'{Company} share price')
plt.legend()
plt.show()


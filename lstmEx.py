import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Activation

#min-max normalization
def normalize(data):
    normalized = []
    for i in data:
        normalized.append([(j - i.min())/(i.max()-i.min()) for j in i])
    return np.array(normalized)

#load data
training_data = pd.read_csv('data/samsung.csv')
training_data.head()
hynix_data = pd.read_csv('data/Skhynix.csv')

high_pri = training_data['High'].values
low_pri = training_data['Low'].values
mid_pri = (high_pri + low_pri)/2

hy_mid = (hynix_data['High'].values + hynix_data['Low'].values)/2

#divide data by 60 days
days = 30
interval_len = days + 1
data_lens = len(high_pri)
result = []
hy_result = []

for i in range(data_lens - interval_len):
    result.append(mid_pri[i:i+interval_len])
    hy_result.append(hy_mid[i: i+interval_len])

#normalize data
result = normalize(result)
hy_result = normalize(hy_result)

#split train data and test data
row = int(round(result.shape[0]*0.9))
train = result[:row, :]
np.random.shuffle(train)

train_in = train[:, :-1]
train_in = np.reshape(train_in, (train_in.shape[0], train_in.shape[1], 1))
train_out = train[:, -1]

test_in = result[row:, :-1]
test_in = np.reshape(test_in, (test_in.shape[0], test_in.shape[1], 1))
test_out = result[row:, -1]

hy_in = hy_result[:, :-1]
hy_in = np.reshape(hy_in, (hy_in.shape[0], hy_in.shape[1], 1))
hy_out = hy_result[:, -1]

#make model
model = Sequential()
model.add(LSTM(30, return_sequences=True, input_shape=(30, 1)))
model.add(LSTM(128, return_sequences=False))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse')
model.summary()

model.fit(train_in, train_out, validation_data=(test_in, test_out), batch_size=10, epochs=20)

prediction_s = model.predict(test_in)
prediction_h = model.predict(hy_in)

figure = plt.figure(facecolor='white', figsize=(20, 10))
ax = figure.add_subplot(111)
ax.plot(test_out, label='real')
ax.plot(prediction_s, label='prediction')
ax.legend()
plt.show()

figure = plt.figure(facecolor='white', figsize=(20, 10))
ax = figure.add_subplot(111)
ax.plot(hy_out, label='real')
ax.plot(prediction_h, label='prediction')
ax.legend()
plt.show()

plt.show()










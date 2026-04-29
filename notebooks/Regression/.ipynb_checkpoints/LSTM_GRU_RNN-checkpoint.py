from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM, GRU, SimpleRNN
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import preprocessing
from ta import add_all_ta_features # Library that does financial technical analysis 

from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler(feature_range=(0, 1))

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

hist = pd.read_csv('Data/BCP.csv', index_col="Date", parse_dates=True)

# Add all technical analysis to the dataframe we've already loaded
hist = add_all_ta_features(hist, "Open", "High", "Low", "Close", "Volume", fillna=True) 

target_col = 'Close'

window_len = 5
test_size = 0.2
zero_base = True

def train_test_split(df, test_size=0.2):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data

train, test = train_test_split(hist, test_size=0.2)

def line_plot(line1, line2, label1=None, label2=None, title='Close Price history', lw=2):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_ylabel('Price', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=16)
    plt.show()

line_plot(train[target_col], test[target_col], 'training', 'test', title='Close Price history')

def extract_window_data(df, window_len=5, zero_base=True):
    window_data = []
    for idx in range(len(df) - window_len):
        tmp = df[idx: (idx + window_len)].copy()
        if zero_base:
            X = tmp.values
            # X = preprocessing.scale(X)
            X = scaler.fit_transform(X)
        window_data.append(X)
    return np.array(window_data)

def prepare_data(df, target_col, window_len=5, zero_base=True, test_size=0.2):
    train_data, test_data = train_test_split(df, test_size=test_size)
    X_train = extract_window_data(train_data, window_len, zero_base)
    X_test = extract_window_data(test_data, window_len, zero_base)
    y_train = train_data[target_col][window_len:].values
    y_test = test_data[target_col][window_len:].values
    if zero_base:
        y_train = y_train / train_data[target_col][:-window_len].values - 1
        y_test = y_test / test_data[target_col][:-window_len].values - 1
    return train_data, test_data, X_train, X_test, y_train, y_test

def build_lstm_model(X_train, y_train, X_test, y_test):
    # The LSTM architecture
    regressorLSTM = Sequential()    
    regressorLSTM.add(LSTM(256, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    regressorLSTM.add(Dropout(0.2))
    regressorLSTM.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    regressorLSTM.add(Dropout(0.2))
    regressorLSTM.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
    regressorLSTM.add(Dropout(0.2))
    regressorLSTM.add(Dense(32,kernel_initializer="uniform",activation='relu'))        
    regressorLSTM.add(Dense(1,kernel_initializer="uniform",activation='linear'))
    regressorLSTM.add(Activation("linear"))

    regressorLSTM.compile(loss="mean_squared_error", optimizer="adam", metrics=['accuracy'])
    regressorLSTM.fit(X_train, y_train, epochs=100, batch_size=96)
    lstm_pred = regressorLSTM.predict(X_test).squeeze()

    lstm_MSE = mean_squared_error(y_test, lstm_pred)
    lstm_MAE = mean_absolute_error(y_test, lstm_pred)
    lstm_RMS = np.sqrt(np.mean(np.power((np.array(y_test)-np.array(lstm_pred)),2)))
    lstm_R2 = r2_score(y_test, lstm_pred)
    print("Coefficient of Determination: {}".format(lstm_R2))
    print('LSTM Mean Absolute Error: {}'.format(lstm_MAE))
    print('LSTM MSE: {}'.format(lstm_MSE))
    print('LSTM RMS: {}'.format(lstm_RMS))

    return regressorLSTM, lstm_pred

def build_gru_model(X_train, y_train, X_test, y_test):
    # The GRU architecture
    regressorGRU = Sequential()
    regressorGRU.add(GRU(256, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    regressorGRU.add(Dropout(0.2))
    regressorGRU.add(GRU(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    regressorGRU.add(Dropout(0.2))
    regressorGRU.add(GRU(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
    regressorGRU.add(Dropout(0.2))
    regressorGRU.add(Dense(32,kernel_initializer="uniform",activation='relu'))        
    regressorGRU.add(Dense(1,kernel_initializer="uniform",activation='linear'))
    regressorGRU.add(Activation("linear"))
    
    regressorGRU.compile(loss="mean_squared_error", optimizer="adam", metrics=['accuracy'])
    regressorGRU.fit(X_train, y_train, epochs=100, batch_size=96)
    gru_pred = regressorGRU.predict(X_test).squeeze()

    gru_MSE = mean_squared_error(y_test, gru_pred)
    gru_MAE = mean_absolute_error(y_test, gru_pred)
    gru_RMS = np.sqrt(np.mean(np.power((np.array(y_test)-np.array(gru_pred)),2)))
    gru_R2 = r2_score(y_test, gru_pred)
    print("Coefficient of Determination: {}".format(gru_R2))
    print('GRU Mean Absolute Error: {}'.format(gru_MAE))
    print('GRU MSE: {}'.format(gru_MSE))
    print('GRU RMS: {}'.format(gru_RMS))

    return regressorGRU, gru_pred

def build_rnn_model(X_train, y_train, X_test, y_test):
    # The RNN architecture
    regressorRNN = Sequential()
    regressorRNN.add(SimpleRNN(256, return_sequences=True))
    regressorRNN.add(SimpleRNN(128, return_sequences=True))
    regressorRNN.add(SimpleRNN(64, return_sequences=True))
    regressorRNN.add(SimpleRNN(32))
    regressorRNN.add(Dense(1))
    regressorRNN.add(Activation("linear"))

    regressorRNN.compile(loss="mean_squared_error", optimizer="adam", metrics=['accuracy'])
    regressorRNN.fit(X_train, y_train, epochs=100, batch_size=96)
    rnn_pred = regressorRNN.predict(X_test).squeeze()

    rnn_MSE = mean_squared_error(y_test, rnn_pred)
    rnn_MAE = mean_absolute_error(y_test, rnn_pred)
    rnn_RMS = np.sqrt(np.mean(np.power((np.array(y_test)-np.array(rnn_pred)),2)))
    rnn_R2 = r2_score(y_test, rnn_pred)
    print("Coefficient of Determination: {}".format(rnn_R2))
    print('RNN Mean Absolute Error: {}'.format(rnn_MAE))
    print('RNN MSE: {}'.format(rnn_MSE))
    print('RNN RMS: {}'.format(rnn_RMS))

    return regressorRNN, rnn_pred

train, test, X_train, X_test, y_train, y_test = prepare_data(
    hist, target_col, window_len=window_len, zero_base=zero_base, test_size=test_size)

targets = test[target_col][window_len:]

regressorGRU, gru_pred = build_gru_model(X_train, y_train, X_test, y_test)
regressorRNN, rnn_pred = build_rnn_model(X_train, y_train, X_test, y_test)
regressorLSTM, lstm_pred = build_lstm_model(X_train, y_train, X_test, y_test)


lstm_pred = test[target_col].values[:-window_len] * (lstm_pred + 1)
lstm_pred = pd.Series(index=targets.index, data=lstm_pred)

gru_pred = test[target_col].values[:-window_len] * (gru_pred + 1)
gru_pred = pd.Series(index=targets.index, data=gru_pred)

rnn_pred = test[target_col].values[:-window_len] * (rnn_pred + 1)
rnn_pred = pd.Series(index=targets.index, data=rnn_pred)

line_plot(targets, lstm_pred, 'actual', 'prediction', lw=3)
line_plot(targets, gru_pred, 'actual', 'prediction', lw=3)
line_plot(targets, rnn_pred, 'actual', 'prediction', lw=3)
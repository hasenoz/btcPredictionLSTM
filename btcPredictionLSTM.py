import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# ilgili datanın yfinance kütüphanesinden çekilmesi
data = yf.download("BTC-USD", start="2014-01-01", end="2023-01-01")
data.set_index(pd.to_datetime(data.index), inplace=True)

# verinin scaler ile ölçeklendirilmesi
scaler = MinMaxScaler(feature_range=(0, 1))
data['Close'] = scaler.fit_transform(data[['Close']])

# train ve test verilerini ayırma
training_size = int(len(data) * 0.8)
test_size = len(data) - training_size
train_data, test_data = data[:training_size], data[training_size:]


# veri önhazırlığı
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step):
        a = dataset[i:(i + time_step)]
        X.append(a)
        Y.append(dataset[i + time_step])
    return np.array(X), np.array(Y)


time_step = 50
X_train, y_train = create_dataset(train_data['Close'], time_step)
X_test, y_test = create_dataset(test_data['Close'], time_step)

# veriyi yeniden şekillendirme
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# lstm modeli
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    #Dropout(0.25),  # Dropout katmanı eklendi
    LSTM(50, return_sequences=True),
    #Dropout(0.25),  # Dropout katmanı eklendi
    LSTM(50),
    #Dropout(0.25),  # Dropout katmanı eklendi
    Dense(1)
])
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

# modeli seçilen parametreler ile eğitme:
'''
Early stoppingi epok sayısına göre ayarladık. Epok sayısını normal düzeyde tutarken düşük bir patience değeri kullandık.
Sonucun daha verimli olması için loss değerindeki düşüşü gözlemledik. Yüksek epok sayısı ile düşük batch size kullanarak
detaylı bir eğitim yapılmasını sağlamayı amaçladık. Sonucu overfitting açısından inceleyerek parametrelere gerekli 
müdaheleleri yaptık.
'''
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=50, batch_size=128, verbose=1,
                    validation_data=(X_test, y_test), callbacks=[early_stopping])

# tahminlemenin model tarafından gerçekleştirimi
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Gerçek ve tahmin edilen değerleri çizdirme
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
actual = scaler.inverse_transform(data[['Close']].values)

plt.figure(figsize=(10, 6))
plt.plot(actual, label='Actual Prices')
plt.plot(np.concatenate([np.nan * np.ones(time_step), train_predict.flatten(),
                         np.nan * np.ones(len(actual) - len(train_predict) - time_step)]), label='Train Prediction')
plt.plot(np.concatenate([np.nan * np.ones(len(train_predict) + 2 * time_step), test_predict.flatten()]),
         label='Test Prediction')
plt.legend()
plt.show()

last_input = data['Close'][-time_step:].values.reshape(1, time_step, 1)


# Bu kısımdan sonrası, ne eğitime ne teste katılmamış veriyi tahminleme bölümü
def predict_future(model, last_input, scaler, steps=450):
    future_predictions = []
    for _ in range(steps):
        # Son girdiye göre yeni tahmin yapılıyor
        new_pred = model.predict(last_input)
        future_predictions.append(new_pred[0, 0])  # Tahmini kaydediyoruz
        # Yeni tahmini son girdiye ekleyip, en eski girdiyi çıkarıyoruz
        last_input = np.append(last_input[:, 1:, :], new_pred.reshape(1, 1, 1), axis=1)
    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))


# Tahminleri gerçekleştirme
future_predictions = predict_future(model, last_input, scaler, steps=450)

# Gerçek verilerin indirilmesi
actual_future_data = yf.download("BTC-USD", start="2023-01-01", end="2024-04-01")
actual_future_data.set_index(pd.to_datetime(actual_future_data.index), inplace=True)
actual_future_prices = actual_future_data['Close'].values

# Tahminlerin gerçek verilerle karşılaştırılması
plt.figure(figsize=(15, 7))
plt.plot(actual_future_prices, label='Actual Future Prices', color='red')
plt.plot(future_predictions[:len(actual_future_prices)], label='Predicted Future Prices', color='blue')
plt.title('Actual vs Predicted Prices')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()

# MAE ile gelecek için yaptığımız predictionun başarısının ölçülmesi
mae = mean_absolute_error(actual_future_prices[:len(future_predictions)],
                          future_predictions[:len(actual_future_prices)])
print(f"Mean Absolute Error on future data: {mae}")

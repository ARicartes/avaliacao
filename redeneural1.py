import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error


# Carregar os dados
df = pd.read_csv('areas_loteaveis.csv')

# Converter 'perimetro urbano' para valores numéricos
df['perimetro_urbano_num'] = df['perimetro urbano'].map({'sim': 1, 'nao': 0})

# Selecionar as características do imóvel que você quer usar para prever o valor
caracteristicas = ['area','distancia_centro', 'latitude', 'longitude']

X = df[caracteristicas]
y = df['valor']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Criar e treinar o modelo
modelo = Sequential()
modelo.add(Dense(32, input_dim=len(caracteristicas), activation='relu'))
modelo.add(Dense(16, activation='relu'))
modelo.add(Dense(1))

modelo.compile(loss='mean_squared_error', optimizer='adam')
modelo.fit(X_train, y_train, epochs=50, batch_size=32)

# Fazer previsões com o conjunto de teste
y_pred = modelo.predict(X_test)

# Calcular o erro quadrático médio das previsões
mse = mean_squared_error(y_test, y_pred)

print('O erro quadrático médio do modelo é:', mse)

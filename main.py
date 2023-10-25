import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Dense
degrees = np.array([0, 30, 45, 60, 90, 120, 135, 150, 180])
radians = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, 3*np.pi/4, 5*np.pi/6, np.pi])

inp = int(input('Введите величину в градусах: '))
model = keras.Sequential()
model.add(Dense(2, input_shape=(1,), activation='linear'))
model.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(0.1))
history = model.fit(degrees,radians,epochs=1000,verbose=0)
degrees_to_convert = np.array([inp])
radians_predicted = model.predict(degrees_to_convert)
print(degrees_to_convert[0], ' градусов в радианах = ', radians_predicted[0][0], sep='')
print(model.get_weights())

plt.plot(history.history['loss'])
plt.grid(True)
plt.show()

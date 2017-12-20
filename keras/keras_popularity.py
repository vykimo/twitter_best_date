from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
dataset = numpy.loadtxt("data\BarackObama_format.data", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:2]
Z = dataset[:,2]
Y  = []
mean = numpy.mean(Z)
for t in Z:
	Y.append(int(t>mean))

# create model
model = Sequential()
model.add(Dense(12, input_dim=2, activation='relu'))
model.add(Dense(2, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=150, batch_size=10, verbose=2)
# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
#predictions = model.predict(X)
# round predictions
#rounded = [round(x[0]) for x in predictions]
#print(rounded)
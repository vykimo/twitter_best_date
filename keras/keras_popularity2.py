from sklearn import datasets, linear_model
import numpy
from sklearn.cross_validation import train_test_split
# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
dataset = numpy.loadtxt("data\BarackObama_format.data", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:2]
y = dataset[:,2]
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=.7,random_state=42)

# create model
model = linear_model.LinearRegression()
model.fit(X_train, y_train) #train model on train data
model.score(X_train, y_train) #check score

print ('Coefficient: \n', model.coef_)
print ('Intercept: \n', model.intercept_) 
coefs = zip(model.coef_, X.columns)
model.__dict__
print "sl = %.1f + " % model.intercept_ + \
     " + ".join("%.1f %s" % coef for coef in coefs) #linear model
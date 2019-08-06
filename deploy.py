from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


app = Flask(__name__)

@app.route("/")

def hello():
	return render_template('home.html')


@app.route("/predict", methods=['POST'])

def predict():
	df = pd.read_csv('spam.csv', encoding="latin-1")
	df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
	df['label'] = df['v1'].map({'ham': 0,'spam': 1 })

	x = df['v2']
	y = df['label']
	cv = CountVectorizer()
	X = cv.fit_transform(x)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

	clf = MultinomialNB()
	clf.fit(X_train, y_train)
	clf.score(X_test, y_test)

	#y_pred = clf.predict(X_test)

	#print(classification_report(y_test, y_pred))

	#joblib.dump(clf, 'NB_spam_model.pkl')

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html', prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True, port=5000)

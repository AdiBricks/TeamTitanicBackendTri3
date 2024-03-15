from flask import Flask, request, jsonify
from flask import Blueprint
from flask_restful import Api, Resource
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import numpy as np

titanic_api = Blueprint('titanic_api', __name__, url_prefix='/api/titanic')
api = Api(titanic_api)

class TitanicAPI(Resource):
    def __init__(self):
        # Load the titanic dataset
        titanic_data = sns.load_dataset('titanic')
        td = titanic_data.copy()
        td.drop(['alive', 'who', 'adult_male', 'class', 'embark_town', 'deck'], axis=1, inplace=True)
        td.dropna(inplace=True)
        td['sex'] = td['sex'].apply(lambda x: 1 if x == 'male' else 0)
        td['alone'] = td['alone'].apply(lambda x: 1 if x else 0)

        # Encode categorical variables
        self.enc = OneHotEncoder(handle_unknown='ignore')
        embarked_encoded = self.enc.fit_transform(td[['embarked']].values.reshape(-1, 1))
        self.encoded_cols = self.enc.get_feature_names_out(['embarked'])

        # Add the encoded columns to the DataFrame and drop the original 'embarked' column
        td[self.encoded_cols] = embarked_encoded.toarray()
        td.drop(['embarked'], axis=1, inplace=True)

        # Train a logistic regression model with increased max_iter
        self.logreg = LogisticRegression(max_iter=1000)
        X = td.drop('survived', axis=1)
        y = td['survived']
        self.logreg.fit(X, y)

    def predict_survival(self, data):
        try:
            passenger = pd.DataFrame([data])  # Wrap data in a list to ensure DataFrame creation
            # Preprocess the passenger data
            passenger['sex'] = passenger['sex'].apply(lambda x: 1 if x == 'male' else 0)
            passenger['alone'] = passenger['alone'].apply(lambda x: 1 if x else 0)

            # Encode 'embarked' feature using the pre-fitted encoder
            embarked_encoded = self.enc.transform(passenger[['embarked']].values.reshape(-1, 1))
            passenger[self.encoded_cols] = embarked_encoded.toarray()
            passenger.drop(['embarked', 'name'], axis=1, inplace=True)

            # Predict the survival probability for the new passenger
            dead_proba, alive_proba = np.squeeze(self.logreg.predict_proba(passenger))

            # Return the survival probability
            return {
                'Death probability': '{:.2%}'.format(dead_proba),
                'Survival probability': '{:.2%}'.format(alive_proba)
            }
        except Exception as e:
            return {'error': str(e)}


    def post(self):
        try:
            data = request.json
            result = self.predict_survival(data)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)})

# Define API resources
api.add_resource(TitanicAPI, '/predict')


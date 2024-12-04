import pickle
import numpy as np

# Load the pre-trained model
model = pickle.load(open('phishing_model.pkl', 'rb'))

def predict_phishing(url):
    """
    Predict whether a given URL is phishing or safe.

    Parameters:
        url (str): The URL to be classified.

    Returns:
        str: "Phishing" if the URL is vulnerable, "Safe" otherwise.
    """
    prediction = model.predict([url])
    return "Phishing" if prediction[0] == 'bad' else "Safe"

# Example usage
while True:
    url_to_check = input('url: ')
    result = predict_phishing(url_to_check)
    print(f"The URL '{url_to_check}' is classified as: {result}\n")

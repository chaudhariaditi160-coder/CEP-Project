import re
import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

def clean_text(text):

    # Lowercase
    text = text.lower()

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    # Remove stopwords
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]

    cleaned_text = " ".join(filtered_words)

    return cleaned_text

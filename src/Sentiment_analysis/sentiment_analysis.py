from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np
import torch

def sentiment_analysis(df, text):
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    labels = ['Negative', 'Neutral', 'Positive']

    # Create empty lists to store the max score label, max score, and polarity
    max_score_labels = []
    max_scores = []
    polarities = []

    for tweet in df[text]:
        # Sentiment analysis
        encoded_tweet = tokenizer(tweet, return_tensors='pt')
        output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])

        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        # Find the index of the label with the highest score
        max_score_index = np.argmax(scores)

        # Get the corresponding label and score
        max_score_label = labels[max_score_index]
        max_score = scores[max_score_index]

        # Calculate the polarity as the geometric mean of positive and negative scores
        positive_score = scores[labels.index('Positive')]
        negative_score = scores[labels.index('Negative')]
        polarity = np.sqrt(positive_score * negative_score)

        # Append the max score label, max score, and polarity to the lists
        max_score_labels.append(max_score_label)
        max_scores.append(max_score)
        polarities.append(polarity)

    # Add the max score label, max score, and polarity as new columns in the DataFrame
    df['sentiment_label'] = max_score_labels
    df['sentiment_score'] = max_scores
    df['polarity'] = polarities
    return df

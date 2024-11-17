import requests
import pandas as pd
import numpy as np

# Function to get classification from LLM API (Hugging Face)
def classify_grant_purpose(grant_purpose, categories, record_number):
    api_url = "https://api-inference.huggingface.co/models/microsoft/deberta-v3-large"
    headers = {"Authorization": "Bearer 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"}  # Get a token from Hugging Face

    # Request payload
    # data = {
    #     "inputs": {
    #         "text": f"Classify the following grant purpose into one of these categories: {', '.join(categories)}. Grant Purpose: {grant_purpose}",
    #     }
    # }

    data = {
        "inputs": grant_purpose,
        "parameters": {
            "candidate_labels": categories  # Pass your predefined categories here
        }
    }

    try:
        response = requests.post(api_url, headers=headers, json=data)
        if response.status_code == 200:
            classification = response.json()['labels'][0]
            print(f"Processed record: {record_number}")
            return classification
        else:
            print(f"Error in response: {response.status_code}, {response.text}")
            return "Error in classification"
    except Exception as e:
        print(f"Error: {e}")
        return "API Error"

# Example dataset loading (assuming your data is in a CSV file)
df = pd.read_csv('990_output.csv')
random_10 = df.sample(n=10)

# Define the categories you want to classify into
categories = [
    "Education", "Scholarship", "Healthcare", "Research", 
    "Disaster Relief", "Youth Programs", "Community Development", "Others"
]

random_10['GrantPurpose'].replace([np.nan, np.inf, -np.inf], '', inplace=True)

random_10['GrantPurpose'] = random_10['GrantPurpose'].apply(lambda x: str(x) if isinstance(x, str) and len(str(x)) < 1000 else '')

# Classify each grant purpose and create a new column for the classification
random_10['GrantPurposeCategory'] = [
    classify_grant_purpose(purpose, categories, idx + 1) if purpose else 'Unknown'
    for idx, purpose in enumerate(random_10['GrantPurpose'])
]

# Save the updated dataframe with the new classification column
random_10.to_csv('updated_data_with_categories.csv', index=False)

print("Classification complete. Data saved.")

import json
import argparse
import os
import pickle
import requests
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="Generate QA pairs from wiki passages using an LLM API.")
parser.add_argument('--input', default=os.path.join(SCRIPT_DIR, '..', 'wiki_dataset.json'),
                    help='Path to input wiki dataset JSON file')
parser.add_argument('--output', default=None,
                    help='Path to output JSON file (default: input filename with _with_qa suffix)')
args = parser.parse_args()

input_file = args.input
if args.output is None:
    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_with_qa{ext}"
else:
    output_file = args.output

with open(input_file, 'r') as f:
    wiki_dataset = json.load(f)


api_key = os.getenv('HARVARD_API_KEY')


def api_call(prompt):
    url = "https://go.apis.huit.harvard.edu/ais-openai-direct-limited-schools/v1/chat/completions"
    payload = json.dumps({
        # "model": "o3-mini-2025-01-31",
        "model": "gpt-5.2-2025-12-11",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1
    })
    headers = {
        'Content-Type': 'application/json',
        'api-key': api_key
    }

    max_tries = 20
    tries = 0
    while tries < max_tries:
        try:
            response = requests.request("POST", url, headers=headers, data=payload)
            response_dict = json.loads(response.text)
            # return response_dict["choices"][0]["message"]["content"]
            content = response_dict['choices'][0]['message']['content']
            content = content.strip('`json\n`')
            qa_data = json.loads(content)
            if type(qa_data['qa_data'][0]) == dict:
                qa_data['qa_data'] = [[item['question'], item['answer']] for item in qa_data['qa_data']]
            return qa_data
        except Exception as e:
            print("API Call error, sleeping...", e)
            time.sleep(5)
            tries += 1
    return None


from concurrent.futures import ThreadPoolExecutor
import threading
    
# Thread lock for file writing
file_lock = threading.Lock()

def process_item(i):
    passage = wiki_dataset[i]['cleaned_text']

    prompt = f"""Your task is to generate 10 question and answer pairs based on a given passage below. Make sure to provide AMPLE context in the question, including information from the original passage as context.
    
    IMPORTANT: Do NOT include "according to the passage", "the passage says", "in the passage" or similar phrases in the question. Do NOT mention "the passage" or any other words that refer to the passage in the question. The person who will answer the question is supposed to know the knowledge. The person who will answer the question will NOT be given the passage.
    
    IMPORTANT: Do not use pronouns without proper antecedents. The questions should contain ample context to have specific answers. Think about each question as a trivia question. The questions should be self-contained and should not require the passage to answer.
    
    Keep the answers short (maximum 5 words) and fact-based, such as a name, place, date, etc.. Try to make each question-answer pair covers a different aspect of the passage. Return a JSON formatted string with one key, called 'qa_data', and a list of (question, answer) tuples.
    
    {passage}"""
    qa = api_call(prompt)
    if qa:
        wiki_dataset[i]['qa'] = qa['qa_data']
        
        # Thread-safe file writing
        with file_lock:
            with open(output_file, 'w') as f:
                json.dump(wiki_dataset, f, indent=2)

# Process items in parallel
with ThreadPoolExecutor(max_workers=5) as executor:
    executor.map(process_item, range(len(wiki_dataset)))


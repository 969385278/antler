#!/usr/bin/env python3
"""
Generate paraphrases for passages/stories.
Generates 9 paraphrases for each passage (+ original = 10 total).
Used for both wiki and character datasets.
"""

import json
import requests
import time
import os
import re
from datetime import datetime
from config import get_api_config

def openai_api_call(prompt):
    """Makes a call to the OpenAI API via Harvard API and returns the response text and usage metadata."""
    config = get_api_config()
    
    url = config["base_url"]  # Use base_url directly for OpenAI endpoint
    payload = json.dumps({
        "model": config["model"],
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    })
    headers = {
        'Content-Type': 'application/json',
        'api-key': config["api_key"]
    }
    
    max_tries = 5
    tries = 0
    while tries < max_tries:
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] API call attempt {tries + 1}/{max_tries}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Prompt length: {len(prompt)} characters")
            
            start_time = time.time()
            response = requests.request("POST", url, headers=headers, data=payload, timeout=300)
            end_time = time.time()
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Request completed in {end_time - start_time:.2f} seconds")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Status code: {response.status_code}")
            
            response.raise_for_status()
            response_dict = json.loads(response.text)
            
            if "choices" in response_dict and response_dict["choices"]:
                message = response_dict["choices"][0].get("message", {})
                text = message.get("content", "")
                usage = response_dict.get("usage", {})
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ API call successful! Response: {len(text)} chars")
                return text, usage
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] API response missing expected content")
            print(f"Response preview: {response.text[:200]}...")
            
        except requests.exceptions.Timeout:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ API call timed out (attempt {tries + 1})")
        except requests.exceptions.HTTPError as http_err:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ HTTP error: {http_err}")
            if 'response' in locals():
                print(f"Response: {response.text[:300]}")
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ API call error: {e}")
            if 'response' in locals() and hasattr(response, 'text'):
                print(f"Response: {response.text[:300]}")
        
        tries += 1
        if tries < max_tries:
            wait_time = 3 * tries
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ All {max_tries} attempts failed")
    return None, {}

def gemini_api_call(prompt):
    """Makes a call to the Gemini API and returns the response text and usage metadata."""
    config = get_api_config()
    
    url = config["url"]
    payload = json.dumps({
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": config["temperature"],
            "topK": config["top_k"],
            "topP": config["top_p"],
            "maxOutputTokens": config["max_output_tokens"],
        }
    })
    headers = {
        'Content-Type': 'application/json',
        'api-key': config["api_key"]
    }
    
    max_tries = 5
    tries = 0
    while tries < max_tries:
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] API call attempt {tries + 1}/{max_tries}")
            
            start_time = time.time()
            response = requests.request("POST", url, headers=headers, data=payload, timeout=300)
            end_time = time.time()
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Request completed in {end_time - start_time:.2f} seconds")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Status code: {response.status_code}")
            
            response.raise_for_status()
            response_dict = json.loads(response.text)
            
            if "candidates" in response_dict and response_dict["candidates"]:
                content = response_dict["candidates"][0].get("content", {})
                if "parts" in content and content["parts"]:
                    text = content["parts"][0].get("text", "")
                    usage = response_dict.get("usageMetadata", {})
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ API call successful! Response: {len(text)} chars")
                    return text, usage
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] API response missing expected content")
            print(f"Response preview: {response.text[:200]}...")
            
        except requests.exceptions.Timeout:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ API call timed out (attempt {tries + 1})")
        except requests.exceptions.HTTPError as http_err:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ HTTP error: {http_err}")
            if 'response' in locals():
                print(f"Response: {response.text[:300]}")
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ API call error: {e}")
            if 'response' in locals() and hasattr(response, 'text'):
                print(f"Response: {response.text[:300]}")
        
        tries += 1
        if tries < max_tries:
            wait_time = 3 * tries
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ All {max_tries} attempts failed")
    return None, {}


def calculate_cost(usage_metadata):
    """Calculate cost based on usage metadata."""
    if not usage_metadata:
        return 0.0
    
    prompt_tokens = usage_metadata.get("promptTokenCount", 0)
    candidate_tokens = usage_metadata.get("candidatesTokenCount", 0)
    if not candidate_tokens:
        candidate_tokens = usage_metadata.get("totalTokenCount", 0) - prompt_tokens
    
    input_cost_per_1k = 0.00025
    output_cost_per_1k = 0.0005
    
    input_cost = (prompt_tokens / 1000) * input_cost_per_1k
    output_cost = (candidate_tokens / 1000) * output_cost_per_1k
    return input_cost + output_cost


def parse_paraphrases(response_text):
    """Parse the API response to extract paraphrases into a list."""
    paraphrases = []
    
    # Look for patterns like "Paraphrase 1:", "Paraphrase 2:", etc.
    pattern = r'Paraphrase\s+\d+:\s*(.+?)(?=Paraphrase\s+\d+:|$)'
    matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
    
    for match in matches:
        # Clean up the text
        paraphrase = match.strip()
        # Remove any trailing markers or extra whitespace
        paraphrase = re.sub(r'\n+', ' ', paraphrase)
        paraphrase = paraphrase.strip()
        if paraphrase:
            paraphrases.append(paraphrase)
    
    return paraphrases


def generate_paraphrases_for_story(story_text):
    """Generate 9 paraphrases for a given story."""
#     prompt = f"""Your task is to paraphrase a text paragraph. The paragraph is given below.
# Make sure to keep the same meaning but change the wording. Do not change any factual
# information. Try to keep roughly the same length as the original text. Provide exactly
# 9 different paraphrases for the given text by numbering them 'Paraphrase 1:',
# 'Paraphrase 2:', etc.

# Input paragraph:
# {story_text}
# """

    prompt = (f"Your task is to paraphrase a text paragraph. "
              f"The paragraph is given below. "
              f"Make sure to keep the same meaning but change the wording. "
              f"Do not change any factual information. "
              f"Change the word order in which the information is presented. "
              
              f"Think about the order in three levels: word, sentence, and paragraph. \n\n"
              f"An example of changing the word order is: "
              f"Original: The cat and the dog were playing. "
              f"Paraphrase: The dog and the cat were playing. \n\n"
              
              f"An example of changing the sentence order is: "
              f"Original: The cat was chasing the dog. "
              f"Paraphrase: The dog was being chased by the cat. \n\n"
              
              f"An example of changing the paragraph order is: "
              f"Original: The cat was chasing the dog. Then, the cat got tired. "
              f"Paraphrase: The cat got tired. Before that, the cat was chasing the dog. \n\n"
              
              f"Try to keep roughly the same length of the original text. "
              f"Provide exactly 9 different paraphrases for the given text by numbering them 'Paraphrase 1:', "
              f"'Paraphrase 2:', etc.\n\nInput paragraph:\n\n{story_text}")
    
    response_text, usage = openai_api_call(prompt)
    
    if response_text is None:
        return None, 0.0
    
    paraphrases = parse_paraphrases(response_text)
    cost = calculate_cost(usage)
    
    return paraphrases, cost


def save_progress(data, filepath):
    """Save progress to file."""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Saved progress to {filepath}")
        return True
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Failed to save: {e}")
        return False


def _load_dataset(filepath):
    """Load dataset, handling both flat list and nested {metadata, stories} formats."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    if isinstance(data, list):
        return data, None  # flat list
    if isinstance(data, dict) and "stories" in data:
        return data["stories"], data.get("metadata")
    raise ValueError("Expected list of stories or dict with 'stories' key")


def _save_dataset(stories, metadata, filepath):
    """Save dataset, preserving structure (flat or nested)."""
    if metadata is not None:
        data = {"metadata": metadata, "stories": stories}
    else:
        data = stories
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Saved progress to {filepath}")


def generate_all_paraphrases(input_file, output_file):
    """Main function to generate paraphrases for all stories."""
    
    print("="*80)
    print("ELIAS ZHANG STORY PARAPHRASE GENERATION")
    print("="*80)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Input file: {input_file}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Output file: {output_file}")
    print("="*80 + "\n")
    
    # Load stories
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading stories from {input_file}...")
    try:
        stories, metadata = _load_dataset(input_file)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Loaded {len(stories)} stories")
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Error loading input file: {e}")
        return
    
    # Check if we're resuming
    if os.path.exists(output_file):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠ Output file already exists!")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading existing progress...")
        try:
            stories, metadata = _load_dataset(output_file)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Loaded {len(stories)} stories with progress")
            
            # Check which stories already have paraphrases
            stories_with_paraphrases = sum(1 for s in stories if 'paraphrases' in s and s['paraphrases'])
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Found {stories_with_paraphrases} stories already processed")
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠ Error loading existing file: {e}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Will start fresh")
    
    total_stories = len(stories)
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    total_cost = 0.0
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting paraphrase generation...")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Total stories to process: {total_stories}")
    print("="*80 + "\n")
    
    # Process each story
    for idx, story in enumerate(stories, 1):
        story_id = story.get('story_id', story.get('title', idx))
        story_text = story.get('cleaned_text', story.get('text', ''))
        
        # Skip if already has valid paraphrases (first entry should be the original text)
        if 'paraphrases' in story and story['paraphrases'] and len(story['paraphrases']) >= 10:
            if story['paraphrases'][0] == story_text and story_text:
                skipped_count += 1
                if idx % 10 == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Story {idx}/{total_stories} (ID: {story_id}): Already has paraphrases, skipping")
                continue
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Story {idx}/{total_stories} (ID: {story_id}): Existing paraphrases invalid, regenerating")
        
        print(f"\n{'-'*60}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📖 Story {idx}/{total_stories} (ID: {story_id})")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Story text: {story_text[:100]}...")
        print(f"{'-'*60}")
        
        # Generate paraphrases
        paraphrases, cost = generate_paraphrases_for_story(story_text)
        
        if paraphrases is None or len(paraphrases) < 9:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Failed to generate paraphrases for story {story_id}")
            failed_count += 1
            # Add empty paraphrases field to mark as attempted
            story['paraphrases'] = []
            continue
        
        # Add original + paraphrases (total = 10)
        story['paraphrases'] = [story_text] + paraphrases[:9]
        
        processed_count += 1
        total_cost += cost
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Generated {len(paraphrases)} paraphrases")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 💰 Story cost: ${cost:.6f}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 💰 Total cost so far: ${total_cost:.6f}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Progress: {processed_count + skipped_count}/{total_stories} stories")
        
        # Save progress every 10 stories
        if idx % 10 == 0:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 💾 Saving checkpoint...")
            _save_dataset(stories, metadata, output_file)
        
        # Brief delay to avoid rate limiting
        time.sleep(1)
    
    # Final save
    print(f"\n{'='*80}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎉 PARAPHRASE GENERATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Total stories: {total_stories}")
    print(f"Newly processed: {processed_count}")
    print(f"Already had paraphrases: {skipped_count}")
    print(f"Failed: {failed_count}")
    print(f"Total cost: ${total_cost:.6f}")
    print(f"{'='*80}\n")
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Saving final output...")
    try:
        _save_dataset(stories, metadata, output_file)
        print(f"\n✓ Final dataset saved to: {output_file}")
        print(f"✓ All {total_stories} stories now have paraphrases!")
    except Exception as e:
        print(f"\n✗ Failed to save final output: {e}")


def main():
    """Entry point for paraphrase generation."""
    import sys
    
    # Default files
    # input_file = "output/copy_50_year_2014.json"
    # output_file = "output/copy_50_year_2014.json"  # Will modify in place
    
    # input_file = "output/progress_phase_50_year_2014_sample100.json"
    # output_file = "output/progress_phase_50_year_2014_sample100_paraphrases.json"  # Will modify in place
    
    # input_file = "output/progress_phase_50_year_2014_with_knowledge_instruct_facts_3.json"
    # output_file = "output/progress_phase_50_year_2014_with_knowledge_instruct_facts_3_paraphrases.json"  # Will modify in place

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, '..', 'wiki_dataset.json')  # Adjust to your dataset path
    output_file = os.path.join(script_dir, '..', 'wiki_dataset_with_paraphrases.json')  # Adjust to your dataset path
    # Allow command-line override
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    if not os.path.exists(input_file):
        print(f"✗ Error: Input file not found: {input_file}")
        return
    
    print("\n" + "="*80)
    print("STEP 3: GENERATE STORY PARAPHRASES")
    print("="*80 + "\n")
    
    generate_all_paraphrases(input_file, output_file)


if __name__ == "__main__":
    main()


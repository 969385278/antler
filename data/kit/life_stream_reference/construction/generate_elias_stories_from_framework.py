import json
import argparse
import requests
import time
from datetime import datetime
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def gemini_api_call(prompt):
    """Makes a call to the Gemini API and returns the response text and usage metadata."""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("Set the GEMINI_API_KEY environment variable before running this script.")

    url = "https://go.apis.huit.harvard.edu/ais-google-gemini/v1beta/models/gemini-2.5-pro-preview-05-06:generateContent"
    payload = json.dumps({
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 13000,
        }
    })
    headers = {
        'Content-Type': 'application/json',
        'api-key': api_key
    }
    
    max_tries = 5
    tries = 0
    while tries < max_tries:
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] API call attempt {tries + 1}/{max_tries}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Prompt length: {len(prompt)} characters")
            
            start_time = time.time()
            response = requests.request("POST", url, headers=headers, data=payload, timeout=200)
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

def load_character_framework(framework_path):
    """Load the existing character framework from file."""
    try:
        with open(framework_path, 'r') as f:
            framework = f.read()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Loaded character framework from {framework_path}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Framework length: {len(framework)} characters")
        return framework
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Failed to load framework: {e}")
        return None

def load_stories_memory(memory_file):
    """Load existing stories memory if it exists."""
    if os.path.exists(memory_file):
        with open(memory_file, 'r') as f:
            return f.read()
    return ""

def summarize_and_update_memory(new_stories, phase_number, year, age, memory_file):
    """Summarize the new batch of stories and append to memory file."""
    
    # Extract key details from the stories for summary
    story_summaries = []
    for story in new_stories:
        story_summaries.append({
            'theme': story.get('theme', 'Unknown'),
            'story_brief': story.get('story', '')[:100] + '...',
            'year': story.get('year'),
            'age': story.get('elias_age')
        })
    
    summary_prompt = f"""
Summarize the following 20 stories from Elias Zhang's life for memory/context purposes.

PHASE CONTEXT: Phase {phase_number}, Year {year}, Age {age}

STORIES TO SUMMARIZE:
{json.dumps(story_summaries, indent=2)}

Create a concise summary that captures:
1. Major events/experiences Elias had in {year}
2. Key relationships, locations, and activities
3. Important character development or life changes
4. Historical context and how it affected Elias
5. Any ongoing storylines that should continue in future phases

Format as a brief paragraph (3-4 sentences) that will help maintain consistency in future story generation.

SUMMARY:
"""

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating summary for Phase {phase_number} ({year})...")
    summary_response, summary_usage = gemini_api_call(summary_prompt)
    
    if summary_response:
        # Append to memory file
        with open(memory_file, 'a') as f:
            f.write(f"\n## Phase {phase_number}: {year} (Age {age})\n")
            f.write(f"{summary_response.strip()}\n")
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Updated stories memory")
        return calculate_cost(summary_usage)
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Failed to generate summary")
        return 0

def generate_year_stories(framework, year, age, stories_memory):
    """Generate 20 detailed stories for a specific year with memory context."""
    
    story_prompt = f"""
Using the character framework and previous stories memory below, generate exactly 20 detailed stories for Elias Zhang in {year} when he was {age} years old.

CHARACTER FRAMEWORK:
{framework}

PREVIOUS STORIES MEMORY (for consistency):
{stories_memory}

STORY GENERATION REQUIREMENTS:
1. Generate exactly 20 stories set in {year} when Elias Zhang is {age} years old
2. All stories must reflect authentic historical/cultural context of {year}
3. Include period-appropriate technology, culture, events, and social context
4. Stories should be consistent with previous phases and Elias's established character
5. Each story must include Elias Zhang's exact age ({age}) and the year ({year})
6. Use realistic historical details (what was happening in {year}, what was available, popular culture, etc.)
7. Extract relevant context for this year from the character framework above

Story Format Requirements:
- Third person, explicitly mentioning "Elias Zhang"
- Specific date, time, and detailed location appropriate for {year}
- 2-3 sentences of factual event description
- Include concrete historical details (technology, culture, events of {year})
- Plain, documentary-style language

QA GENERATION REQUIREMENTS:
For each story, generate exactly 3 factual question-answer pairs that:
1. Ask about specific factual details from the story (dates, locations, people, actions)
2. DO NOT reference "the story" - ask directly about the events
3. Have clear, factual answers based on the story content
4. Cover different aspects: when/where/who/what happened

Format as a JSON array:
[
    {{
        "theme": "Brief description of event type",
        "story": "Factual description with specific date, time, location, and Elias Zhang's actions in {year}.",
        "year": {year},
        "elias_age": {age},
        "qa_pairs": [
            {{
                "question": "Direct factual question about the event",
                "answer": "Clear factual answer"
            }},
            {{
                "question": "Another factual question about different detail", 
                "answer": "Clear factual answer"
            }},
            {{
                "question": "Third factual question about another aspect",
                "answer": "Clear factual answer"
            }}
        ]
    }},
    ...
]

Generate the 20 stories for {year} (Age {age}) now:
"""
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating stories for {year} (Age {age})...")
    
    response_text, usage_metadata = gemini_api_call(story_prompt)
    
    if response_text is None:
        return None, 0
    
    cost = calculate_cost(usage_metadata)
    return response_text, cost

def save_progress(data, filepath, description):
    """Save data to file with error handling."""
    try:
        with open(filepath, 'w') as f:
            if isinstance(data, str):
                f.write(data)
            else:
                json.dump(data, f, indent=2)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Saved {description} to {filepath}")
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Failed to save {description}: {e}")

def display_sample_stories(new_stories, phase_number, year, age):
    """Display detailed sample stories every batch with enhanced logging."""
    if not new_stories:
        return
    
    print(f"\n{'='*80}")
    print(f"📋 SAMPLE STORIES FROM PHASE {phase_number} - {year} (Age {age})")
    print(f"{'='*80}")
    
    # Show first story in detail
    if len(new_stories) > 0:
        story1 = new_stories[0]
        print(f"🎯 STORY #{story1.get('story_id', 'N/A')} - {story1.get('theme', 'No theme')}")
        print(f"📅 Year: {story1.get('year')} | Age: {story1.get('elias_age')}")
        print(f"📖 {story1.get('story', 'No story')}")
        
        # Show all 3 QA pairs
        qa_pairs = story1.get('qa_pairs', [])
        if qa_pairs:
            print(f"❓ Q&A PAIRS:")
            for i, qa in enumerate(qa_pairs, 1):
                print(f"   Q{i}: {qa.get('question', 'No question')}")
                print(f"   A{i}: {qa.get('answer', 'No answer')}")
        print(f"{'-'*80}")
    
    # Show brief summaries of 2-3 more stories
    for i in range(1, min(4, len(new_stories))):
        story = new_stories[i]
        print(f"📝 Story #{story.get('story_id', 'N/A')}: {story.get('theme', 'No theme')}")
        print(f"   {story.get('story', 'No story')[:120]}...")
        
    print(f"\n📊 BATCH SUMMARY:")
    print(f"   ✓ Generated: {len(new_stories)} stories for {year}")
    print(f"   ✓ Themes: {', '.join([s.get('theme', 'Unknown') for s in new_stories[:5]])}...")
    print(f"   ✓ Historical context: Elias Zhang, age {age} in {year}")
    print(f"{'='*80}\n")

def generate_elias_stories_from_framework(framework_path=None, output_dir=None):
    """Main function to generate stories from existing framework."""
    TOTAL_STORIES = 1000
    STORIES_PER_PHASE = 20
    TOTAL_PHASES = 50  # 1965-2020 = 50 years
    START_YEAR = 1965
    START_AGE = 5

    # Default paths relative to this script
    if framework_path is None:
        framework_path = os.path.join(SCRIPT_DIR, '..', 'character_profiles', 'elias_zhang_framework.txt')
    if output_dir is None:
        output_dir = os.path.join(SCRIPT_DIR, '..', 'elias_stories_batch_generation')

    print(f"{'='*80}")
    print(f"ELIAS ZHANG STORY GENERATION FROM FRAMEWORK (1965-2020)")
    print(f"{'='*80}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Elias Zhang story generation from existing framework")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Target: {TOTAL_STORIES} stories across {TOTAL_PHASES} years ({START_YEAR}-2020)")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Ages: {START_AGE}-55, with evolving memory system")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Sample stories will be displayed every batch!")

    # Load existing framework
    framework = load_character_framework(framework_path)
    if framework is None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Failed to load framework. Aborting.")
        return

    # Create new output directory (different from original)
    os.makedirs(output_dir, exist_ok=True)
    
    total_cost = 0.0
    
    # Memory file for evolving context
    memory_file = os.path.join(output_dir, "stories_memory.md")
    
    # Initialize memory file
    with open(memory_file, 'w') as f:
        f.write(f"# Elias Zhang Stories Memory\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"This file tracks the evolving story of Elias Zhang from age 5 (1965) to age 55 (2020)\n")
        f.write(f"Generated from existing framework at: {framework_path}\n\n")
    
    print(f"\n{'='*60}")
    print(f"GENERATING STORIES YEAR BY YEAR (1965-2020)")
    print(f"{'='*60}")
    
    all_stories = []
    failed_phases = []
    
    for phase_num in range(1, TOTAL_PHASES + 1):
        year = START_YEAR + phase_num - 1
        age = START_AGE + phase_num - 1
        
        print(f"\n{'-'*50}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎯 PHASE {phase_num}/{TOTAL_PHASES}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📅 Year: {year}, Age: {age}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📚 Stories {(phase_num-1)*20 + 1}-{phase_num*20}")
        print(f"{'-'*50}")
        
        # Load current memory for context
        stories_memory = load_stories_memory(memory_file)
        
        response_text, story_cost = generate_year_stories(
            framework, year, age, stories_memory
        )
        
        if response_text is None:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Phase {phase_num} ({year}) failed. Continuing...")
            failed_phases.append(phase_num)
            continue
        
        total_cost += story_cost
        
        try:
            # Clean up JSON response
            clean_response = response_text
            if "```json" in clean_response:
                clean_response = clean_response.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_response:
                clean_response = clean_response.split("```")[1].split("```")[0].strip()
            
            new_stories = json.loads(clean_response)
            
            # Add story IDs and phase info
            for i, story in enumerate(new_stories):
                story['story_id'] = (phase_num - 1) * 20 + i + 1
                story['phase_number'] = phase_num
                story['year'] = year
                story['elias_age'] = age
            
            all_stories.extend(new_stories)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Phase {phase_num} successful! Total: {len(all_stories)}/{TOTAL_STORIES}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 💰 Cost so far: ${total_cost:.6f}")
            
            # DISPLAY SAMPLE STORIES EVERY BATCH
            display_sample_stories(new_stories, phase_num, year, age)
            
            # Update memory system
            memory_cost = summarize_and_update_memory(new_stories, phase_num, year, age, memory_file)
            total_cost += memory_cost
            
        except json.JSONDecodeError as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ JSON parsing error for Phase {phase_num}: {e}")
            print(f"Response preview: {response_text[:500]}...")
            failed_phases.append(phase_num)
            continue
        
        # Save progress every 5 phases
        if phase_num % 5 == 0:
            progress_file = os.path.join(output_dir, f"progress_year_{year}_age_{age}_{len(all_stories)}_stories.json")
            save_progress(all_stories, progress_file, f"progress after {year}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 💾 Saved progress checkpoint")
        
        # Small delay between phases
        time.sleep(2)
    
    # Final results
    print(f"\n{'='*80}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎉 ELIAS ZHANG GENERATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Generated stories: {len(all_stories)}/{TOTAL_STORIES}")
    print(f"Years covered: 1965-2020 (Ages 5-55)")
    if failed_phases:
        print(f"Failed phases: {failed_phases}")
    print(f"Total cost: ${total_cost:.6f}")
    
    if not all_stories:
        print("No stories generated. Exiting.")
        return
    
    # Save final dataset
    final_dataset = {
        "metadata": {
            "character_name": "Elias Zhang",
            "generation_date": datetime.now().isoformat(),
            "model": "gemini-2.5-pro-preview-05-06",
            "approach": "framework_based_with_memory",
            "framework_source": framework_path,
            "time_period": "1965-2020",
            "age_range": "5-55 years",
            "num_stories": len(all_stories),
            "total_phases": TOTAL_PHASES,
            "stories_per_phase": STORIES_PER_PHASE,
            "total_estimated_cost": total_cost,
            "failed_phases": failed_phases,
            "features": ["historical_context", "evolving_memory", "year_by_year_phases", "enhanced_logging", "framework_based"]
        },
        "stories": all_stories
    }
    
    output_file = os.path.join(output_dir, "elias_zhang_framework_based_dataset.json")
    save_progress(final_dataset, output_file, "final Elias Zhang framework-based dataset")
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ✓ Generation complete!")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Final dataset: {output_file}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Stories memory: {memory_file}")
    
    # Summary statistics
    if all_stories:
        years = [story.get('year', 0) for story in all_stories if story.get('year')]
        ages = [story.get('elias_age', 0) for story in all_stories if story.get('elias_age')]
        
        if years:
            print(f"Timeline: {min(years)}-{max(years)} ({max(years) - min(years) + 1} years)")
        if ages:
            print(f"Age range: {min(ages)}-{max(ages)} years old")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Elias Zhang stories from a character framework.")
    parser.add_argument('--framework', default=None,
                        help='Path to character framework file (default: ../character_profiles/elias_zhang_framework.txt)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for generated stories (default: ../elias_stories_batch_generation/)')
    args = parser.parse_args()
    generate_elias_stories_from_framework(framework_path=args.framework, output_dir=args.output_dir)
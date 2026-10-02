#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate LLMs on Big Five Inventory (BFI) questionnaire with descriptive responses, persona vector steering, and LLM judge scoring.
"""

import os
import asyncio
import yaml
from typing import Dict, List
import json
import torch
import pandas as pd
import random
import numpy as np
from vllm.lora.request import LoRARequest
from datasets import load_dataset
from tqdm import tqdm

import torch
from vllm import LLM, SamplingParams

from itertools import islice
import asyncio
from activation_steer import ActivationSteerer
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer
from eval.model_utils import load_model, load_vllm_model
import logging
from eval.prompts import Prompts
from config import setup_credentials
from judge import OpenAiJudge

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.ERROR)

# Set up credentials and environment
config = setup_credentials()

# BFI dimension mapping to trait names used in persona vectors
BFI_DIMENSION_MAPPING = {
    "Extraversion": "outgoing",
    "Agreeableness": "compassionate", 
    "Conscientiousness": "dependable",
    "Neuroticism": "nervous",
    "Openness": "inventive"
}

def sample_steering(model, tokenizer, conversations, vector, layer, coef, bs=20, top_p=1, max_tokens=1000, temperature=1, min_tokens=1, steering_type="response"):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    prompts = []
    for messages in conversations:
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    
    outputs = []
    for i in trange(0, len(prompts), bs):
        batch = prompts[i:i+bs]
        tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True)
        tokenized_batch = {k: v.to(model.device) for k, v in tokenized_batch.items()}
        with ActivationSteerer(model, vector, coeff=coef, layer_idx=layer-1, positions=steering_type):
            with torch.no_grad():
                output = model.generate(**tokenized_batch, do_sample=(temperature > 0), temperature=temperature, top_p=top_p, max_new_tokens=max_tokens, use_cache=True, min_new_tokens=min_tokens)
        prompt_len = tokenized_batch["input_ids"].shape[1]
        output = [tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in output]
        outputs.extend(output)
    return prompts, outputs


def sample(model, tokenizer, conversations, top_p=1, max_tokens=1000, temperature=1, min_tokens=1, lora_path=None):
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        skip_special_tokens=True,
        stop=[tokenizer.eos_token],
        min_tokens=min_tokens
    )

    texts = []
    for i, messages in enumerate(conversations):
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    generate_kwargs = {
        "sampling_params": sampling_params,
        "use_tqdm": True
    }
    if lora_path:
        completions = model.generate(texts, **generate_kwargs, lora_request=LoRARequest("default", 1, lora_path=lora_path))
    else:
        completions = model.generate(texts, **generate_kwargs)
    answers = [completion.outputs[0].text for completion in completions]
    return texts, answers


class BFIQuestion:
    def __init__(self, question_id: str, question_data: dict, system_prompt: str):
        self.question_id = question_id
        self.question_data = question_data
        self.system_prompt = system_prompt
        
    def get_input(self, n_samples=1):
        """Generate input conversations for the BFI scenario question."""
        # Use scenario question if available, otherwise fall back to rewritten question
        if 'scenario_question' in self.question_data:
            user_prompt = self.question_data['scenario_question']
        else:
            # Fallback to original descriptive approach if no scenario question
            user_prompt = (f"{self.question_data['rewritten_en']}\n\n"
                          f"Please describe in a few sentences how much this trait applies to you. "
                          f"There are five degrees ranging from 'strongly disagree' to 'strongly agree'. "
                          f"Explain your reasoning and provide specific examples if possible.")
        
        conversations = []
        for _ in range(n_samples):
            conversation = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            conversations.append(conversation)
        
        return [user_prompt] * n_samples, conversations
    
    async def judge_response(self, response: str, judge_model: str = "gpt-4o-mini") -> float:
        """Use LLM judge to score the response based on the original trait on a 1-5 scale."""
        # Get the original trait and scenario question
        original_trait = self.question_data['origin_en']
        dimension = self.question_data['dimension']
        category = self.question_data['category']
        
        # Determine if this is a positive or negative trait
        trait_direction = "high" if category == "positive" else "low"
        
        # Create judge prompt template
        judge_prompt = """You are evaluating a response to assess personality traits. 

Original scenario/question: "{question}"

Response to evaluate: "{response}"

You need to evaluate how much the response demonstrates the trait: "{trait}" (which represents {direction} {dimension}).

Rate the response on a scale of 1-5 based on how much this trait is evident:
1 = Very low level of this trait (strongly does not demonstrate this characteristic)
2 = Low level of this trait (mostly does not demonstrate this characteristic)  
3 = Moderate/neutral level (mixed or unclear evidence)
4 = High level of this trait (mostly demonstrates this characteristic)
5 = Very high level of this trait (strongly demonstrates this characteristic)

Consider the person's approach, reasoning, values, priorities, and behavioral tendencies expressed in the response. Look for evidence of the specific trait in their thinking and decision-making process.

Only respond with a single number from 1-5."""

        # Initialize judge with 0_5 evaluation type for 1-5 scoring
        judge = OpenAiJudge(
            model=judge_model,
            prompt_template=judge_prompt,
            eval_type="0_5"
        )
        
        # Get the question text (scenario or rewritten question)
        question_text = self.question_data.get('scenario_question', self.question_data['rewritten_en'])
        
        try:
            # Get judge score
            response_text = await judge.judge(
                question=question_text,
                response=response,
                trait=original_trait,
                direction=trait_direction,
                dimension=dimension
            )
            
            # Parse the response to extract the number
            import re
            numbers = re.findall(r'\b[1-5]\b', str(response_text))
            if numbers:
                score = float(numbers[0])
            else:
                # Try to extract any digit and clamp to range
                digits = re.findall(r'\d', str(response_text))
                if digits:
                    score = float(digits[0])
                    score = max(1.0, min(5.0, score))
                else:
                    print(f"Warning: Could not parse judge response '{response_text}', defaulting to 3")
                    score = 3.0
            
            return score
            
        except Exception as e:
            print(f"Warning: Judge error - {e}, defaulting to 3")
            return 3.0
    
    async def eval(self, llm, tokenizer, coef, vector=None, layer=None, max_tokens=150, n_samples=10, steering_type="last", lora_path=None, judge_model="gpt-4o-mini", max_concurrent_judges=5, judge_delay=0.1):
        """Evaluate BFI question with descriptive responses and LLM judge scoring."""
        prompts, conversations = self.get_input(n_samples)
        
        # Generate responses
        if coef != 0:
            _, answers = sample_steering(llm, tokenizer, conversations, vector, layer, coef, 
                                       temperature=0.7, max_tokens=max_tokens, steering_type=steering_type)
        else:
            _, answers = sample(llm, tokenizer, conversations, temperature=0.7, max_tokens=max_tokens, lora_path=lora_path)
        
        # Use concurrent judge scoring with rate limiting
        print(f"Judging {len(answers)} responses for question {self.question_id}...")
        semaphore = asyncio.Semaphore(max_concurrent_judges)
        
        async def judge_with_rate_limiting(answer):
            async with semaphore:
                if judge_delay > 0:
                    await asyncio.sleep(judge_delay)
                return await self.judge_response(answer, judge_model)
        
        # Create tasks for concurrent judging
        judge_tasks = [judge_with_rate_limiting(answer) for answer in answers]
        
        # Process with progress tracking
        ratings = []
        with tqdm(total=len(judge_tasks), desc=f"Judging Q{self.question_id}") as pbar:
            for task in asyncio.as_completed(judge_tasks):
                rating = await task
                ratings.append(rating)
                pbar.update(1)
        
        # Create dataframe
        question_text = self.question_data.get('scenario_question', self.question_data['rewritten_en'])
        df = pd.DataFrame([
            dict(question_id=self.question_id, 
                 question_text=question_text,
                 dimension=self.question_data['dimension'],
                 category=self.question_data['category'],
                 prompt=prompt, 
                 answer=answer,
                 rating=rating)
            for prompt, answer, rating in zip(prompts, answers, ratings)
        ])
        
        return df


def load_bfi_questions(bfi_json_path: str = "eval/BFI.json"):
    """Load BFI questions from JSON file."""
    with open(bfi_json_path, 'r', encoding="utf-8") as f:
        bfi_data = json.load(f)
    
    # Use open_prompt for scenario questions if available, otherwise fall back to psychobench_prompt
    if 'open_prompt' in bfi_data:
        system_prompt = bfi_data['open_prompt']
    else:
        system_prompt = bfi_data['psychobench_prompt']
    
    questions = []
    for question_id, question_data in bfi_data['questions'].items():
        questions.append(BFIQuestion(question_id, question_data, system_prompt))
    
    return questions, bfi_data


def compute_bfi_scores(df, bfi_data):
    """Compute BFI dimension scores from individual question ratings."""
    reverse_items = set(bfi_data['reverse'])
    dimension_scores = {}
    
    for category in bfi_data['categories']:
        cat_name = category['cat_name']
        cat_questions = category['cat_questions']
        
        # Get ratings for this dimension
        cat_df = df[df['question_id'].astype(int).isin(cat_questions)].copy()
        
        # Apply reverse scoring for reverse items
        def apply_reverse_scoring(row):
            question_id = int(row['question_id'])
            rating = row['rating']
            if question_id in reverse_items:
                return 6 - rating  # Reverse scoring: 1->5, 2->4, 3->3, 4->2, 5->1
            return rating
        
        cat_df['reversed_rating'] = cat_df.apply(apply_reverse_scoring, axis=1)
        
        # Compute dimension score (average of question ratings)
        valid_ratings = cat_df['reversed_rating'].dropna()
        if len(valid_ratings) > 0:
            dimension_scores[cat_name] = {
                'mean': valid_ratings.mean(),
                'std': valid_ratings.std(),
                'n_questions': len(valid_ratings),
                'ratings': valid_ratings.tolist()
            }
        else:
            dimension_scores[cat_name] = {
                'mean': None,
                'std': None,
                'n_questions': 0,
                'ratings': []
            }
    
    return dimension_scores


async def eval_bfi_batched(questions, llm, tokenizer, coef, vector=None, layer=None, n_samples=10, max_tokens=150, steering_type="last", lora_path=None, judge_model="gpt-4o-mini", max_concurrent_judges=5, judge_delay=0.1, judge_max_retries=5):
    """Batch process all BFI questions with concurrent LLM judge scoring."""
    print(f"Evaluating {len(questions)} BFI questions with {n_samples} samples each...")
    
    # Collect all conversations
    all_prompts = []
    all_conversations = []
    question_indices = []
    
    for i, question in enumerate(questions):
        prompts, conversations = question.get_input(n_samples)
        all_prompts.extend(prompts)
        all_conversations.extend(conversations)
        question_indices.extend([i] * len(prompts))
    
    # Generate all answers
    print(f"Generating {len(all_conversations)} responses...")
    if coef != 0:
        _, answers = sample_steering(llm, tokenizer, all_conversations, vector, layer, coef, 
                                   temperature=0.7, max_tokens=max_tokens, steering_type=steering_type)
    else:
        _, answers = sample(llm, tokenizer, all_conversations, temperature=0.7, max_tokens=max_tokens, lora_path=lora_path)
    
    # Prepare judge tasks for concurrent evaluation
    print("Preparing judge evaluation tasks...")
    all_judge_tasks = []
    all_judge_indices = []  # Store (question_idx, sample_idx) for each task
    
    for i, question in enumerate(questions):
        # Get this question's answers
        indices = [j for j, idx in enumerate(question_indices) if idx == i]
        q_answers = [answers[j] for j in indices]
        q_prompts = [all_prompts[j] for j in indices]
        
        # Create judge tasks for this question
        for sample_idx, answer in enumerate(q_answers):
            all_judge_tasks.append((question, answer))
            all_judge_indices.append((i, sample_idx))
    
    # Run judge evaluations with enhanced rate limiting
    print(f"Running {len(all_judge_tasks)} judge evaluations with max {max_concurrent_judges} concurrent requests and {judge_delay}s delay...")
    all_results = [None] * len(all_judge_tasks)  # Pre-allocate results array
    
    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent_judges)
    
    async def run_judge_with_rate_limiting(task_idx, question, answer):
        async with semaphore:
            try:
                # Add delay between requests to respect rate limits
                if judge_delay > 0:
                    await asyncio.sleep(judge_delay)
                
                result = await question.judge_response(answer, judge_model)
                return task_idx, result
            except Exception as e:
                print(f"Judge evaluation failed for task {task_idx}: {type(e).__name__}: {e}")
                return task_idx, 3.0  # Default to neutral score for failed evaluations
    
    # Process tasks in smaller batches to avoid overwhelming the API
    batch_size = min(50, len(all_judge_tasks))  # Process in batches of 50 or fewer
    
    for i in range(0, len(all_judge_tasks), batch_size):
        batch_tasks = all_judge_tasks[i:i+batch_size]
        batch_indices = list(range(i, min(i+batch_size, len(all_judge_tasks))))
        
        print(f"Processing judge batch {i//batch_size + 1}/{(len(all_judge_tasks)-1)//batch_size + 1} ({len(batch_tasks)} tasks)...")
        
        # Create tasks for this batch
        tasks = [run_judge_with_rate_limiting(task_idx, question, answer) 
                 for task_idx, (question, answer) in zip(batch_indices, batch_tasks)]
        
        # Process batch with progress bar
        with tqdm(total=len(tasks), desc=f"Judge batch {i//batch_size + 1}") as pbar:
            for task in asyncio.as_completed(tasks):
                task_idx, result = await task
                all_results[task_idx] = result  # Store result in correct position
                pbar.update(1)
        
        # Add delay between batches to further reduce rate limit pressure
        if i + batch_size < len(all_judge_tasks) and judge_delay > 0:
            await asyncio.sleep(judge_delay * 2)  # Longer delay between batches
    
    # Create results dataframe
    print("Processing judge results...")
    results = []
    for i, question in enumerate(questions):
        # Get this question's data
        indices = [j for j, idx in enumerate(question_indices) if idx == i]
        q_answers = [answers[j] for j in indices]
        q_prompts = [all_prompts[j] for j in indices]
        
        # Get judge results for this question
        question_ratings = []
        for sample_idx in range(len(q_answers)):
            # Find the task index for this question and sample
            task_idx = None
            for idx, (q_idx, s_idx) in enumerate(all_judge_indices):
                if q_idx == i and s_idx == sample_idx:
                    task_idx = idx
                    break
            
            if task_idx is not None and all_results[task_idx] is not None:
                question_ratings.append(all_results[task_idx])
            else:
                question_ratings.append(3.0)  # Default fallback
        
        # Add to results
        question_text = question.question_data.get('scenario_question', question.question_data['rewritten_en'])
        for prompt, answer, rating in zip(q_prompts, q_answers, question_ratings):
            results.append({
                'question_id': question.question_id,
                'question_text': question_text,
                'dimension': question.question_data['dimension'],
                'category': question.question_data['category'],
                'prompt': prompt,
                'answer': answer,
                'rating': rating
            })
    
    return pd.DataFrame(results)


def main(model, output_path, coef=0, vector_path=None, layer=None, steering_type="response", max_tokens=150, n_samples=10, batch_process=True, bfi_json_path="eval/BFI.json", overwrite=False, judge_model="gpt-4o-mini", max_concurrent_judges=8, judge_delay=0.1, judge_max_retries=5, coef1=None, vector_path1=None):
    """Evaluate a model on the BFI questionnaire with LLM judge scoring."""
    
    if os.path.exists(output_path) and not overwrite:
        print(f"Output path {output_path} already exists, skipping...")
        df = pd.read_csv(output_path)
        # Load BFI data for score computation
        with open(bfi_json_path, 'r', encoding="utf-8") as f:
            bfi_data = json.load(f)
        dimension_scores = compute_bfi_scores(df, bfi_data)
        print("\nBFI Dimension Scores:")
        for dim, scores in dimension_scores.items():
            if scores['mean'] is not None:
                print(f"{dim}: {scores['mean']:.2f} +/- {scores['std']:.2f} (n={scores['n_questions']})")
        return
    
    print(f"Evaluating BFI questionnaire: {output_path}")
    print(f"Rate limiting: max_concurrent_judges={max_concurrent_judges}, judge_delay={judge_delay}s, judge_max_retries={judge_max_retries}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load model
    if coef != 0:
        llm, tokenizer = load_model(model)
        lora_path = None
        vector = torch.load(vector_path, weights_only=False)[layer]
        if coef1 is not None and vector_path1 is not None:
            vector1 = torch.load(vector_path1, weights_only=False)[layer]
            vector = vector * coef + vector1 * coef1
    else:
        llm, tokenizer, lora_path = load_vllm_model(model)
        vector = None
    
    # Load BFI questions
    questions, bfi_data = load_bfi_questions(bfi_json_path)
    
    if batch_process:
        print(f"Batch processing {len(questions)} BFI questions...")
        outputs = asyncio.run(eval_bfi_batched(questions, llm, tokenizer, coef, vector, layer, n_samples, max_tokens, steering_type=steering_type, lora_path=lora_path, judge_model=judge_model, max_concurrent_judges=max_concurrent_judges, judge_delay=judge_delay, judge_max_retries=judge_max_retries))
    else:
        outputs = []
        print(f"Sequential processing {len(questions)} BFI questions...")
        for question in tqdm(questions, desc="Processing BFI questions"):
            result = asyncio.run(question.eval(llm, tokenizer, coef, vector, layer, max_tokens, n_samples, steering_type=steering_type, lora_path=lora_path, judge_model=judge_model, max_concurrent_judges=max_concurrent_judges, judge_delay=judge_delay))
            outputs.append(result)
        outputs = pd.concat(outputs)
    
    # Save results
    outputs.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    
    # Compute and display BFI dimension scores
    dimension_scores = compute_bfi_scores(outputs, bfi_data)
    print("\nBFI Dimension Scores:")
    for dim, scores in dimension_scores.items():
        if scores['mean'] is not None:
            print(f"{dim}: {scores['mean']:.2f} +/- {scores['std']:.2f} (n={scores['n_questions']})")
        else:
            print(f"{dim}: No valid scores")
    
    # Save dimension scores
    scores_path = output_path.replace('.csv', '_dimensions.json')
    with open(scores_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        serializable_scores = {}
        for dim, scores in dimension_scores.items():
            serializable_scores[dim] = {
                'mean': float(scores['mean']) if scores['mean'] is not None else None,
                'std': float(scores['std']) if scores['std'] is not None else None,
                'n_questions': int(scores['n_questions']),
                'ratings': [int(r) for r in scores['ratings']]
            }
        json.dump(serializable_scores, f, indent=2)
    print(f"Dimension scores saved to: {scores_path}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)

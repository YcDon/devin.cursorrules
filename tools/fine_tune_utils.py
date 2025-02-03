#!/usr/bin/env python3

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from openai import OpenAI
import tiktoken
from .token_tracker import TokenUsage, APIResponse, get_token_tracker

@dataclass
class FineTuningDataset:
    """Represents a dataset for fine-tuning"""
    name: str
    objective: str  # 'tone', 'tool_calling', or 'sales'
    examples: List[Dict[str, List[Dict[str, str]]]]  # List of {"messages": [...]} objects
    validation_examples: Optional[List[Dict[str, List[Dict[str, str]]]]] = None
    total_tokens: int = 0
    estimated_cost: float = 0.0

@dataclass
class FineTuningJob:
    """Represents a fine-tuning job"""
    job_id: str
    model: str
    dataset: FineTuningDataset
    status: str
    created_at: float
    finished_at: Optional[float] = None
    metrics: Optional[Dict[str, float]] = None
    hyperparameters: Optional[Dict[str, Union[int, float, str]]] = None

class FineTuningManager:
    """Manages fine-tuning jobs and datasets"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found")
        self.client = OpenAI(api_key=self.api_key)
        self.token_tracker = get_token_tracker()
        self.data_dir = Path('data/fine_tuning')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tokenizer for token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens in a message list using tiktoken"""
        num_tokens = 0
        for message in messages:
            # Every message follows <im_start>{role/name}\n{content}<im_end>\n
            num_tokens += 4
            for key, value in message.items():
                num_tokens += len(self.tokenizer.encode(str(value)))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens

    def estimate_cost(self, total_tokens: int, model: str = "gpt-4o-mini", n_epochs: int = 3) -> float:
        """Estimate the cost of fine-tuning based on token count"""
        # Cost per 1M tokens (as of documentation)
        if model == "gpt-4o-mini-2024-07-18":
            base_cost_per_1m = 3.0  # Example cost, adjust based on actual pricing
        else:
            base_cost_per_1m = 8.0  # Higher cost for full gpt-4o
        
        return (base_cost_per_1m / 1_000_000) * total_tokens * n_epochs
    
    def prepare_dataset(self, examples: List[Dict[str, str]], objective: str, 
                       system_prompt: str, train_test_split: float = 0.8,
                       model: str = "gpt-4o-mini") -> Tuple[FineTuningDataset, FineTuningDataset]:
        """Prepare training and validation datasets with proper formatting"""
        formatted_examples = []
        total_tokens = 0
        
        for example in examples:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example["input"]},
                {"role": "assistant", "content": example["output"]}
            ]
            
            # Count tokens
            example_tokens = self.count_tokens(messages)
            if example_tokens > 65536:  # OpenAI's current limit
                print(f"Warning: Example exceeds token limit ({example_tokens} tokens)")
                continue
                
            total_tokens += example_tokens
            formatted_examples.append({"messages": messages})
        
        # Split into train/test
        split_idx = int(len(formatted_examples) * train_test_split)
        train_examples = formatted_examples[:split_idx]
        test_examples = formatted_examples[split_idx:]
        
        # Calculate costs
        train_tokens = sum(self.count_tokens(ex["messages"]) for ex in train_examples)
        test_tokens = sum(self.count_tokens(ex["messages"]) for ex in test_examples)
        
        train_dataset = FineTuningDataset(
            name=f"{objective}_train",
            objective=objective,
            examples=train_examples,
            total_tokens=train_tokens,
            estimated_cost=self.estimate_cost(train_tokens, model)
        )
        
        test_dataset = FineTuningDataset(
            name=f"{objective}_test",
            objective=objective,
            examples=test_examples,
            total_tokens=test_tokens,
            estimated_cost=self.estimate_cost(test_tokens, model)
        )
        
        return train_dataset, test_dataset
    
    def save_dataset(self, dataset: FineTuningDataset) -> Path:
        """Save dataset to a JSONL file with proper formatting"""
        file_path = self.data_dir / f"{dataset.name}.jsonl"
        with open(file_path, 'w') as f:
            for example in dataset.examples:
                f.write(json.dumps(example) + '\n')
        return file_path
    
    def create_fine_tuning_job(self, dataset: FineTuningDataset, 
                              model: str = "gpt-4o-mini-2024-07-18",
                              hyperparameters: Optional[Dict[str, Union[int, float, str]]] = None) -> FineTuningJob:
        """Create a new fine-tuning job with specified hyperparameters"""
        file_path = self.save_dataset(dataset)
        
        # Upload the file
        with open(file_path, 'rb') as f:
            response = self.client.files.create(
                file=f,
                purpose='fine-tune'
            )
        file_id = response.id
        
        # Prepare job creation parameters
        job_params = {
            "training_file": file_id,
            "model": model,
        }
        
        # Add hyperparameters if specified
        if hyperparameters:
            job_params["hyperparameters"] = hyperparameters
        
        # Create fine-tuning job
        response = self.client.fine_tuning.jobs.create(**job_params)
        
        return FineTuningJob(
            job_id=response.id,
            model=model,
            dataset=dataset,
            status=response.status,
            created_at=time.time(),
            hyperparameters=hyperparameters
        )
    
    def get_job_status(self, job: FineTuningJob) -> FineTuningJob:
        """Get the current status of a fine-tuning job"""
        response = self.client.fine_tuning.jobs.retrieve(job.job_id)
        job.status = response.status
        if response.status == "succeeded":
            job.finished_at = time.time()
            # Add metrics if available
            if hasattr(response, 'metrics'):
                job.metrics = response.metrics
        return job
    
    def evaluate_model(self, job: FineTuningJob, test_examples: List[Dict[str, str]]) -> Dict[str, float]:
        """Evaluate a fine-tuned model using OpenAI's metrics"""
        if not job.finished_at:
            raise ValueError("Job not completed yet")
        
        metrics = {
            "train_loss": 0.0,
            "valid_loss": 0.0,
            "train_mean_token_accuracy": 0.0,
            "valid_mean_token_accuracy": 0.0
        }
        
        # Get metrics from the job
        response = self.client.fine_tuning.jobs.retrieve(job.job_id)
        if hasattr(response, 'metrics'):
            metrics.update(response.metrics)
        
        return metrics

def main():
    """CLI interface for fine-tuning utilities"""
    parser = argparse.ArgumentParser(description='Fine-tuning utilities for OpenAI models')
    parser.add_argument('--action', choices=['prepare', 'train', 'status', 'evaluate'], required=True)
    parser.add_argument('--objective', choices=['tone', 'tool_calling', 'sales'], required=True)
    parser.add_argument('--data', type=str, help='Path to training data file')
    parser.add_argument('--model', type=str, default='gpt-4o-mini-2024-07-18', help='Model to fine-tune')
    parser.add_argument('--job-id', type=str, help='Fine-tuning job ID')
    parser.add_argument('--system-prompt', type=str, help='System prompt for the examples')
    args = parser.parse_args()
    
    manager = FineTuningManager()
    
    if args.action == 'prepare':
        if not args.data or not args.system_prompt:
            raise ValueError("Data file and system prompt required for preparation")
        with open(args.data) as f:
            examples = json.load(f)
        train_dataset, test_dataset = manager.prepare_dataset(
            examples, args.objective, args.system_prompt, model=args.model
        )
        print(f"Created datasets:\nTrain: {len(train_dataset.examples)} examples, {train_dataset.total_tokens} tokens"
              f"\nTest: {len(test_dataset.examples)} examples, {test_dataset.total_tokens} tokens"
              f"\nEstimated cost: ${train_dataset.estimated_cost:.2f}")
        
    elif args.action == 'train':
        if not args.data:
            raise ValueError("Data file required for training")
        with open(args.data) as f:
            dataset = FineTuningDataset(**json.load(f))
        job = manager.create_fine_tuning_job(dataset, model=args.model)
        print(f"Created fine-tuning job: {job.job_id}")
        
    elif args.action == 'status':
        if not args.job_id:
            raise ValueError("Job ID required for status check")
        job = FineTuningJob(args.job_id, args.model, None, "unknown", time.time())
        job = manager.get_job_status(job)
        print(f"Job status: {job.status}")
        if job.metrics:
            print("Metrics:", json.dumps(job.metrics, indent=2))
            
    elif args.action == 'evaluate':
        if not args.job_id or not args.data:
            raise ValueError("Job ID and test data required for evaluation")
        with open(args.data) as f:
            test_examples = json.load(f)
        job = FineTuningJob(args.job_id, args.model, None, "succeeded", time.time(), time.time())
        metrics = manager.evaluate_model(job, test_examples)
        print("Evaluation metrics:", json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main() 
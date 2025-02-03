#!/usr/bin/env python3

import os
import json
import time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

def create_finetune_job(client: OpenAI, training_file_id: str) -> str:
    """Create a fine-tuning job using DPO"""
    response = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        model="gpt-4o-2024-08-06",  # Base model to fine-tune
        method={
            "type": "dpo"  # Use Direct Preference Optimization
        }
    )
    return response.id

def upload_training_file(client: OpenAI, file_path: str) -> str:
    """Upload a training file and return its ID"""
    with open(file_path, 'rb') as f:
        response = client.files.create(
            file=f,
            purpose='fine-tune'
        )
    return response.id

def wait_for_job_completion(client: OpenAI, job_id: str, poll_interval: int = 60):
    """Wait for the fine-tuning job to complete"""
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status
        
        if status == "succeeded":
            print(f"Job completed successfully! Model: {job.fine_tuned_model}")
            break
        elif status == "failed":
            print(f"Job failed: {job.error}")
            break
        else:
            print(f"Job status: {status}")
            if hasattr(job, 'metrics'):
                print("Current metrics:", json.dumps(job.metrics, indent=2))
            time.sleep(poll_interval)

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # Training file path
    training_file = "td_long_to_short.jsonl"
    
    try:
        # Upload training file
        print(f"Uploading training file: {training_file}")
        file_id = upload_training_file(client, training_file)
        print(f"File uploaded successfully. ID: {file_id}")
        
        # Create fine-tuning job
        print("Creating fine-tuning job...")
        job_id = create_finetune_job(client, file_id)
        print(f"Fine-tuning job created. ID: {job_id}")
        
        # Wait for completion
        print("Waiting for job completion...")
        wait_for_job_completion(client, job_id)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 
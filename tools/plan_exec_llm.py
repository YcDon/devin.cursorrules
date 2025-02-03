#!/usr/bin/env python3

import os
import sys
import argparse
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import time

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tools import TokenUsage, APIResponse, get_token_tracker, query_llm

STATUS_FILE = '.cursorrules'

def load_environment():
    """Load environment variables from .env files"""
    env_files = ['.env.local', '.env', '.env.example']
    env_loaded = False
    
    for env_file in env_files:
        env_path = Path('.') / env_file
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            env_loaded = True
            break
    
    if not env_loaded:
        print("Warning: No .env files found. Using system environment variables only.", file=sys.stderr)

def read_plan_status():
    """Read the content of the plan status file, only including content after Multi-Agent Scratchpad"""
    status_file = STATUS_FILE
    try:
        with open(status_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Find the Multi-Agent Scratchpad section
            scratchpad_marker = "# Multi-Agent Scratchpad"
            if scratchpad_marker in content:
                return content[content.index(scratchpad_marker):]
            else:
                print(f"Warning: '{scratchpad_marker}' section not found in {status_file}", file=sys.stderr)
                return ""
    except Exception as e:
        print(f"Error reading {status_file}: {e}", file=sys.stderr)
        return ""

def read_file_content(file_path):
    """Read content from a specified file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return None

def create_llm_client():
    """Create OpenAI client"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    return OpenAI(api_key=api_key)

def query_llm(plan_content, user_prompt=None, file_content=None):
    """Query the LLM with combined prompts"""
    client = create_llm_client()
    
    # Combine prompts into a single user message
    combined_prompt = """You are working on a multi-agent context. The executor is the one who actually does the work. And you are the planner. Now the executor is asking you for help. Please analyze the provided project plan and status, then address the executor's specific query or request.

You need to think like a founder. Prioritize agility and don't over-engineer. Think deep. Try to foresee challenges and derisk earlier. If opportunity sizing or probing experiments can reduce risk with low cost, instruct the executor to do them.
    
Project Plan and Status:
======
""" + plan_content + "\n======\n"

    if file_content:
        combined_prompt += f"\nFile Content:\n======\n{file_content}\n======\n"

    if user_prompt:
        combined_prompt += f"\nUser Query:\n{user_prompt}\n"

    combined_prompt += """\nYour response should be focusing on revising the Multi-Agent Scratchpad section in the .cursorrules file. There is no need to regenerate the entire document. You can use the following format to prompt how to revise the document:

<<<<<<<SEARCH
<text in the original document>
=======
<Proprosed changes>
>>>>>>>

We will do the actual changes in the .cursorrules file.
"""

    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model="o1-preview",
            messages=[
                {"role": "user", "content": combined_prompt}
            ],
            response_format={"type": "text"}
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error querying LLM: {e}", file=sys.stderr)
        return None

def main():
    parser = argparse.ArgumentParser(description='Plan and execute tasks using LLM')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt for the LLM')
    parser.add_argument('--file', type=str, help='Optional file to include in analysis')
    args = parser.parse_args()

    # Load environment variables
    load_environment()

    # Read plan status
    plan_content = read_plan_status()

    # Read file content if specified
    file_content = None
    if args.file:
        file_content = read_file_content(args.file)
        if file_content is None:
            sys.exit(1)

    # Query LLM and output response
    response = query_llm(plan_content, args.prompt, file_content)
    if response:
        print('Following is the instruction on how to revise the Multi-Agent Scratchpad section in .cursorrules:')
        print('========================================================')
        print(response)
        print('========================================================')
        print('Now please do the actual changes in the .cursorrules file. And then switch to the executor role, and read the content of the file to decide what to do next.')
    else:
        print("Failed to get response from LLM")
        sys.exit(1)

if __name__ == "__main__":
    main() 
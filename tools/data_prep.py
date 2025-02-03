#!/usr/bin/env python3

import json
import tiktoken
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum, auto

class TonePreference(Enum):
    CONCISE = auto()
    COMPLYING = auto()
    SENSATIONAL = auto()

class SalesStrategy(Enum):
    OBJECTION_HANDLING = auto()
    BENEFIT_HIGHLIGHTING = auto()
    SOCIAL_PROOF = auto()
    URGENCY_CREATION = auto()
    CLOSING_TECHNIQUE = auto()

@dataclass
class ToolCall:
    name: str
    parameters: Dict[str, Union[str, int, float, bool, List, Dict]]
    expected_output: Optional[str] = None

@dataclass
class TrainingExample:
    """Represents a single training example that combines multiple objectives"""
    conversation: List[Dict[str, str]]  # List of messages with roles and content
    tone: TonePreference
    tool_calls: List[ToolCall]
    sales_strategies: List[SalesStrategy]
    metadata: Dict[str, str]  # Additional annotations and context
    
    def to_dict(self) -> Dict:
        """Convert the example to a dictionary format for JSONL"""
        return {
            "messages": self.conversation,
            "metadata": {
                "tone": self.tone.name,
                "tool_calls": [
                    {
                        "name": tc.name,
                        "parameters": tc.parameters,
                        "expected_output": tc.expected_output
                    } for tc in self.tool_calls
                ],
                "sales_strategies": [s.name for s in self.sales_strategies],
                **self.metadata
            }
        }

class DataPreparationPipeline:
    def __init__(self, tokenizer_name: str = "cl100k_base"):
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        self.max_tokens = 65536  # OpenAI's current limit for fine-tuning
        
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
    
    def validate_example(self, example: TrainingExample) -> List[str]:
        """Validate a training example and return a list of any issues found"""
        issues = []
        
        # Check token count
        token_count = self.count_tokens(example.conversation)
        if token_count > self.max_tokens:
            issues.append(f"Example exceeds token limit: {token_count} > {self.max_tokens}")
        
        # Validate conversation format
        if not example.conversation:
            issues.append("Conversation is empty")
        else:
            # Check message format
            for i, msg in enumerate(example.conversation):
                if "role" not in msg:
                    issues.append(f"Message {i} missing 'role' field")
                if "content" not in msg:
                    issues.append(f"Message {i} missing 'content' field")
                if msg["role"] not in ["system", "user", "assistant"]:
                    issues.append(f"Message {i} has invalid role: {msg['role']}")
        
        # Validate tool calls
        for i, tool in enumerate(example.tool_calls):
            if not tool.name:
                issues.append(f"Tool call {i} missing name")
            if not isinstance(tool.parameters, dict):
                issues.append(f"Tool call {i} parameters must be a dictionary")
        
        return issues
    
    def save_dataset(self, examples: List[TrainingExample], output_file: Path):
        """Save examples to a JSONL file"""
        with open(output_file, 'w') as f:
            for example in examples:
                issues = self.validate_example(example)
                if issues:
                    print(f"Warning: Example has issues: {issues}")
                    continue
                f.write(json.dumps(example.to_dict()) + '\n')
    
    def split_dataset(self, examples: List[TrainingExample], train_ratio: float = 0.8) -> tuple[List[TrainingExample], List[TrainingExample]]:
        """Split examples into training and validation sets while maintaining objective distribution"""
        # Sort examples by objectives to ensure even distribution
        examples = sorted(examples, key=lambda x: (x.tone.name, len(x.tool_calls), len(x.sales_strategies)))
        
        # Calculate split index
        split_idx = int(len(examples) * train_ratio)
        
        # Split while maintaining distribution
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]
        
        return train_examples, val_examples

def create_example_template() -> TrainingExample:
    """Create a template for a training example"""
    return TrainingExample(
        conversation=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant skilled in providing information and assistance while maintaining appropriate tone and using available tools effectively."
            },
            {
                "role": "user",
                "content": "Hello! I'm interested in learning more about your services."
            },
            {
                "role": "assistant",
                "content": "I'd be happy to tell you about our services! [Note: This is where you'd customize the response based on tone and sales strategy]"
            }
        ],
        tone=TonePreference.COMPLYING,
        tool_calls=[
            ToolCall(
                name="example_tool",
                parameters={"param1": "value1"},
                expected_output="Expected tool output"
            )
        ],
        sales_strategies=[SalesStrategy.BENEFIT_HIGHLIGHTING],
        metadata={
            "context": "Initial customer inquiry",
            "objective": "Demonstrate value proposition"
        }
    )

def main():
    """Example usage of the data preparation pipeline"""
    pipeline = DataPreparationPipeline()
    
    # Create example dataset
    examples = [create_example_template()]
    
    # Split dataset
    train_examples, val_examples = pipeline.split_dataset(examples)
    
    # Save datasets
    output_dir = Path("data/training")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pipeline.save_dataset(train_examples, output_dir / "train.jsonl")
    pipeline.save_dataset(val_examples, output_dir / "validation.jsonl")

if __name__ == "__main__":
    main() 
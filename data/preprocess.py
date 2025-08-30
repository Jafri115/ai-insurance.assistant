"""
Data preprocessing pipeline for insurance datasets.
Cleans, filters, and formats insurance Q&A data for fine-tuning.
"""

import os
import json
import pandas as pd
import re
from typing import List, Dict, Any
import argparse
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InsuranceDataPreprocessor:
    """Preprocesses insurance datasets for model training."""
    
    def __init__(self, min_length: int = 10, max_length: int = 2048):
        self.min_length = min_length
        self.max_length = max_length
        self.toxic_keywords = [
            'hate', 'kill', 'die', 'stupid', 'idiot', 'moron',
            # Add more as needed
        ]
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Fix common typos in insurance domain
        replacements = {
            'insurace': 'insurance',
            'poilcy': 'policy',
            'claime': 'claim',
            'preimum': 'premium',
            'deductable': 'deductible',
        }
        
        for wrong, right in replacements.items():
            text = text.replace(wrong, right)
        
        return text
    
    def is_valid_text(self, text: str) -> bool:
        """Check if text meets quality criteria."""
        if not text or len(text) < self.min_length or len(text) > self.max_length:
            return False
        
        # Check for toxic content
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in self.toxic_keywords):
            return False
        
        # Ensure it's insurance-related
        insurance_keywords = [
            'insurance', 'policy', 'claim', 'premium', 'deductible',
            'coverage', 'liability', 'accident', 'health', 'auto',
            'life', 'property', 'casualty', 'underwriting'
        ]
        
        if not any(keyword in text_lower for keyword in insurance_keywords):
            return False
        
        return True
    
    def format_for_training(self, qa_pairs: List[Dict[str, str]], 
                           template: str = "alpaca") -> List[Dict[str, str]]:
        """Format Q&A pairs for specific training templates."""
        formatted_data = []
        
        for pair in qa_pairs:
            question = pair.get('question', '').strip()
            answer = pair.get('answer', '').strip()
            
            if not question or not answer:
                continue
            
            if template == "alpaca":
                formatted_item = {
                    "instruction": "Answer the following insurance-related question accurately and helpfully.",
                    "input": question,
                    "output": answer
                }
            elif template == "chat":
                formatted_item = {
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer}
                    ]
                }
            else:  # Simple Q&A format
                formatted_item = {
                    "question": question,
                    "answer": answer
                }
            
            formatted_data.append(formatted_item)
        
        return formatted_data
    
    def load_raw_data(self, data_path: str) -> List[Dict[str, str]]:
        """Load raw data from various formats."""
        path = Path(data_path)
        
        if not path.exists():
            logger.error(f"Data path {data_path} does not exist")
            return []
        
        qa_pairs = []
        
        if path.suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    qa_pairs = data
                else:
                    qa_pairs = data.get('qa_pairs', [])
        
        elif path.suffix == '.csv':
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                qa_pairs.append({
                    'question': str(row.get('question', '')),
                    'answer': str(row.get('answer', ''))
                })
        
        elif path.is_dir():
            # Process multiple files in directory
            for file_path in path.glob('*.json'):
                qa_pairs.extend(self.load_raw_data(str(file_path)))
        
        return qa_pairs
    
    def process_dataset(self, input_path: str, output_path: str, 
                       template: str = "alpaca", train_split: float = 0.8):
        """Complete preprocessing pipeline."""
        logger.info(f"Loading data from {input_path}")
        raw_data = self.load_raw_data(input_path)
        
        if not raw_data:
            logger.error("No data loaded. Please check your input path.")
            return
        
        logger.info(f"Loaded {len(raw_data)} raw Q&A pairs")
        
        # Clean and filter data
        cleaned_data = []
        for pair in raw_data:
            question = self.clean_text(pair.get('question', ''))
            answer = self.clean_text(pair.get('answer', ''))
            
            if self.is_valid_text(question) and self.is_valid_text(answer):
                cleaned_data.append({
                    'question': question,
                    'answer': answer
                })
        
        logger.info(f"After cleaning: {len(cleaned_data)} valid pairs")
        
        # Format for training
        formatted_data = self.format_for_training(cleaned_data, template)
        
        # Split data
        split_idx = int(len(formatted_data) * train_split)
        train_data = formatted_data[:split_idx]
        val_data = formatted_data[split_idx:]
        
        # Save processed data
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train_path = output_dir / "train.json"
        val_path = output_dir / "validation.json"
        
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        
        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(train_data)} training samples to {train_path}")
        logger.info(f"Saved {len(val_data)} validation samples to {val_path}")
        
        # Generate sample synthetic data if no input data
        if len(raw_data) == 0:
            self.generate_sample_data(output_dir)
    
    def generate_sample_data(self, output_dir: Path):
        """Generate sample insurance Q&A data for demonstration."""
        sample_qa = [
            {
                "question": "What is the difference between comprehensive and collision coverage?",
                "answer": "Comprehensive coverage protects against damages from events like theft, vandalism, weather, and animal collisions. Collision coverage protects against damages from accidents with other vehicles or objects. Both are optional but recommended for newer vehicles."
            },
            {
                "question": "How do I file an auto insurance claim?",
                "answer": "To file an auto insurance claim: 1) Contact your insurance company immediately, 2) Provide your policy number and incident details, 3) Take photos of damages, 4) Get a police report if required, 5) Cooperate with the claims adjuster, and 6) Keep records of all communications."
            },
            {
                "question": "What factors affect my life insurance premium?",
                "answer": "Life insurance premiums are affected by age, health status, lifestyle habits (smoking, drinking), occupation, hobbies, coverage amount, policy type, and family medical history. Younger, healthier individuals typically pay lower premiums."
            },
            {
                "question": "What is a deductible and how does it work?",
                "answer": "A deductible is the amount you pay out-of-pocket before your insurance coverage kicks in. For example, with a $500 deductible, you pay the first $500 of covered damages, and insurance pays the rest. Higher deductibles typically mean lower premium costs."
            },
            {
                "question": "Do I need umbrella insurance?",
                "answer": "Umbrella insurance provides additional liability coverage beyond your standard policies. Consider it if you have significant assets to protect, engage in high-risk activities, or want extra protection against large lawsuits. It's relatively inexpensive for the coverage amount."
            }
        ]
        
        formatted_data = self.format_for_training(sample_qa, "alpaca")
        
        sample_path = output_dir / "sample_data.json"
        with open(sample_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generated {len(formatted_data)} sample Q&A pairs at {sample_path}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess insurance datasets")
    parser.add_argument("--input", type=str, default="data/raw/",
                       help="Input data path (file or directory)")
    parser.add_argument("--output", type=str, default="data/processed/",
                       help="Output directory")
    parser.add_argument("--template", type=str, default="alpaca",
                       choices=["alpaca", "chat", "simple"],
                       help="Training data template format")
    parser.add_argument("--train-split", type=float, default=0.8,
                       help="Training data split ratio")
    parser.add_argument("--min-length", type=int, default=10,
                       help="Minimum text length")
    parser.add_argument("--max-length", type=int, default=2048,
                       help="Maximum text length")
    
    args = parser.parse_args()
    
    preprocessor = InsuranceDataPreprocessor(
        min_length=args.min_length,
        max_length=args.max_length
    )
    
    preprocessor.process_dataset(
        input_path=args.input,
        output_path=args.output,
        template=args.template,
        train_split=args.train_split
    )

if __name__ == "__main__":
    main()

"""
Discord reaction prediction reward functions for GRPO training.
"""

import torch
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global reward model (loaded once)
_reward_model = None
_reward_tokenizer = None

def load_reward_model():
    """Load the reward model once and cache it"""
    global _reward_model, _reward_tokenizer
    
    if _reward_model is None:
        logger.info("Loading Discord reaction reward model...")
        try:
            _reward_model = AutoModelForSequenceClassification.from_pretrained(
                "/workspace/fine-tuning/TBD-reward-7B/merged/",
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            _reward_tokenizer = AutoTokenizer.from_pretrained(
                "/workspace/fine-tuning/TBD-reward-7B/merged/",
                trust_remote_code=True
            )
            _reward_model.eval()
            logger.info("Reward model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load reward model: {e}")
            raise
    
    return _reward_model, _reward_tokenizer

def reaction_reward_func(completions: List[str], prompts: Optional[List[str]] = None, **kwargs) -> List[float]:
    """
    Reward function that uses TBD-reward-7B to score Em's responses
    for predicted Discord reaction counts.
    
    Args:
        completions: List of Em's generated responses  
        prompts: List of IRC conversation contexts
        **kwargs: Additional TRL arguments
        
    Returns:
        List of float scores (predicted reaction counts)
    """
    if not completions:
        return []
        
    model, tokenizer = load_reward_model()
    
    if prompts is None:
        logger.warning("No prompts provided to reward function")
        return [0.0] * len(completions)
    
    if tokenizer is None:
        logger.error("Tokenizer not loaded")
        return [0.0] * len(completions)
    
    scores = []
    logger.info(f"Scoring {len(completions)} completions with reward model")
    
    with torch.no_grad():
        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            try:
                # Clean up completion - just Em's response
                if completion.startswith("<Em>"):
                    completion = completion[4:].strip()
                
                # Format exactly like your reward model evaluation:
                # Full conversation context + Em's response
                conversation_text = f"{prompt}\n<Em> {completion}"
                
                # Tokenize exactly like your evaluation script
                inputs = tokenizer(
                    conversation_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=384,  # Match your evaluation script
                    padding=True
                )
                
                # Move to GPU 0 (following your evaluation script)
                inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
                
                # Get prediction (same as your evaluation)
                outputs = model(**inputs)
                score = outputs.logits.squeeze().item()
                
                # Your reward model outputs preference scores, use directly
                # Clamp to reasonable range for RL stability  
                score = max(0.0, min(10.0, score))
                scores.append(score)
                
            except Exception as e:
                logger.warning(f"Error scoring completion {i}: {e}")
                scores.append(0.0)
    
    if scores:
        min_score = min(scores)
        max_score = max(scores)
        mean_score = sum(scores) / len(scores)
        logger.info(f"Reward scores - min: {min_score:.2f}, max: {max_score:.2f}, mean: {mean_score:.2f}")
    
    return scores

"""
Dataset transforms for IRC conversation data for GRPO training.
"""

import logging

# Set up logging
logger = logging.getLogger(__name__)

def conversation_transform(cfg, *args, **kwargs):
    """
    Transform IRC context to chat format for Em.
    Em uses format: [user message]\n<Em> [assistant message]\n
    
    Input format:
    {
        "conversation_id": "synthetic_0000", 
        "context": "<alice> message\n<bob> reply...",  # IRC conversation
        "starter": "original topic",  # Topic that started the conversation 
        "messages": [...],
        "metadata": {...}
    }
    
    GRPO expects: {"prompt": [{"role": "user", "content": "..."}]}
    """
    def transform_fn(example, tokenizer=None):
        # Get the IRC context
        context = example.get("context", "")
        
        if not context:
            logger.warning("Empty context in conversation, using fallback")
            context = "<user> hello"
        
        # Return as chat message for Em's format
        # Include starter as additional context for reward function
        result = {
            "prompt": [{"role": "user", "content": context}]
        }
        
        # Add starter if available (passed to reward function as kwargs)
        if "starter" in example:
            result["starter"] = example["starter"]
            
        return result
    
    return transform_fn, {
        "remove_columns": ["conversation_id", "messages", "metadata"]
    }

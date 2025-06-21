"""
Shared test fixtures and configuration for grpo_code tests.
"""

import pytest
from unittest.mock import Mock, patch


@pytest.fixture
def sample_irc_conversation():
    """Sample IRC conversation data for testing transforms."""
    return {
        "conversation_id": "synthetic_0000",
        "starter": "matched with someone on dating app and they asked for my salary",
        "context": "<alice> matched with someone on dating app and they asked for my salary\n<matt> what the FUCK\n<matt> no way that's not a scam\n<alice> no it was an asian guy in london (or so i thought)",
        "messages": [
            {"username": "alice", "content": "matched with someone on dating app and they asked for my salary"},
            {"username": "matt", "content": "what the FUCK"},
            {"username": "matt", "content": "no way that's not a scam"},
            {"username": "alice", "content": "no it was an asian guy in london (or so i thought)"}
        ],
        "metadata": {"generated": True}
    }


@pytest.fixture
def sample_test_prompts():
    """Sample IRC prompts for testing reward functions."""
    return [
        "<alice> matched with someone on dating app and they asked for my salary\n<matt> what the FUCK\n<matt> no way that's not a scam",
        "<charlie> just got back from vacation\n<dave> how was it?\n<emma> probably better than this place"
    ]


@pytest.fixture
def sample_test_completions():
    """Sample completions for testing reward functions."""
    return [
        "finally someone with taste around here",
        "vacation sounds nice, i'm stuck debugging this conversation"
    ]


@pytest.fixture
def mock_reward_model():
    """Mock reward model for testing without loading actual model."""
    with patch('grpo_code.rewards.load_reward_model') as mock_load:
        mock_model = Mock()
        mock_tokenizer = Mock()
        
        # Default tokenizer behavior
        mock_tokenizer.return_value = {
            'input_ids': Mock(),
            'attention_mask': Mock()
        }
        
        # Default model behavior
        mock_output = Mock()
        mock_output.logits.squeeze.return_value.item.return_value = 5.0
        mock_model.return_value = mock_output
        
        mock_load.return_value = (mock_model, mock_tokenizer)
        
        yield {
            'load_reward_model': mock_load,
            'model': mock_model,
            'tokenizer': mock_tokenizer,
            'output': mock_output
        }


@pytest.fixture(autouse=True)
def reset_reward_model_cache():
    """Reset the global reward model cache before each test."""
    import grpo_code.rewards as rewards_module
    rewards_module._reward_model = None
    rewards_module._reward_tokenizer = None

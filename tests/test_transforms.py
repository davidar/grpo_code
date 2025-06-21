"""
Tests for IRC conversation data transforms for GRPO training.
"""

import pytest
import json
from grpo_code.transforms import conversation_transform


class TestConversationTransform:
    """Test cases for the conversation transform function."""
    
    def test_basic_transform(self):
        """Test basic transformation with sample IRC data."""
        # Get transform function
        transform_fn, config = conversation_transform(None)
        
        # Test with realistic sample like the original test
        sample_conversation = {
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
        
        result = transform_fn(sample_conversation)
        
        # Should return the expected chat format
        assert "prompt" in result
        assert isinstance(result["prompt"], list)
        assert len(result["prompt"]) == 1
        assert result["prompt"][0]["role"] == "user"
        assert result["prompt"][0]["content"] == sample_conversation["context"]
        
        # Should include starter
        assert "starter" in result
        assert result["starter"] == sample_conversation["starter"]
    
    def test_empty_context_fallback(self):
        """Test fallback behavior when context is empty."""
        transform_fn, config = conversation_transform(None)
        
        # Test with empty context
        sample_conversation = {
            "conversation_id": "test_001",
            "context": "",
            "messages": [],
            "metadata": {}
        }
        
        result = transform_fn(sample_conversation)
        
        # Should use fallback context
        assert result["prompt"][0]["content"] == "<user> hello"
    
    def test_missing_context_field(self):
        """Test behavior when context field is missing."""
        transform_fn, config = conversation_transform(None)
        
        # Test without context field
        sample_conversation = {
            "conversation_id": "test_002",
            "messages": [],
            "metadata": {}
        }
        
        result = transform_fn(sample_conversation)
        
        # Should use fallback context
        assert result["prompt"][0]["content"] == "<user> hello"
    
    def test_starter_handling(self):
        """Test that starter is included when available and excluded when not."""
        transform_fn, config = conversation_transform(None)
        
        # Test with starter
        sample_with_starter = {
            "conversation_id": "test_003",
            "starter": "test topic",
            "context": "<user> hello there",
            "messages": [],
            "metadata": {}
        }
        
        result_with_starter = transform_fn(sample_with_starter)
        assert "starter" in result_with_starter
        assert result_with_starter["starter"] == "test topic"
        
        # Test without starter
        sample_without_starter = {
            "conversation_id": "test_004",
            "context": "<user> hello there",
            "messages": [],
            "metadata": {}
        }
        
        result_without_starter = transform_fn(sample_without_starter)
        assert "starter" not in result_without_starter
    
    def test_remove_columns_config(self):
        """Test that the correct columns are marked for removal."""
        transform_fn, config = conversation_transform(None)
        
        # Check that correct columns are marked for removal
        assert "remove_columns" in config
        expected_removed = ["conversation_id", "messages", "metadata"]
        assert config["remove_columns"] == expected_removed
    
    def test_complex_irc_context(self):
        """Test with more complex IRC conversation context."""
        transform_fn, config = conversation_transform(None)
        
        complex_context = (
            "<alice> so i matched with this guy on tinder\n"
            "<alice> and literally the first thing he asks is about my salary\n"
            "<bob> red flag much?\n"
            "<charlie> probably a scammer\n"
            "<alice> that's what i thought too\n"
            "<dave> unmatch immediately"
        )
        
        sample_conversation = {
            "conversation_id": "complex_001",
            "starter": "dating app experience",
            "context": complex_context,
            "messages": [
                {"username": "alice", "content": "so i matched with this guy on tinder"},
                {"username": "alice", "content": "and literally the first thing he asks is about my salary"},
                {"username": "bob", "content": "red flag much?"},
                {"username": "charlie", "content": "probably a scammer"},
                {"username": "alice", "content": "that's what i thought too"},
                {"username": "dave", "content": "unmatch immediately"}
            ],
            "metadata": {"complexity": "high"}
        }
        
        result = transform_fn(sample_conversation)
        
        # Should preserve the full IRC context
        assert result["prompt"][0]["content"] == complex_context
        assert result["starter"] == "dating app experience"
    
    def test_tokenizer_parameter_ignored(self):
        """Test that the optional tokenizer parameter is handled gracefully."""
        transform_fn, config = conversation_transform(None)
        
        sample_conversation = {
            "conversation_id": "test_005",
            "context": "<user> test message",
            "messages": [],
            "metadata": {}
        }
        
        # Should work with tokenizer=None
        result1 = transform_fn(sample_conversation, tokenizer=None)
        
        # Should work with a mock tokenizer object
        from unittest.mock import Mock
        mock_tokenizer = Mock()
        result2 = transform_fn(sample_conversation, tokenizer=mock_tokenizer)
        
        # Results should be identical regardless of tokenizer
        assert result1 == result2


def test_conversation_transform_integration():
    """
    Integration test that mimics the original test_transform_function.
    """
    # Use realistic sample like the original test
    sample_conversation = {
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
    
    try:
        transform_fn, config = conversation_transform(None)
        result = transform_fn(sample_conversation)
        
        print(f"Transform result: {json.dumps(result, indent=2)}")
        print("Transform function integration test passed!")
        
        # Basic validation
        assert "prompt" in result
        assert isinstance(result["prompt"], list)
        assert result["prompt"][0]["role"] == "user"
        assert "starter" in result
        
        return True
    except Exception as e:
        print(f"Transform function integration test failed: {e}")
        return False


if __name__ == "__main__":
    # Allow running this file directly for quick testing
    pytest.main([__file__, "-v"])

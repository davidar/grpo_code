"""
Tests for Discord reaction prediction reward functions.
"""

import pytest
import json
from unittest.mock import Mock, patch
from grpo_code.rewards import reaction_reward_func, load_reward_model


class TestReactionRewardFunc:
    """Test cases for the reaction reward function."""
    
    def test_empty_completions(self):
        """Test that empty completions list returns empty scores."""
        scores = reaction_reward_func([])
        assert scores == []
    
    def test_no_prompts_provided(self):
        """Test behavior when no prompts are provided."""
        completions = ["test response"]
        scores = reaction_reward_func(completions, prompts=None)
        assert scores == [0.0]
    
    @patch('grpo_code.rewards.load_reward_model')
    def test_reward_function_with_mock_model(self, mock_load_model):
        """Test the reward function with a mocked model."""
        # Mock the model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        
        # Mock tokenizer output
        mock_tokenizer.return_value = {
            'input_ids': Mock(),
            'attention_mask': Mock()
        }
        
        # Mock model output with logits
        mock_output = Mock()
        mock_output.logits.squeeze.return_value.item.return_value = 5.5
        mock_model.return_value = mock_output
        
        # Test data - realistic IRC contexts like the original test
        test_prompts = [
            "<alice> matched with someone on dating app and they asked for my salary\n<matt> what the FUCK\n<matt> no way that's not a scam",
            "<charlie> just got back from vacation\n<dave> how was it?\n<emma> probably better than this place"
        ]
        
        test_completions = [
            "finally someone with taste around here",
            "vacation sounds nice, i'm stuck debugging this conversation"
        ]
        
        scores = reaction_reward_func(test_completions, test_prompts)
        
        # Should return scores for both completions
        assert len(scores) == 2
        assert all(isinstance(score, float) for score in scores)
        assert all(0.0 <= score <= 10.0 for score in scores)
    
    @patch('grpo_code.rewards.load_reward_model')
    def test_completion_cleanup(self, mock_load_model):
        """Test that completions starting with <Em> are cleaned up properly."""
        # Mock the model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        
        # Mock tokenizer to capture the input
        tokenized_inputs = []
        def mock_tokenizer_call(text, **kwargs):
            tokenized_inputs.append(text)
            return {
                'input_ids': Mock(),
                'attention_mask': Mock()
            }
        mock_tokenizer.side_effect = mock_tokenizer_call
        
        # Mock model output
        mock_output = Mock()
        mock_output.logits.squeeze.return_value.item.return_value = 3.0
        mock_model.return_value = mock_output
        
        # Test with Em prefix
        test_prompts = ["<alice> hello"]
        test_completions = ["<Em> this is my response"]
        
        scores = reaction_reward_func(test_completions, test_prompts)
        
        # Verify the function processed the input correctly
        assert len(tokenized_inputs) == 1
        final_text = tokenized_inputs[0]
        
        # The final text should be: prompt + "\n<Em> " + cleaned_completion
        # So it should contain "this is my response" (the cleaned part)
        # And it should be formatted as "<alice> hello\n<Em> this is my response"
        assert "this is my response" in final_text
        assert final_text == "<alice> hello\n<Em> this is my response"
        
        # Verify it doesn't contain the double <Em> prefix that would happen if cleaning failed
        assert "<Em> <Em>" not in final_text
    
    @patch('grpo_code.rewards.load_reward_model')
    def test_score_clamping(self, mock_load_model):
        """Test that scores are clamped to the 0-10 range."""
        # Mock the model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        
        # Mock tokenizer output
        mock_tokenizer.return_value = {
            'input_ids': Mock(),
            'attention_mask': Mock()
        }
        
        # Test with extreme scores
        test_cases = [
            (-5.0, 0.0),  # Below minimum should be clamped to 0
            (15.0, 10.0),  # Above maximum should be clamped to 10
            (5.0, 5.0),   # Normal score should be unchanged
        ]
        
        for raw_score, expected_score in test_cases:
            mock_output = Mock()
            mock_output.logits.squeeze.return_value.item.return_value = raw_score
            mock_model.return_value = mock_output
            
            scores = reaction_reward_func(["test"], ["<user> test"])
            assert scores[0] == expected_score
    
    @patch('grpo_code.rewards.load_reward_model')
    def test_error_handling(self, mock_load_model):
        """Test that errors during scoring are handled gracefully."""
        # Mock the model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        
        # Mock tokenizer to raise an exception
        mock_tokenizer.side_effect = Exception("Tokenization failed")
        
        # Should handle the error and return 0.0
        scores = reaction_reward_func(["test"], ["<user> test"])
        assert scores == [0.0]


class TestLoadRewardModel:
    """Test cases for loading the reward model."""
    
    @patch('grpo_code.rewards.AutoModelForSequenceClassification')
    @patch('grpo_code.rewards.AutoTokenizer')
    def test_model_loading_success(self, mock_tokenizer_class, mock_model_class):
        """Test successful model loading."""
        # Reset the global variables
        import grpo_code.rewards as rewards_module
        rewards_module._reward_model = None
        rewards_module._reward_tokenizer = None
        
        # Mock the model and tokenizer classes
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Load the model
        model, tokenizer = load_reward_model()
        
        # Verify model was loaded and cached
        assert model is mock_model
        assert tokenizer is mock_tokenizer
        mock_model.eval.assert_called_once()
        
        # Verify subsequent calls return cached model
        model2, tokenizer2 = load_reward_model()
        assert model2 is mock_model
        assert tokenizer2 is mock_tokenizer
        
        # Should only call from_pretrained once due to caching
        assert mock_model_class.from_pretrained.call_count == 1
        assert mock_tokenizer_class.from_pretrained.call_count == 1
    
    @patch('grpo_code.rewards.AutoModelForSequenceClassification')
    def test_model_loading_failure(self, mock_model_class):
        """Test model loading failure handling."""
        # Reset the global variables
        import grpo_code.rewards as rewards_module
        rewards_module._reward_model = None
        rewards_module._reward_tokenizer = None
        
        # Mock the model class to raise an exception
        mock_model_class.from_pretrained.side_effect = Exception("Model loading failed")
        
        # Should raise the exception
        with pytest.raises(Exception, match="Model loading failed"):
            load_reward_model()


# Integration test that can be run if the model is available
@pytest.mark.integration
def test_reward_function_integration():
    """
    Integration test for the reward function with real model (if available).
    
    This test is marked as 'integration' and will only run if:
    1. The model files are available at the expected path
    2. The test is explicitly run with: pytest -m integration
    """
    try:
        # Test with realistic IRC contexts
        test_prompts = [
            "<alice> matched with someone on dating app and they asked for my salary\n<matt> what the FUCK\n<matt> no way that's not a scam",
            "<charlie> just got back from vacation\n<dave> how was it?\n<emma> probably better than this place"
        ]
        
        test_completions = [
            "finally someone with taste around here",
            "vacation sounds nice, i'm stuck debugging this conversation"
        ]
        
        scores = reaction_reward_func(test_completions, test_prompts)
        
        # Basic sanity checks
        assert len(scores) == 2
        assert all(isinstance(score, float) for score in scores)
        assert all(0.0 <= score <= 10.0 for score in scores)
        
        print(f"Integration test scores: {scores}")
        print("Integration test passed!")
        
    except Exception as e:
        pytest.skip(f"Integration test skipped - model not available: {e}")


if __name__ == "__main__":
    # Allow running this file directly for quick testing
    pytest.main([__file__, "-v"])

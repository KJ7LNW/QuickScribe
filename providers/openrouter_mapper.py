"""
OpenRouter-specific configuration mapper.
Maps generic configuration to OpenRouter API format.
"""
from typing import Dict, Any
from .provider_config_mapper import ProviderConfigMapper


class OpenRouterMapper(ProviderConfigMapper):
    """Configuration mapper for OpenRouter provider."""

    def map_reasoning_params(self, enable_reasoning: str, thinking_budget: int) -> Dict[str, Any]:
        """
        Map reasoning configuration to OpenRouter's unified reasoning format.

        OpenRouter uses: reasoning = {effort: str} OR reasoning = {max_tokens: int}
        Only one field allowed, not both.
        """
        params = {}
        reasoning_config = {}

        # Prioritize explicit token budget over effort level
        if thinking_budget > 0:
            reasoning_config['max_tokens'] = thinking_budget
        elif enable_reasoning in ['low', 'medium', 'high']:
            reasoning_config['effort'] = enable_reasoning

        if reasoning_config:
            params['reasoning'] = reasoning_config

        return params

    def supports_reasoning(self, model_name: str) -> bool:
        """
        OpenRouter supports reasoning for underlying reasoning models.
        Return True - let OpenRouter validate model capability.
        """
        return True

    def map_route_to_completion_params(self, route_value: str) -> Dict[str, Any]:
        """
        Map route value to OpenRouter provider configuration.

        Configures provider selection with fallback and latency optimization.
        The route_value specifies the preferred provider name.

        Args:
            route_value: Provider name from model_id@provider syntax

        Returns:
            Dictionary with extra_body.provider configuration
        """
        return {
            "extra_body": {
                "provider": {
                    "order": [route_value],
                    "allow_fallbacks": True,
                    "sort": "latency"
                }
            }
        }

"""
Model Provider Compatibility Evaluation.

Evaluates model response quality across supported AI providers:
- Anthropic Direct API
- AWS Bedrock (via OIDC federation in CI)
- Google Vertex AI

This evaluation helps maintainers verify that the register submission
pipeline correctly handles evaluations targeting different providers.

Provider selection is via environment variables:
- ANTHROPIC_API_KEY: Direct API access (not used in CI — Bedrock instead)
- AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY: Bedrock access
- ANTHROPIC_BASE_URL: Custom endpoint (for proxy/LiteLLM setups)
- GOOGLE_APPLICATION_CREDENTIALS: Vertex AI access

In the CI pipeline, Bedrock access is provided via OIDC federation
(no static keys), and the ANTHROPIC_MODEL variable selects the model.
"""
import os
from inspect_ai import Task, task, eval
from inspect_ai.dataset import example_dataset
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import generate, chain_of_thought

# Detect available provider
PROVIDER = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
USE_BEDROCK = os.environ.get("AWS_REGION") is not None
USE_VERTEX = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") is not None
USE_DIRECT = os.environ.get("ANTHROPIC_API_KEY") is not None

@task
def model_compat_check():
    """Verify model response quality across providers."""
    return Task(
        dataset=example_dataset("theory_of_mind"),
        solver=[chain_of_thought(), generate()],
        scorer=model_graded_fact(),
    )

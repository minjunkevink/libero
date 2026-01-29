"""
LLM-based Language Augmenter for LIBERO datasets.

Uses OpenAI GPT to generate extended instructions for counterfactual
language augmentation.

Supports both the openai package (if available) and a fallback
urllib-based implementation for environments like GraalVM Python.
"""

import json
import hashlib
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required if env vars are set directly


def _call_openai_api_urllib(
    api_key: str,
    model: str,
    system_message: str,
    user_message: str,
    temperature: float = 0.8,
    max_tokens: int = 1000
) -> str:
    """
    Call OpenAI API using urllib (fallback for GraalVM Python).
    """
    import urllib.request
    import urllib.error
    
    url = "https://api.openai.com/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode('utf-8'),
        headers=headers,
        method='POST'
    )
    
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result['choices'][0]['message']['content']
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        raise Exception(f"OpenAI API error {e.code}: {error_body}")


class LLMLanguageAugmenter:
    """
    Generates extended language instructions using LLM.
    
    Features:
    - Caches responses to avoid redundant API calls
    - Mock mode for testing without API calls
    - Grounded generation using scene inventory
    """
    
    def __init__(
        self, 
        cache_dir: str = "./augmentation_cache",
        mock: bool = False,
        model: str = "gpt-4o-mini"
    ):
        """
        Initialize the augmenter.
        
        Args:
            cache_dir: Directory to store cached LLM responses
            mock: If True, generate mock responses without API calls
            model: OpenAI model to use (default: gpt-4o-mini for cost efficiency)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.mock = mock
        self.model = model
        
        # Load prompt template
        prompt_path = Path(__file__).parent.parent / "prompts" / "augmentation_prompt.txt"
        if prompt_path.exists():
            self.prompt_template = prompt_path.read_text()
        else:
            # Fallback template
            self.prompt_template = self._get_default_prompt()
        
        # Initialize OpenAI client (lazy)
        self._client = None
    
    def _get_api_key(self) -> str:
        """Get OpenAI API key from environment."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment. "
                "Please set it in .env file or environment variables."
            )
        return api_key
    
    def _get_openai_client(self):
        """Lazy initialization of OpenAI client (if available)."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self._get_api_key())
            except ImportError:
                # Will use urllib fallback
                self._client = "urllib_fallback"
        return self._client
    
    def _get_default_prompt(self) -> str:
        """Return default prompt template if file not found."""
        return '''You are a Robot Instruction Generator for a tabletop manipulation task.

I will give you a completed robot task and a list of objects available in the scene.
Your job is to generate variations that append a plausible "Next Step" to the instruction.

RULES:
1. The "Next Step" must logically follow the completed task.
2. The "Next Step" MUST ONLY involve objects from the "Available Objects" or "Available Fixtures" lists.
3. Do NOT invent objects.
4. Keep phrasing natural and varied.

Current Task (already completed): "{original_instruction}"

Available Objects: {object_list}
Available Fixtures: {fixture_list}
Available Actions: pick up, place, put, open, close, turn on, turn off, push

Generate exactly {num_variations} extended instructions.

Output ONLY a valid JSON array of strings:
["extended instruction 1", "extended instruction 2", ...]'''
    
    def _get_cache_key(
        self, 
        instruction: str, 
        objects: List[str], 
        fixtures: List[str],
        num_variations: int
    ) -> str:
        """Generate cache key from inputs."""
        content = f"{instruction}|{sorted(objects)}|{sorted(fixtures)}|{num_variations}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_cache(self, cache_key: str) -> Optional[List[str]]:
        """Load cached response if exists."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text())
                return data.get('results', None)
            except json.JSONDecodeError:
                return None
        return None
    
    def _save_cache(self, cache_key: str, results: List[str], metadata: Dict[str, Any] = None):
        """Save response to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        data = {
            'results': results,
            'metadata': metadata or {}
        }
        cache_file.write_text(json.dumps(data, indent=2))
    
    def _generate_mock_responses(
        self, 
        instruction: str, 
        objects: List[str], 
        fixtures: List[str],
        num_variations: int
    ) -> List[str]:
        """Generate mock responses for testing."""
        # Create plausible extensions based on available objects/fixtures
        extensions = []
        
        connectors = ["and then", "then", "after that", "followed by"]
        actions = [
            ("open", "drawer", "wooden cabinet"),
            ("close", "drawer", "wooden cabinet"),
            ("turn on", "stove", "flat stove"),
            ("turn off", "stove", "flat stove"),
            ("put", "on the plate", "plate"),
            ("place", "on the stove", "flat stove"),
            ("put", "in the bowl", "bowl"),
            ("place", "on top of the cabinet", "wooden cabinet"),
        ]
        
        for i in range(num_variations):
            connector = connectors[i % len(connectors)]
            action, prep, _ = actions[i % len(actions)]
            
            if "open" in action or "close" in action or "turn" in action:
                extension = f"{instruction} {connector} {action} the {prep}"
            else:
                extension = f"{instruction} {connector} {action} it {prep}"
            
            extensions.append(extension)
        
        return extensions[:num_variations]
    
    def _call_llm_api(
        self, 
        instruction: str, 
        objects: List[str], 
        fixtures: List[str],
        num_variations: int
    ) -> List[str]:
        """
        Call OpenAI API to generate extensions.
        Supports both openai package and urllib fallback.
        """
        client = self._get_openai_client()
        
        # Format prompt
        prompt = self.prompt_template.format(
            original_instruction=instruction,
            object_list=", ".join(objects),
            fixture_list=", ".join(fixtures),
            num_variations=num_variations
        )
        
        system_message = "You are a helpful assistant that generates robot task instructions. Always respond with valid JSON."
        
        try:
            if client == "urllib_fallback":
                # Use urllib-based fallback (for GraalVM Python)
                content = _call_openai_api_urllib(
                    api_key=self._get_api_key(),
                    model=self.model,
                    system_message=system_message,
                    user_message=prompt,
                    temperature=0.8,
                    max_tokens=1000
                )
            else:
                # Use openai package
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.8,
                    max_tokens=1000
                )
                content = response.choices[0].message.content.strip()
            
            # Try to extract JSON from response
            # Sometimes GPT wraps it in markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            results = json.loads(content)
            
            if not isinstance(results, list):
                raise ValueError("Response is not a list")
            
            return results
            
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse LLM response as JSON: {e}")
            print(f"Raw response: {content[:500]}")
            # Return mock results as fallback
            return self._generate_mock_responses(instruction, objects, fixtures, num_variations)
        except Exception as e:
            print(f"Warning: LLM API call failed: {e}")
            return self._generate_mock_responses(instruction, objects, fixtures, num_variations)
    
    def generate_extensions(
        self, 
        instruction: str, 
        objects: List[str], 
        fixtures: List[str],
        num_variations: int = 10
    ) -> List[str]:
        """
        Generate extended instructions for the given task.
        
        Args:
            instruction: Original task instruction
            objects: List of available object types in the scene
            fixtures: List of available fixture types in the scene
            num_variations: Number of variations to generate
            
        Returns:
            List of extended instruction strings
        """
        cache_key = self._get_cache_key(instruction, objects, fixtures, num_variations)
        
        # Check cache first
        cached = self._load_cache(cache_key)
        if cached:
            print(f"  [Cache hit] Using cached results for: {instruction[:50]}...")
            return cached
        
        print(f"  [Generating] Creating {num_variations} variations for: {instruction[:50]}...")
        
        if self.mock:
            results = self._generate_mock_responses(instruction, objects, fixtures, num_variations)
        else:
            results = self._call_llm_api(instruction, objects, fixtures, num_variations)
        
        # Save to cache
        self._save_cache(cache_key, results, {
            'original_instruction': instruction,
            'objects': objects,
            'fixtures': fixtures,
            'mock': self.mock,
            'model': self.model if not self.mock else 'mock'
        })
        
        return results


if __name__ == "__main__":
    # Test the augmenter
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LLM Language Augmenter")
    parser.add_argument("--mock", action="store_true", help="Use mock responses")
    parser.add_argument("--instruction", type=str, default="Put the bowl on the stove")
    parser.add_argument("--num", type=int, default=5)
    args = parser.parse_args()
    
    # Sample LIBERO-Goal scene
    objects = ["black bowl", "cream cheese", "wine bottle", "plate"]
    fixtures = ["wooden cabinet", "flat stove", "wine rack"]
    
    augmenter = LLMLanguageAugmenter(mock=args.mock)
    
    print(f"\nOriginal instruction: {args.instruction}")
    print(f"Objects: {objects}")
    print(f"Fixtures: {fixtures}")
    print(f"\nGenerating {args.num} variations...\n")
    
    results = augmenter.generate_extensions(
        args.instruction, 
        objects, 
        fixtures, 
        num_variations=args.num
    )
    
    print("Generated variations:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r}")

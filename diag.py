import json
from llm_advisor import LLMAdvisor
r = LLMAdvisor().diagnose_loss("Manyata Block D Garden", "a tree", 4, 12.0)
print(json.dumps(r, indent=2, ensure_ascii=False, default=str))

# tests/test_safety.py

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.safety import SafetyGuardrails
from src.agent import LiDARFailureAnalyzer

if __name__ == "__main__":
    print("="*80)
    print("TESTING SAFETY WITH AGENT")
    print("="*80)
    
    agent = LiDARFailureAnalyzer()
    guardrails = SafetyGuardrails()
    
    test_queries = [
        # Valid - should pass
        ("LiDAR fails at night in fog", True, True),
        # Too short - input validation fails
        ("Why LiDAR?", False, True),
        # Injection attack - security fails
        ("'; DROP TABLE papers; --", False, False),
        # Out of domain - domain check fails
        ("How do I cook pasta?", True, False),
        # Out of domain but mentions sensor
        ("Can sensors detect good spaghetti?", True, False),
        # Valid technical query
        ("How does BEV fusion handle multipath reflections?", True, True),
    ]
    
    print("\nLayer 1 & 2 Tests (Input + Domain):\n")
    
    for query, should_pass_input, should_pass_domain in test_queries:
        is_valid, reason = guardrails.validate_input(query)
        is_in_domain, score = guardrails.check_domain(query)
        
        input_status = "PASS" if is_valid == should_pass_input else "FAIL"
        domain_status = "PASS" if is_in_domain == should_pass_domain else "FAIL"
        
        print(f"Query: {query[:50]:50}")
        print(f"  Input: {input_status} (valid={is_valid}, expected={should_pass_input})")
        print(f"  Domain: {domain_status} (domain={is_in_domain}, score={score:.0%})")
        print()
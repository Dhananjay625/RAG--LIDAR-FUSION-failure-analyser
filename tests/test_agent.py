# tests/test_agent_with_safety.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import LiDARFailureAnalyzer

if __name__ == "__main__":
    print("="*80)
    print("TESTING AGENT WITH SAFETY GUARDRAILS")
    print("="*80)
    
    agent = LiDARFailureAnalyzer()
    
    # Test with both valid and invalid queries
    test_queries = [
        ("LiDAR fails at night in fog", "VALID"),
        ("Why LiDAR?", "INVALID - too short"),
        ("How do I cook pasta?", "INVALID - out of domain"),
        ("BEV fusion accuracy drops in adverse weather", "VALID"),
    ]
    
    for query, expected in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"Expected: {expected}")
        print(f"{'='*80}")
        
        result = agent.run(query)
        
        print(f"\nSafety Status: {'PASSED' if result.get('safety_passed') else 'BLOCKED'}")
        
        if not result.get('safety_passed'):
            print(f"Reason: {result.get('safety_reason', 'Unknown')}")
        
        print(f"\nDiagnosis (Confidence: {result['confidence']:.0%}):")
        print(result['diagnosis'][:200])
        
        if result['citations']:
            print(f"\nSources: {len(result['citations'])} citations")
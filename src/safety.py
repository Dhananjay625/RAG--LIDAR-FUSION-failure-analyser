# src/safety.py

from typing import List, Tuple, Dict
import re

class InputValidator:
    """Validates user queries before processing."""
    
    MAX_LENGTH = 1000
    MIN_LENGTH = 10
    BLOCKED_KEYWORDS = [
        "hack", "exploit", "bypass", "sql injection",
        "delete all", "drop table", "credit card", "password"
    ]
    
    @staticmethod
    def validate(query: str) -> Tuple[bool, str]:
        """Returns: (is_valid, reason_if_invalid)"""
        
        # Check length
        if len(query) < InputValidator.MIN_LENGTH:
            return False, "Query too short (min 10 chars)"
        
        if len(query) > InputValidator.MAX_LENGTH:
            return False, "Query too long (max 1000 chars)"
        
        # Check for blocked keywords (injection attacks)
        query_lower = query.lower()
        for blocked in InputValidator.BLOCKED_KEYWORDS:
            if blocked in query_lower:
                return False, f"Query contains blocked keyword: {blocked}"
        
        # Check it's not gibberish
        words = query.split()
        if len(words) < 3:
            return False, "Query too vague (use at least 3 words)"
        
        # Check it's not just numbers/special chars
        alpha_count = sum(1 for c in query if c.isalpha())
        if alpha_count / len(query) < 0.5:
            return False, "Query appears to be gibberish"
        
        return True, ""


class DomainChecker:
    """Ensures query is about LiDAR/sensor fusion."""
    
    IN_DOMAIN_KEYWORDS = [
        "lidar", "bev", "fusion", "sensor", "radar", "camera",
        "3d detection", "perception", "autonomous", "detection",
        "weather", "fog", "rain", "night", "adverse",
        "segmentation", "object detection", "sensor fusion",
        "point cloud", "feature", "model", "network"
    ]
    
    OUT_OF_DOMAIN_KEYWORDS = [
        "recipe", "cooking", "music", "sports", "history",
        "politics", "finance", "legal advice", "medical",
        "quantum", "biology", "art", "literature", "movie"
    ]
    
    @staticmethod
    def check_domain(query: str) -> Tuple[bool, float]:
        """Returns: (is_in_domain, confidence_score)"""
        query_lower = query.lower()
        
        # Check out-of-domain first (hard rejection)
        for out_keyword in DomainChecker.OUT_OF_DOMAIN_KEYWORDS:
            if out_keyword in query_lower:
                return False, 0.0
        
        # Count in-domain keywords
        in_domain_count = sum(
            1 for keyword in DomainChecker.IN_DOMAIN_KEYWORDS
            if keyword in query_lower
        )
        
        domain_score = in_domain_count / len(DomainChecker.IN_DOMAIN_KEYWORDS)
        
        # Need at least 1 in-domain keyword
        if in_domain_count > 0:
            return True, min(domain_score, 1.0)
        else:
            return False, domain_score


class HallucinationDetector:
    """Detects if LLM made up claims not in papers."""
    
    @staticmethod
    def extract_claims(diagnosis: str) -> List[str]:
        """Extract factual claims from diagnosis."""
        claims = []
        sentences = diagnosis.split(". ")
        
        for sentence in sentences:
            # Claim if it contains specific findings
            if any(word in sentence.lower() for word in 
                   ["percent", "%", "db", "accuracy", "error", 
                    "reduces", "increases", "causes", "leads to",
                    "shows", "demonstrates", "confirms", "paper"]):
                cleaned = sentence.strip()
                if len(cleaned) > 10:
                    claims.append(cleaned)
        
        return claims
    
    @staticmethod
    def verify_claims(claims: List[str], papers_text: List[str]) -> Tuple[float, List[str]]:
        """
        Verify claims are grounded in papers.
        Returns: (grounding_score 0-1, unsupported_claims)
        """
        if not claims:
            return 1.0, []
        
        unsupported = []
        supported = 0
        
        for claim in claims:
            # Extract keywords from claim
            words = claim.lower().split()
            claim_keywords = [w for w in words if len(w) > 3]
            
            if not claim_keywords:
                continue
            
            # Check if any paper mentions similar concepts
            is_grounded = False
            for paper_text in papers_text:
                paper_lower = paper_text.lower()
                
                # If 2+ keywords from claim appear in paper
                keyword_matches = sum(
                    1 for kw in claim_keywords 
                    if kw in paper_lower
                )
                
                if keyword_matches >= 2:
                    is_grounded = True
                    break
            
            if is_grounded:
                supported += 1
            else:
                unsupported.append(claim)
        
        score = supported / len(claims) if claims else 1.0
        return score, unsupported


class ConfidenceGate:
    """Controls when to return answers vs abstain."""
    
    CONFIDENCE_THRESHOLD = 0.75
    CITATION_THRESHOLD = 2
    GROUNDING_THRESHOLD = 0.80
    
    @staticmethod
    def should_return(confidence: float, num_citations: int, 
                     hallucination_score: float) -> Tuple[bool, str]:
        """
        Decide: return diagnosis or abstain?
        Returns: (should_return, reason)
        """
        
        # Check confidence
        if confidence < ConfidenceGate.CONFIDENCE_THRESHOLD:
            reason = f"Confidence too low: {confidence:.1%} < {ConfidenceGate.CONFIDENCE_THRESHOLD:.0%}"
            return False, reason
        
        # Check citations
        if num_citations < ConfidenceGate.CITATION_THRESHOLD:
            reason = f"Too few citations: {num_citations} < {ConfidenceGate.CITATION_THRESHOLD}"
            return False, reason
        
        # Check grounding
        if hallucination_score < ConfidenceGate.GROUNDING_THRESHOLD:
            reason = f"Claims not well grounded: {hallucination_score:.0%} < {ConfidenceGate.GROUNDING_THRESHOLD:.0%}"
            return False, reason
        
        return True, "All checks passed"


class SafetyGuardrails:
    """Multi-layer safety checks for RAG agent."""
    
    def __init__(self):
        self.input_validator = InputValidator()
        self.domain_checker = DomainChecker()
        self.hallucination_detector = HallucinationDetector()
        self.confidence_gate = ConfidenceGate()
    
    def validate_input(self, query: str) -> Tuple[bool, str]:
        """Layer 1: Input validation."""
        is_valid, reason = self.input_validator.validate(query)
        if not is_valid:
            print(f"  [SECURITY] Input rejected: {reason}")
        return is_valid, reason
    
    def check_domain(self, query: str) -> Tuple[bool, float]:
        """Layer 2: Domain checking."""
        is_in_domain, score = self.domain_checker.check_domain(query)
        if not is_in_domain:
            print(f"  [DOMAIN] Out-of-domain query (relevance: {score:.0%})")
        return is_in_domain, score
    
    def detect_hallucinations(self, diagnosis: str, 
                            papers_text: List[str]) -> Tuple[float, List[str]]:
        """Layer 3: Hallucination detection."""
        claims = self.hallucination_detector.extract_claims(diagnosis)
        if not claims:
            return 1.0, []
        
        score, unsupported = self.hallucination_detector.verify_claims(
            claims, papers_text
        )
        
        if unsupported:
            print(f"  [HALLUCINATION] Detected {len(unsupported)} unsupported claims ({len(unsupported)}/{len(claims)})")
        
        return score, unsupported
    
    def gate_response(self, confidence: float, num_citations: int,
                     hallucination_score: float) -> Tuple[bool, str]:
        """Layer 4: Confidence gating."""
        should_return, reason = self.confidence_gate.should_return(
            confidence, num_citations, hallucination_score
        )
        
        if not should_return:
            print(f"  [GATE] Response blocked: {reason}")
        
        return should_return, reason
    
    def full_check(self, query: str, diagnosis: str, confidence: float,
                  citations: List[str], papers_text: List[str]) -> Tuple[bool, str, str]:
        """
        Run all 4 layers of guardrails.
        Returns: (passed, reason, filtered_diagnosis)
        """
        
        print("\n[SAFETY CHECKS]")
        
        # Layer 1: Input validation
        is_valid, reason = self.validate_input(query)
        if not is_valid:
            return False, f"[INPUT] {reason}", diagnosis
        
        print("  [LAYER 1] Input validation: PASSED")
        
        # Layer 2: Domain check
        is_in_domain, domain_score = self.check_domain(query)
        if not is_in_domain:
            return False, f"[DOMAIN] Out-of-domain query (relevance: {domain_score:.0%})", diagnosis
        
        print(f"  [LAYER 2] Domain check: PASSED (relevance: {domain_score:.0%})")
        
        # Layer 3: Hallucination detection
        hallucination_score, unsupported = self.detect_hallucinations(
            diagnosis, papers_text
        )
        
        filtered_diagnosis = diagnosis
        if unsupported:
            for claim in unsupported:
                filtered_diagnosis = filtered_diagnosis.replace(
                    claim, 
                    "[REMOVED: Unsupported claim]"
                )
            print(f"  [LAYER 3] Hallucination detection: PASSED with {len(unsupported)} claims filtered")
        else:
            print(f"  [LAYER 3] Hallucination detection: PASSED (no unsupported claims)")
        
        # Layer 4: Confidence gating
        should_return, gate_reason = self.gate_response(
            confidence, len(citations), hallucination_score
        )
        
        if not should_return:
            return False, f"[GATE] {gate_reason}", filtered_diagnosis
        
        print(f"  [LAYER 4] Confidence gating: PASSED")
        print("  [ALL CHECKS] PASSED\n")
        
        return True, "All safety checks passed", filtered_diagnosis


if __name__ == "__main__":
    """Test the safety guardrails."""
    
    print("="*80)
    print("TESTING SAFETY GUARDRAILS")
    print("="*80)
    
    guardrails = SafetyGuardrails()
    
    # Test cases
    test_cases = [
        {
            "name": "Valid LiDAR question",
            "query": "Why does LiDAR fail at night in fog?",
            "should_pass_input": True,
            "should_pass_domain": True
        },
        {
            "name": "Too short query",
            "query": "Why LiDAR?",
            "should_pass_input": False,
            "should_pass_domain": True
        },
        {
            "name": "Injection attack",
            "query": "'; DROP TABLE papers; --",
            "should_pass_input": False,
            "should_pass_domain": True
        },
        {
            "name": "Out of domain (cooking)",
            "query": "How do I cook pasta with sensors?",
            "should_pass_input": True,
            "should_pass_domain": False
        },
        {
            "name": "LiDAR with technical terms",
            "query": "How does BEV fusion handle multipath reflections in dense traffic?",
            "should_pass_input": True,
            "should_pass_domain": True
        },
    ]
    
    for test in test_cases:
        print(f"\nTest: {test['name']}")
        print(f"Query: {test['query']}")
        
        is_valid, reason = guardrails.validate_input(test['query'])
        is_domain, score = guardrails.check_domain(test['query'])
        
        input_status = "PASS" if is_valid == test['should_pass_input'] else "FAIL"
        domain_status = "PASS" if is_domain == test['should_pass_domain'] else "FAIL"
        
        print(f"  Input validation: {input_status} (valid={is_valid})")
        print(f"  Domain check: {domain_status} (domain={is_domain}, score={score:.0%})")
    
    # Test hallucination detection
    print("\n" + "="*80)
    print("TESTING HALLUCINATION DETECTION")
    print("="*80)
    
    diagnosis = """
    LiDAR fails at night because cameras lose 85% sensitivity in low light.
    This is due to thermal sensor drift. The fusion model uses static weights,
    which cannot adapt to changing sensor reliability. Additionally, quantum
    entanglement destabilizes the point cloud processing (this is made up).
    """
    
    papers = [
        "Camera sensitivity drops 85% in low light conditions at night.",
        "Thermal sensor drift reduces accuracy by 3-5% in adverse weather.",
        "Sensor fusion requires adaptive weighting mechanisms for robustness.",
        "The BEV projection fails when camera features are unreliable."
    ]
    
    print(f"\nDiagnosis: {diagnosis[:100]}...")
    
    claims = guardrails.hallucination_detector.extract_claims(diagnosis)
    print(f"\nExtracted {len(claims)} claims:")
    for i, claim in enumerate(claims, 1):
        print(f"  {i}. {claim[:70]}...")
    
    score, unsupported = guardrails.hallucination_detector.verify_claims(claims, papers)
    print(f"\nHallucination score: {score:.0%}")
    print(f"Unsupported claims: {len(unsupported)}/{len(claims)}")
    for claim in unsupported:
        print(f"  - {claim[:70]}...")
    
    # Test confidence gating
    print("\n" + "="*80)
    print("TESTING CONFIDENCE GATING")
    print("="*80)
    
    test_gates = [
        {"confidence": 0.92, "citations": 4, "hallucination": 0.95, "should_pass": True},
        {"confidence": 0.65, "citations": 4, "hallucination": 0.95, "should_pass": False},
        {"confidence": 0.92, "citations": 1, "hallucination": 0.95, "should_pass": False},
        {"confidence": 0.92, "citations": 4, "hallucination": 0.70, "should_pass": False},
    ]
    
    for test in test_gates:
        should_return, reason = guardrails.confidence_gate.should_return(
            test["confidence"], test["citations"], test["hallucination"]
        )
        
        status = "PASS" if should_return == test["should_pass"] else "FAIL"
        print(f"{status}: conf={test['confidence']:.0%}, cit={test['citations']}, hal={test['hallucination']:.0%}")
        if not should_return:
            print(f"  Reason: {reason}")
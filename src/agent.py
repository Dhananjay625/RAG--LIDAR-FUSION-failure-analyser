# src/agent.py

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END, START  
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from src.retriever import LiDARRetriever
from src.safety import SafetyGuardrails
from dotenv import load_dotenv
import os
import json
import time
import random

load_dotenv()

# Define the agent state
class FailureAnalysisState(TypedDict):
    query: str
    retrieved_papers: List[Dict]
    diagnosis: str
    citations: List[str]
    confidence: float
    is_validated: bool
    retry_count: int
    all_retrieved_papers: List[Dict]

class LiDARFailureAnalyzer:
    """
    Agent that diagnoses LiDAR sensor fusion failures using RAG.
    Smart retry logic: Up to 2 retries with broader search if confidence is low.
    """
    
    def __init__(self):
        """Initialize agent components."""
        # 1. Initialize retriever
        self.retriever = LiDARRetriever(chroma_path="chroma_db")
        print("[OK] Retriever initialized")

        # 2. Initialize LLM
        self.llm = ChatNVIDIA(
            model="nvidia/nemotron-3-super-120b-a12b",
            api_key=os.getenv("NVIDIA_API_KEY"),
            temperature=0.7,
            max_completion_tokens=2000
        )
        print("[OK] LLM initialized")

        # 3. Create LangGraph StateGraph
        graph = StateGraph(FailureAnalysisState)

        # 4. Add nodes
        graph.add_node("retrieve", self.retrieve_node)
        graph.add_node("diagnose", self.diagnose_node)
        graph.add_node("validate", self.validate_node)

        # 5. Add edges with conditional logic
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "diagnose")
        graph.add_conditional_edges(
            "diagnose",
            self.should_retry,
            {"validate": "validate", "retrieve": "retrieve"}
        )
        graph.add_edge("validate", END)

        # 6. Compile the graph
        self.graph = graph.compile()
        print("[OK] Agent graph compiled (smart retry enabled)")
    
    def retrieve_node(self, state: FailureAnalysisState) -> Dict:
        """Node 1: Retrieve relevant papers."""
        retry_count = state.get("retry_count", 0)
        query = state["query"]
        
        if retry_count == 0:
            k = 5
            strategy = "Initial search"
        elif retry_count == 1:
            k = 8
            strategy = "Broader search (retry 1)"
        else:
            k = 10
            strategy = "Expanded search (retry 2)"
        
        print(f"  [Retrieve-{retry_count}] {strategy} - searching for {k} papers...")
        print(f"           Query: {query[:60]}...")
        
        results = self.retriever.retrieve_with_metadata(query, k=k)
        
        print(f"  [Retrieve-{retry_count}] Found {len(results)} papers")
        
        # Accumulate papers across retries
        all_papers = state.get("all_retrieved_papers", [])
        filenames_seen = {p['filename'] for p in all_papers}
        for paper in results:
            if paper['filename'] not in filenames_seen:
                all_papers.append(paper)
                filenames_seen.add(paper['filename'])
        
        return {
            "retrieved_papers": results,
            "all_retrieved_papers": all_papers,
            "retry_count": retry_count
        }
    
    def diagnose_node(self, state: FailureAnalysisState) -> Dict:
        """Node 2: LLM reasons over papers and generates diagnosis."""
        retry_count = state.get("retry_count", 0)
        papers_to_use = state.get("all_retrieved_papers", state["retrieved_papers"])
        
        print(f"  [Diagnose-{retry_count}] LLM analyzing {len(papers_to_use)} papers...")
        
        papers_context = "\n".join([
            f"Paper: {p['filename']} (chunk {p['chunk_id']})\nContent: {p['text'][:250]}"
            for p in papers_to_use[:10]
        ])

        prompt = f"""You are a LiDAR sensor fusion expert with deep knowledge of autonomous driving systems.
Given these research papers and a failure description, diagnose the root cause.

Papers:
{papers_context}

Failure Description: {state["query"]}

Provide your diagnosis in valid JSON format with these keys:
- diagnosis: (string) Root cause diagnosis with technical explanation
- confidence: (float between 0 and 1) Your confidence in this diagnosis
- citations: (list of strings) Which papers support this (format: "filename, chunk N")

Return ONLY valid JSON, no markdown or extra text."""
        
        max_retries = 2
        wait_times = [1, 2]
        
        for attempt in range(max_retries):
            try:
                print(f"  [Diagnose-{retry_count}] Calling LLM (attempt {attempt + 1}/{max_retries})...")
                response_text = "".join(
                    chunk.content for chunk in self.llm.stream(prompt)
                )
                print(f"  [Diagnose-{retry_count}] LLM response received")
                
                try:
                    diagnosis_data = json.loads(response_text)
                    diagnosis = diagnosis_data.get("diagnosis", "Unable to diagnose")
                    confidence = float(diagnosis_data.get("confidence", 0.5))
                    citations = diagnosis_data.get("citations", [])
                except json.JSONDecodeError:
                    diagnosis = response_text[:500]
                    confidence = 0.6
                    citations = [p['filename'] for p in papers_to_use[:2]]
                
                print(f"  [Diagnose-{retry_count}] Confidence: {confidence:.1%}")
                
                return {
                    "diagnosis": diagnosis,
                    "confidence": confidence,
                    "citations": citations,
                    "retry_count": retry_count + 1
                }
            
            except Exception as e:
                error_msg = str(e)
                is_timeout = "timeout" in error_msg.lower() or "timed out" in error_msg.lower()
                is_overload = "503" in error_msg or "overload" in error_msg.lower()
                
                if attempt < max_retries - 1 and (is_timeout or is_overload):
                    # Full jitter on the backoff so concurrent users do not
                    # resynchronise and hammer the endpoint together.
                    wait_time = wait_times[attempt] * random.uniform(0.5, 1.5)
                    print(f"  [Diagnose-{retry_count}] Attempt {attempt + 1} failed: {error_msg[:60]}")
                    print(f"  [WAIT] Sleeping {wait_time:.1f}s before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"  [Diagnose-{retry_count}] All LLM retries exhausted")
                    return {
                        "diagnosis": f"Unable to diagnose after {max_retries} attempts",
                        "confidence": 0.0,
                        "citations": [],
                        "retry_count": retry_count + 1
                    }

    
    def validate_node(self, state: FailureAnalysisState) -> Dict:
        """Node 3: Validate diagnosis."""
        confidence = state.get("confidence", 0.5)
        citations_found = len(state["citations"]) > 0
        diagnosis_substantive = len(state["diagnosis"]) > 50
        
        is_valid = citations_found and diagnosis_substantive
        
        retry_count = state.get("retry_count", 0)
        print(f"  [Validate-{retry_count}] Confidence: {confidence:.1%}, Citations: {len(state['citations'])}, Valid: {is_valid}")
        
        if confidence < 0.75:
            print(f"  [WARNING] Low confidence: {confidence:.1%} - answer may be uncertain")
        
        return {"is_validated": is_valid}
    
    def should_retry(self, state: FailureAnalysisState) -> str:
        """Decision function: retry or validate?"""
        confidence = state.get("confidence", 0.5)
        retry_count = state.get("retry_count", 0)
        
        print(f"  [Decision] Retry count: {retry_count}/3, Confidence: {confidence:.1%}")
        
        # HARD LIMIT: 3 diagnose passes total (initial + 2 retries).
        # retry_count is incremented by diagnose_node and arrives here via the
        # returned state - it must NOT be mutated in this function, because
        # LangGraph only merges values that nodes RETURN.
        if retry_count >= 3:
            print(f"  [HARD STOP] Max retries ({retry_count}) reached -> VALIDATE")
            return "validate"
        
        if confidence >= 0.75:
            print(f"  [Decision] Confidence {confidence:.1%} >= 0.75 -> VALIDATE")
            return "validate"
        else:
            print(f"  [Decision] Confidence {confidence:.1%} < 0.75 + retries available -> RETRY #{retry_count + 1}")
            return "retrieve"

    def run(self, failure_description: str) -> Dict:
        """Run agent with full safety guardrails."""
        
        guardrails = SafetyGuardrails()
        
        print(f"\n[ANALYZE] {failure_description[:60]}...\n")
        
        # Layer 1-2: Validate input + check domain
        is_valid, reason = guardrails.validate_input(failure_description)
        if not is_valid:
            return {
                "query": failure_description,
                "diagnosis": f"[REJECTED] Input validation failed: {reason}",
                "confidence": 0.0,
                "citations": [],
                "is_validated": False,
                "safety_passed": False
            }
        
        is_in_domain, domain_score = guardrails.check_domain(failure_description)
        if not is_in_domain:
            return {
                "query": failure_description,
                "diagnosis": f"[OUT-OF-DOMAIN] This question doesn't appear to be about LiDAR or sensor fusion. Please ask about: LiDAR, BEV fusion, sensor fusion, 3D object detection, autonomous driving perception, or related topics.",
                "confidence": 0.0,
                "citations": [],
                "is_validated": False,
                "safety_passed": False
            }
        
        print(f"  [PASS] Input validation")
        print(f"  [PASS] Domain check (relevance: {domain_score:.0%})")
        
        # Run normal agent logic
        initial_state = {
            "query": failure_description,
            "retrieved_papers": [],
            "diagnosis": "",
            "citations": [],
            "confidence": 0.0,
            "is_validated": False,
            "retry_count": 0,
            "all_retrieved_papers": []
        }
        
        try:
            # Circuit breaker: bounds the graph even if the retry wiring regresses.
            result = self.graph.invoke(initial_state, config={"recursion_limit": 10})
            
            # Layer 3-4: Hallucination detection + confidence gating
            papers_text = [p['text'] for p in result.get("all_retrieved_papers", [])]
            
            safety_passed, safety_reason, filtered_diagnosis = guardrails.full_check(
                failure_description,
                result["diagnosis"],
                result["confidence"],
                result["citations"],
                papers_text
            )
            
            result["safety_passed"] = safety_passed
            result["safety_reason"] = safety_reason
            result["diagnosis"] = filtered_diagnosis
            
            if not safety_passed:
                result["diagnosis"] = f"[UNABLE TO DIAGNOSE] {safety_reason}"
            
            return result
        
        except Exception as e:
            print(f"[ERROR] {str(e)[:100]}")
            return {
                "query": failure_description,
                "diagnosis": f"[ERROR] {str(e)[:100]}",
                "confidence": 0.0,
                "citations": [],
                "is_validated": False,
                "safety_passed": False
            }    

if __name__ == "__main__":
    print("="*80)
    print("Initializing LiDAR Failure Analyzer Agent (Smart Retry Enabled)")
    print("="*80)
    agent = LiDARFailureAnalyzer()
    
    test_failures = [
        "My BEV fusion model suddenly drops accuracy at night. All sensors working individually, but detection fails. Heavy fog conditions.",
        "LiDAR gives false positives in dense traffic. Radar and camera agree there's no obstacle, but LiDAR insists there is.",
        "Sensor fusion works in day, fails completely at night in urban areas. GPS also unreliable.",
        "After rain, camera lens gets water droplets. Fusion model confidence drops 40%. Wiper clears lens but model doesn't recover.",
    ]
    
    print("\n" + "="*80)
    print("TESTING LiDAR FAILURE ANALYZER AGENT")
    print("="*80)
    
    for i, failure in enumerate(test_failures, 1):
        print(f"\n{'='*80}")
        print(f"Scenario {i}:")
        print(f"{failure}")
        print(f"{'='*80}")
        
        result = agent.run(failure)
        
        confidence_pct = result['confidence'] * 100
        confidence_indicator = "[HIGH]" if result['confidence'] > 0.75 else "[LOW]"
        
        print(f"\n{confidence_indicator} DIAGNOSIS (Confidence: {confidence_pct:.0f}%, Retries: {result.get('retry_count', 0)}):")
        print(result['diagnosis'])
        
        if result['confidence'] < 0.75:
            print(f"\n[WARNING] Low confidence - answer may be uncertain")
        
        print(f"\nSOURCES ({len(result['citations'])} citations):")
        if result['citations']:
            for citation in result['citations']:
                print(f"  - {citation}")
        else:
            print(f"  (No citations available)")
        
        print(f"\nValidated: {'Yes' if result['is_validated'] else 'No'}")
import json
import re
from pathlib import Path
from statistics import mean
from rag_baseline import answer

EVAL_FILE = Path(__file__).resolve().parents[1] / "outputs" / "eval.jsonl"

def tokenize(text: str) -> list:
    """Simple tokenization for evaluation"""
    return re.findall(r'\w+', text.lower())

def f1_score(pred: str, gold: str) -> float:
    """Calculate F1 score between predicted and gold answers"""
    pred_tokens = tokenize(pred)
    gold_tokens = tokenize(gold)
    
    if not pred_tokens or not gold_tokens:
        return 0.0
    
    # Calculate precision and recall
    common_tokens = set(pred_tokens) & set(gold_tokens)
    
    if not common_tokens:
        return 0.0
    
    # Count overlapping tokens (with frequency)
    pred_counts = {t: pred_tokens.count(t) for t in set(pred_tokens)}
    gold_counts = {t: gold_tokens.count(t) for t in set(gold_tokens)}
    
    overlap = sum(min(pred_counts.get(t, 0), gold_counts.get(t, 0)) for t in common_tokens)
    
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def exact_match(pred: str, gold: str) -> bool:
    """Check if prediction exactly matches gold answer (after normalization)"""
    pred_normalized = ' '.join(tokenize(pred))
    gold_normalized = ' '.join(tokenize(gold))
    return pred_normalized == gold_normalized

def contains_answer(pred: str, gold: str) -> bool:
    """Check if prediction contains key concepts from gold answer"""
    pred_tokens = set(tokenize(pred))
    gold_tokens = set(tokenize(gold))
    
    # Remove common stop words for better matching
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'that', 'this', 'these', 'those'}
    
    gold_content_words = gold_tokens - stop_words
    pred_content_words = pred_tokens - stop_words
    
    if not gold_content_words:
        return True
    
    # Check if at least 30% of content words from gold are in prediction
    overlap = len(gold_content_words & pred_content_words)
    coverage = overlap / len(gold_content_words)
    
    return coverage >= 0.3

def run_evaluation():
    """Run evaluation on all questions in eval.jsonl"""
    if not EVAL_FILE.exists():
        print(f"❌ Evaluation file not found: {EVAL_FILE}")
        return
    
    # Load evaluation questions
    with open(EVAL_FILE, 'r', encoding='utf-8') as f:
        eval_data = [json.loads(line) for line in f]
    
    print(f"🔍 Running evaluation on {len(eval_data)} questions...")
    print("=" * 60)
    
    # Metrics storage
    f1_scores = []
    exact_matches = []
    answer_coverage = []
    
    for i, item in enumerate(eval_data, 1):
        question = item['q']
        gold_answer = item['a']
        
        print(f"\n📝 Question {i}: {question}")
        print(f"🎯 Expected: {gold_answer}")
        
        try:
            # Get prediction from RAG system
            predicted_answer = answer(question, k=4)
            print(f"🤖 Predicted: {predicted_answer}")
            
            # Calculate metrics
            f1 = f1_score(predicted_answer, gold_answer)
            em = exact_match(predicted_answer, gold_answer)
            coverage = contains_answer(predicted_answer, gold_answer)
            
            f1_scores.append(f1)
            exact_matches.append(em)
            answer_coverage.append(coverage)
            
            print(f"📊 Metrics: F1={f1:.3f}, EM={em}, Coverage={coverage}")
            
        except Exception as e:
            print(f"❌ Error processing question {i}: {e}")
            f1_scores.append(0.0)
            exact_matches.append(False)
            answer_coverage.append(False)
        
        print("-" * 40)
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("📈 EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Number of questions: {len(eval_data)}")
    print(f"Average F1 Score: {mean(f1_scores):.3f}")
    print(f"Exact Match Rate: {sum(exact_matches)}/{len(exact_matches)} ({sum(exact_matches)/len(exact_matches)*100:.1f}%)")
    print(f"Answer Coverage Rate: {sum(answer_coverage)}/{len(answer_coverage)} ({sum(answer_coverage)/len(answer_coverage)*100:.1f}%)")
    
    # Detailed breakdown
    print(f"\n📊 Detailed F1 Scores:")
    for i, (item, f1) in enumerate(zip(eval_data, f1_scores), 1):
        print(f"  {i}. {f1:.3f} - {item['q'][:50]}...")
    
    # Save results
    results = {
        "num_questions": len(eval_data),
        "avg_f1": mean(f1_scores),
        "exact_match_rate": sum(exact_matches) / len(exact_matches),
        "coverage_rate": sum(answer_coverage) / len(answer_coverage),
        "individual_f1_scores": f1_scores,
        "questions": [item['q'] for item in eval_data]
    }
    
    results_path = Path(__file__).resolve().parents[1] / "outputs" / "eval_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_path}")

if __name__ == "__main__":
    run_evaluation()

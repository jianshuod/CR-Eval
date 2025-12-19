import json
import math
from collections import defaultdict
from argparse import ArgumentParser


def calculate_pass_at_k(n, c, k):
    """
    Calculate pass@k metric.
    n: total number of samples
    c: number of correct samples
    k: k in pass@k
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def main():
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input jsonl file")
    parser.add_argument("--n", type=int, default=5, help="Number of samples per test case (N)")
    args = parser.parse_args()
    
    # Read the jsonl file
    test_cases = []
    with open(args.input, 'r') as f:
        for line in f:
            if line.strip():
                test_cases.append(json.loads(line))
    
    # Process each test case
    all_results = []
    for item in test_cases:
        # Extract scores from eval_results
        scores = [result["score"] > 1 for result in item["eval_results"].values()]
        
        # Take only the first N scores
        scores = scores[:args.n]
        all_results.append(scores)
    
    # Calculate pass@k for k=1, 3, 5
    k_values = [1, 3, 5]
    pass_at_k_results = {k: [] for k in k_values}
    
    for scores in all_results:
        n = len(scores)
        c = sum(scores)  # number of correct samples
        
        for k in k_values:
            if k <= n:
                pass_at_k_results[k].append(calculate_pass_at_k(n, c, k))
    
    # Calculate average pass@k
    print(f"\nResults for {args.input}:")
    print(f"Total test cases: {len(all_results)}")
    print(f"Samples per test case (N): {args.n}")
    print("-" * 50)
    
    for k in k_values:
        if pass_at_k_results[k]:
            avg_pass_at_k = sum(pass_at_k_results[k]) / len(pass_at_k_results[k])
            print(f"pass@{k}: {avg_pass_at_k* 200:.4f} ({avg_pass_at_k * 100:.2f}%)")
        else:
            print(f"pass@{k}: N/A (k > n)")


if __name__ == "__main__":
    main()

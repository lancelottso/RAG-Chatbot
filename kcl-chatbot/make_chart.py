import json
import matplotlib.pyplot as plt

# Load evaluation results
with open("data/evaluation_results_attempt15_30benchmarkqueries.json") as f: # File name can be changed depending on evaluation results
    results = json.load(f)

# Keep only in-domain queries (exclude the capital of France one)
in_domain = [r for r in results if r["expected_topic"] != "__out_of_domain__"]

# Extract query labels and recall scores
labels = [r["expected_topic"] for r in in_domain]
recalls = [r["similarity"]["recall"] if r["similarity"] else 0 for r in in_domain]

# Sort by recall (ascending) so the chart shows a clear gradient from worst to best
sorted_data = sorted(zip(recalls, labels))
recalls_sorted = [r for r, _ in sorted_data]
labels_sorted = [l for _, l in sorted_data]

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Horizontal bars
bars = ax.barh(labels_sorted, recalls_sorted, color="#5b7ff5", edgecolor="#1a3a80")

# Add average line
avg = sum(recalls_sorted) / len(recalls_sorted)
ax.axvline(avg, color="red", linestyle="--", linewidth=1.5, label=f"Average: {avg:.3f}")

# Formatting
ax.set_xlabel("Recall Score", fontsize=12)
ax.set_ylabel("Query Topic", fontsize=12)
ax.set_title("Recall Distribution Across Benchmark Queries", fontsize=13, pad=15)
ax.set_xlim(0, 1.0)
ax.legend(loc="lower right")
ax.grid(axis="x", linestyle=":", alpha=0.5)

# Tight layout
plt.tight_layout()

# Save to file
plt.savefig("data/recall_distribution.png", dpi=150, bbox_inches="tight")
print("Chart saved to data/recall_distribution.png")

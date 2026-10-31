"""Corpus summaries and optional diagnostic plots."""

from collections import Counter, defaultdict


def compute_corpus_metrics(cleaned_docs: list[dict]) -> dict:
    diagnostics = {
        name: Counter()
        for name in ("unmapped_section_types", "unmapped_fig_types", "unmapped_suppl_headings")
    }
    canonical_counts = Counter()
    synthetic_by_canonical = defaultdict(lambda: [0, 0])
    doc_word_totals = []
    section_word_counts = []
    synthetic_per_doc = []
    year_counts = Counter()
    for doc in cleaned_docs:
        for name in diagnostics:
            diagnostics[name].update(doc.get("diagnostics", {}).get(name, {}))
        doc_words = 0
        year_counts[doc.get("year") or "unknown"] += 1
        synthetic_per_doc.append(doc["n_synthetic_sections"])
        for s in doc["sections"]:
            canonical_counts[s["canonical_section_type"]] += 1
            section_word_counts.append(s["n_words"])
            doc_words += s["n_words"]
            bucket = synthetic_by_canonical[s["canonical_section_type"]]
            bucket[1] += 1
            if s["synthetic"]:
                bucket[0] += 1
        doc_word_totals.append(doc_words)
    return {
        "canonical_counts": canonical_counts,
        "synthetic_by_canonical": synthetic_by_canonical,
        "doc_word_totals": doc_word_totals,
        "section_word_counts": section_word_counts,
        "synthetic_per_doc": synthetic_per_doc,
        "year_counts": year_counts,
        **diagnostics,
    }


def plot_corpus_metrics(metrics: dict, output_path: str = "corpus_metrics.png") -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ax = axes[0, 0]
    pairs = metrics["canonical_counts"].most_common()
    labels, counts = zip(*pairs) if pairs else ([], [])
    ax.bar(labels, counts, color="steelblue")
    ax.set_title("Section count by canonical type")
    ax.tick_params(axis="x", rotation=60)
    ax = axes[0, 1]
    cats = list(metrics["synthetic_by_canonical"].keys())
    synth_vals = [metrics["synthetic_by_canonical"][c][0] for c in cats]
    real_vals = [
        metrics["synthetic_by_canonical"][c][1] - metrics["synthetic_by_canonical"][c][0]
        for c in cats
    ]
    ax.bar(cats, real_vals, label="real heading", color="seagreen")
    ax.bar(cats, synth_vals, bottom=real_vals, label="synthetic", color="indianred")
    ax.set_title("Synthetic vs. real sections by type")
    ax.tick_params(axis="x", rotation=60)
    ax.legend()
    ax = axes[0, 2]
    max_synth = max(metrics["synthetic_per_doc"], default=0)
    ax.hist(metrics["synthetic_per_doc"], bins=range(0, max_synth + 2), color="darkorange")
    ax.set_title("Synthetic sections per document")
    ax.set_xlabel("# synthetic sections")
    ax.set_ylabel("# documents")
    ax = axes[1, 0]
    ax.hist(metrics["section_word_counts"], bins=40, color="mediumpurple")
    ax.set_title("Section length (words)")
    ax.set_xlabel("words per section")
    ax = axes[1, 1]
    ax.hist(metrics["doc_word_totals"], bins=30, color="teal")
    ax.set_title("Document length (total words)")
    ax.set_xlabel("words per document")
    ax = axes[1, 2]
    years = sorted(metrics["year_counts"].keys(), key=lambda y: (str(y) == "unknown", str(y)))
    ax.bar(years, [metrics["year_counts"][y] for y in years], color="slategray")
    ax.set_title("Documents by publication year")
    ax.tick_params(axis="x", rotation=60)
    plt.tight_layout()
    from pathlib import Path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    if metrics["unmapped_section_types"]:
        print("WARNING -- unmapped section_types (bucketed as 'other'):")
        for st, count in metrics["unmapped_section_types"].most_common():
            print(f"  {st}: {count}")
    if metrics["unmapped_fig_types"]:
        print("WARNING -- unmapped FIG passage types (bucketed as 'figure_other'):")
        for st, count in metrics["unmapped_fig_types"].most_common():
            print(f"  {st}: {count}")
    if metrics["unmapped_suppl_headings"]:
        print(
            "\nSUPPL headings currently bucketed as 'supplementary_other' -- REVIEW THESE and split into real categories, don't leave as-is:"
        )
        for heading, count in metrics["unmapped_suppl_headings"].most_common(30):
            print(f"  [{count:4d}] {heading!r}")

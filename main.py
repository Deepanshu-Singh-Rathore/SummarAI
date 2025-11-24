from summarizer import summarize_text, evaluate_summary

# Sample text and reference summary for demonstration
text = """
Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. 
The goal is a computer capable of "understanding" the contents of documents, including the contextual nuances of the language within them. 
The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves.
"""
reference_summary = "NLP enables computers to understand and process human language, extracting insights from text."

# Generate summary
generated_summary = summarize_text(text, summary_length=1)

# Evaluate the summary
scores = evaluate_summary(generated_summary, reference_summary)

print("Generated Summary:\n", generated_summary)
print("\nEvaluation Scores:\n", scores)


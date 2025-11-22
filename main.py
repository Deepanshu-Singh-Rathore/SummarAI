from summarizer import summarize_text

text = open("data/sample.txt", "r").read()
print(summarize_text(text))

import re

def suggest_tip(entry):
    entry = entry.lower()
    if re.search(r'\boverwhelmed\b|\banxious\b|\bstressed\b', entry):
        return "Try a 4-7-8 breathing exercise or a short walk."
    elif re.search(r'\btired\b|\bdrained\b', entry):
        return "Consider a 10-minute mindfulness break or journaling."
    else:
        return "Thanks for sharing. Keep expressing yourself!"

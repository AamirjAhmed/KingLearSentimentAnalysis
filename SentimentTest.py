import re
import nltk                                                                                             # type: ignore
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer                                    # type: ignore
import pandas as pd                                                                                     # type: ignore
import matplotlib.pyplot as plt                                                                         # type: ignore
import seaborn as sns                                                                                   # type: ignore

def read_file_with_skip(file_path, skip_lines=0):
    """Reads the entire file as a single string, optionally skipping the first `skip_lines` lines."""
    with open(file_path, "r", encoding="utf-8") as f:
        if skip_lines > 0:
            return "".join(f.readlines()[skip_lines:])
        return f.read()

# --- File-based Extraction Functions ---

def extract_fool_dialogue(file_path):
    """Extracts the Fool’s dialogue from the play text."""
    text = read_file_with_skip(file_path, skip_lines=106) if file_path == "KingLear.txt" else read_file_with_skip(file_path)
    pattern = re.compile(r'(?<=\nFOOL\.\n)(.*?)(?=\n(?:[A-Z][A-Z\s,\-\'\.]+\.|ACT\s+[IVX]+\.*|SCENE\s+\w+\.|\[.*?\]))', re.DOTALL)
    blocks = pattern.findall(text)
    return [re.sub(r'\[.*?\]', '', b, flags=re.DOTALL).strip() for b in blocks if b.strip()]

def extract_lear_dialogue(file_path):
    """Extracts King Lear’s dialogue from the play text."""
    text = read_file_with_skip(file_path, skip_lines=106) if file_path == "KingLear.txt" else read_file_with_skip(file_path)
    pattern = re.compile(r'(?<=\nLEAR\.\n)(.*?)(?=\n(?:[A-Z][A-Z\s,\-\'\.]+\.|ACT\s+[IVX]+\.*|SCENE\s+\w+\.|\[.*?\]))', re.DOTALL)
    blocks = pattern.findall(text)
    return [re.sub(r'\[.*?\]', '', b, flags=re.DOTALL).strip() for b in blocks if b.strip()]

def extract_edgar_dialogue(file_path):
    """Extracts Edgar’s (and Poor Tom's) dialogue from the play text."""
    text = read_file_with_skip(file_path, skip_lines=106) if file_path == "KingLear.txt" else read_file_with_skip(file_path)
    pattern = re.compile(r'(?<=\nEDGAR\.\n)(.*?)(?=\n(?:[A-Z][A-Z\s,\-\'\.]+\.|ACT\s+[IVX]+\.*|SCENE\s+\w+\.|\[.*?\]))', re.DOTALL)
    blocks = pattern.findall(text)
    return [re.sub(r'\[.*?\]', '', b, flags=re.DOTALL).strip() for b in blocks if b.strip()]

def extract_goneril_dialogue(file_path):
    """Extracts GONERIL’s dialogue from the play text."""
    text = read_file_with_skip(file_path, skip_lines=106) if file_path == "KingLear.txt" else read_file_with_skip(file_path)
    pattern = re.compile(r'(?<=\nGONERIL\.\n)(.*?)(?=\n(?:[A-Z][A-Z\s,\-\'\.]+\.|ACT\s+[IVX]+\.*|SCENE\s+\w+\.|\[.*?\]))', re.DOTALL)
    blocks = pattern.findall(text)
    return [re.sub(r'\[.*?\]', '', b, flags=re.DOTALL).strip() for b in blocks if b.strip()]

def extract_regan_dialogue(file_path):
    """Extracts REGAN’s dialogue from the play text."""
    text = read_file_with_skip(file_path, skip_lines=106) if file_path == "KingLear.txt" else read_file_with_skip(file_path)
    pattern = re.compile(
        r'(?<=\nREGAN\.\n)(.*?)(?=\n(?:[A-Z][A-Z\s,\-\'\.]+\.|ACT\s+[IVX]+\.*|SCENE\s+\w+\.|\[.*?\]))',
        re.DOTALL
    )
    blocks = pattern.findall(text)
    return [re.sub(r'\[.*?\]', '', b, flags=re.DOTALL).strip() for b in blocks if b.strip()]

def extract_cordelia_dialogue(file_path):
    """Extracts CORDELIA’s dialogue from the play text."""
    text = read_file_with_skip(file_path, skip_lines=106) if file_path == "KingLear.txt" else read_file_with_skip(file_path)
    pattern = re.compile(r'(?<=\nCORDELIA\.\n)(.*?)(?=\n(?:[A-Z][A-Z\s,\-\'\.]+\.|ACT\s+[IVX]+\.*|SCENE\s+\w+\.|\[.*?\]))', re.DOTALL)
    blocks = pattern.findall(text)
    return [re.sub(r'\[.*?\]', '', b, flags=re.DOTALL).strip() for b in blocks if b.strip()]

# --- Scene-level Extraction Functions ---

def extract_fool_dialogue_from_text(text):
    pattern = re.compile(r'(?<=\nFOOL\.\n)(.*?)(?=\n(?:[A-Z][A-Z\s,\-\'\.]+\.|ACT\s+[IVX]+\.*|SCENE\s+\w+\.|\[.*?\]))', re.DOTALL)
    blocks = pattern.findall(text)
    return [re.sub(r'\[.*?\]', '', b, flags=re.DOTALL).strip() for b in blocks if b.strip()]

def extract_lear_dialogue_from_text(text):
    pattern = re.compile(r'(?<=\nLEAR\.\n)(.*?)(?=\n(?:[A-Z][A-Z\s,\-\'\.]+\.|ACT\s+[IVX]+\.*|SCENE\s+\w+\.|\[.*?\]))', re.DOTALL)
    blocks = pattern.findall(text)
    return [re.sub(r'\[.*?\]', '', b, flags=re.DOTALL).strip() for b in blocks if b.strip()]

def extract_edgar_dialogue_from_text(text):
    pattern = re.compile(r'(?<=\nEDGAR\.\n)(.*?)(?=\n(?:[A-Z][A-Z\s,\-\'\.]+\.|ACT\s+[IVX]+\.*|SCENE\s+\w+\.|\[.*?\]))', re.DOTALL)
    blocks = pattern.findall(text)
    return [re.sub(r'\[.*?\]', '', b, flags=re.DOTALL).strip() for b in blocks if b.strip()]

def extract_goneril_dialogue_from_text(text):
    pattern = re.compile(r'(?<=\nGONERIL\.\n)(.*?)(?=\n(?:[A-Z][A-Z\s,\-\'\.]+\.|ACT\s+[IVX]+\.*|SCENE\s+\w+\.|\[.*?\]))', re.DOTALL)
    blocks = pattern.findall(text)
    return [re.sub(r'\[.*?\]', '', b, flags=re.DOTALL).strip() for b in blocks if b.strip()]

def extract_regan_dialogue_from_text(text):
    pattern = re.compile(r'(?<=\nREGAN\.\n)(.*?)(?=\n(?:[A-Z][A-Z\s,\-\'\.]+\.|ACT\s+[IVX]+\.*|SCENE\s+\w+\.|\[.*?\]))', re.DOTALL)
    blocks = pattern.findall(text)
    return [re.sub(r'\[.*?\]', '', b, flags=re.DOTALL).strip() for b in blocks if b.strip()]

def extract_cordelia_dialogue_from_text(text):
    pattern = re.compile(r'(?<=\nCORDELIA\.\n)(.*?)(?=\n(?:[A-Z][A-Z\s,\-\'\.]+\.|ACT\s+[IVX]+\.*|SCENE\s+\w+\.|\[.*?\]))', re.DOTALL)
    blocks = pattern.findall(text)
    return [re.sub(r'\[.*?\]', '', b, flags=re.DOTALL).strip() for b in blocks if b.strip()]

###########################################################################################################################################################
                                                        # Main Program
###########################################################################################################################################################

file_path = "KingLear.txt"  # Change this if needed

try:
    choice = input(
        "Enter:\n"
        "  1 for Fool's dialogue sentiment analysis\n"
        "  2 for King Lear's dialogue sentiment analysis\n"
        "  3 for Edgar's dialogue sentiment analysis\n"
        "  4 for GONERIL's dialogue sentiment analysis\n"
        "  5 for REGAN's dialogue sentiment analysis\n"
        "  6 for CORDELIA's dialogue sentiment analysis\n"
        "  7 for a combined normalized (by scene) plot for Fool, King Lear, and Edgar\n"
        "  8 for a combined normalized (by scene) plot for all characters\n"
        "  9 for a combined normalized (by scene) plot for all three sisters\n"
        " 10 for a combined normalized (by scene) plot for GONERIL and REGAN\n"
        " 11 for the top 3 overall most negative and top 3 overall most positive statements\n"
        "  (or q to quit): "
    ).strip()
except KeyboardInterrupt:
    print("\nExiting...")
    exit(0)

if choice.lower() == "q":
    print("Exiting...")
    exit(0)

analyzer = SentimentIntensityAnalyzer()

# --- Options 1-6: Individual Analysis and Plotting ---
if choice in ["1", "2", "3", "4", "5", "6"]:
    if choice == "1":
        dialogue_blocks = extract_fool_dialogue(file_path)
        speaker = "Fool"
    elif choice == "2":
        dialogue_blocks = extract_lear_dialogue(file_path)
        speaker = "King Lear"
    elif choice == "3":
        dialogue_blocks = extract_edgar_dialogue(file_path)
        speaker = "Edgar (and Poor Tom)"
    elif choice == "4":
        dialogue_blocks = extract_goneril_dialogue(file_path)
        speaker = "GONERIL"
    elif choice == "5":
        dialogue_blocks = extract_regan_dialogue(file_path)
        speaker = "REGAN"
    elif choice == "6":
        dialogue_blocks = extract_cordelia_dialogue(file_path)
        speaker = "CORDELIA"
    
    if not dialogue_blocks:
        print(f"No {speaker}'s dialogue was found.")
        exit(1)
    
    sentiment_results = []
    for i, block in enumerate(dialogue_blocks, start=1):
        sentiment = analyzer.polarity_scores(block)
        sentiment_results.append({
            'block_index': i,
            'text': block,
            'compound': sentiment['compound']
        })
    
    df = pd.DataFrame(sentiment_results)
    print(f"\nFirst few rows of {speaker}'s sentiment DataFrame:")
    print(df.head())
    print(f"\nAverage compound score for {speaker}: {df['compound'].mean()}")
    
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x='block_index', y='compound', marker="o", picker=True)
    plt.title(f"Compound Sentiment Score Progression in {speaker}'s Speeches")
    plt.xlabel("Block Index")
    plt.ylabel("Compound Score")
    plt.ylim(-1, 1)  # Set the y-axis from -1 to 1
    plt.grid(True)
    plt.tight_layout()
    
    fig = plt.gcf()
    def onpick(event):
        if event.ind:
            idx = event.ind[0]
            row = df.iloc[idx]
            print(f"\nClicked on Block #{row['block_index']}:")
            print(row['text'])
            print("-" * 40)
    fig.canvas.mpl_connect('pick_event', onpick)
    plt.show()

# --- Options 7-10: Combined Normalized (by Scene) Plots ---
elif choice in ["7", "8", "9", "10"]:
    full_text = read_file_with_skip(file_path, skip_lines=106 if file_path=="KingLear.txt" else 0)
    scenes = re.split(r'\n(?=SCENE\s+[IVX]+\.)', full_text)
    if not scenes:
        print("No scenes found.")
        exit(1)
    scene_data = []
    if choice == "7":
        mapping = {
            'Fool': extract_fool_dialogue_from_text,
            'King Lear': extract_lear_dialogue_from_text,
            'Edgar (and Poor Tom)': extract_edgar_dialogue_from_text
        }
        title = "Fool, King Lear, and Edgar"
    elif choice == "8":
        mapping = {
            'Fool': extract_fool_dialogue_from_text,
            'King Lear': extract_lear_dialogue_from_text,
            'Edgar (and Poor Tom)': extract_edgar_dialogue_from_text,
            'GONERIL': extract_goneril_dialogue_from_text,
            'REGAN': extract_regan_dialogue_from_text,
            'CORDELIA': extract_cordelia_dialogue_from_text
        }
        title = "All Characters"
    elif choice == "9":
        mapping = {
            'GONERIL': extract_goneril_dialogue_from_text,
            'REGAN': extract_regan_dialogue_from_text,
            'CORDELIA': extract_cordelia_dialogue_from_text
        }
        title = "All Three Sisters"
    elif choice == "10":
        mapping = {
            'GONERIL': extract_goneril_dialogue_from_text,
            'REGAN': extract_regan_dialogue_from_text
        }
        title = "GONERIL and REGAN"
    
    for scene_index, scene_text in enumerate(scenes, start=1):
        for speaker_name, extractor in mapping.items():
            dialogues = extractor(scene_text)
            if dialogues:
                scores = [analyzer.polarity_scores(d)['compound'] for d in dialogues]
                avg = sum(scores) / len(scores)
                scene_data.append({'scene': scene_index, 'speaker': speaker_name, 'avg_compound': avg})
    if not scene_data:
        print("No dialogue data found for the selected speakers.")
        exit(1)
    df_scene = pd.DataFrame(scene_data)
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_scene, x='scene', y='avg_compound', hue='speaker', marker="o")
    plt.title(f"Normalized (by Scene) Compound Sentiment Score for {title}")
    plt.xlabel("Scene Number")
    plt.ylabel("Average Compound Score")
    plt.ylim(-1, 1)  # Set the y-axis from -1 to 1
    plt.grid(True)
    plt.legend(title="Speaker")
    plt.tight_layout()
    plt.show()

# --- Option 11: Top 3 Overall Most Negative and Top 3 Overall Most Positive Statements ---
elif choice == "11":
    # Gather dialogue blocks from all characters
    speakers = {
        "Fool": extract_fool_dialogue,
        "King Lear": extract_lear_dialogue,
        "Edgar (and Poor Tom)": extract_edgar_dialogue,
        "GONERIL": extract_goneril_dialogue,
        "REGAN": extract_regan_dialogue,
        "CORDELIA": extract_cordelia_dialogue
    }
    all_results = []
    for speaker, extractor in speakers.items():
        dialogues = extractor(file_path)
        for i, dialogue in enumerate(dialogues, start=1):
            comp = analyzer.polarity_scores(dialogue)['compound']
            all_results.append({"speaker": speaker, "block_index": i, "text": dialogue, "compound": comp})
    negative_sorted = sorted(all_results, key=lambda x: x["compound"])
    positive_sorted = sorted(all_results, key=lambda x: x["compound"], reverse=True)
    top_negative = negative_sorted[:5]
    top_positive = positive_sorted[:5]
    
    print("\n"*10)
    print("\nTop 5 Most Negative Statements Overall:")
    for r in top_negative:
        print(f"Speaker: {r['speaker']}, Block #{r['block_index']} (compound: {r['compound']}):")
        print(f"    {r['text']}")
        print("    " + "-" * 40)
    
    print("\n"*10)

    print("\nTop 5 Most Positive Statements Overall:")
    for r in top_positive:
        print(f"Speaker: {r['speaker']}, Block #{r['block_index']} (compound: {r['compound']}):")
        print(f"    {r['text']}")
        print("    " + "-" * 40)

else:
    print("Invalid choice. Please run the program again and enter a valid option (1-11 or q to quit).")

plt.show()



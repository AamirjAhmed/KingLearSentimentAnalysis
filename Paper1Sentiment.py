import re
import nltk    #type: ignore
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore

def read_file_with_skip(file_path, skip_lines=0):
    """Reads the entire file as a single string, optionally skipping the first `skip_lines` lines."""
    with open(file_path, "r", encoding="utf-8") as f:
        if skip_lines > 0:
            return "".join(f.readlines()[skip_lines:])
        return f.read()

# --- File-based Extraction Functions for King Lear and the Three Sisters ---

def extract_lear_dialogue(file_path):
    """Extracts King Lear's dialogue from the play text."""
    text = read_file_with_skip(file_path, skip_lines=106) if file_path == "KingLear.txt" else read_file_with_skip(file_path)
    pattern = re.compile(
        r'(?<=\nLEAR\.\n)(.*?)(?=\n(?:[A-Z][A-Z\s,\-\'\.]+\.|ACT\s+[IVX]+\.*|SCENE\s+\w+\.|\[.*?\]))',
        re.DOTALL
    )
    blocks = pattern.findall(text)
    return [re.sub(r'\[.*?\]', '', b, flags=re.DOTALL).strip() for b in blocks if b.strip()]

def extract_goneril_dialogue(file_path):
    """Extracts GONERIL's dialogue from the play text."""
    text = read_file_with_skip(file_path, skip_lines=106) if file_path == "KingLear.txt" else read_file_with_skip(file_path)
    pattern = re.compile(
        r'(?<=\nGONERIL\.\n)(.*?)(?=\n(?:[A-Z][A-Z\s,\-\'\.]+\.|ACT\s+[IVX]+\.*|SCENE\s+\w+\.|\[.*?\]))',
        re.DOTALL
    )
    blocks = pattern.findall(text)
    return [re.sub(r'\[.*?\]', '', b, flags=re.DOTALL).strip() for b in blocks if b.strip()]

def extract_regan_dialogue(file_path):
    """Extracts REGAN's dialogue from the play text."""
    text = read_file_with_skip(file_path, skip_lines=106) if file_path == "KingLear.txt" else read_file_with_skip(file_path)
    pattern = re.compile(
        r'(?<=\nREGAN\.\n)(.*?)(?=\n(?:[A-Z][A-Z\s,\-\'\.]+\.|ACT\s+[IVX]+\.*|SCENE\s+\w+\.|\[.*?\]))',
        re.DOTALL
    )
    blocks = pattern.findall(text)
    return [re.sub(r'\[.*?\]', '', b, flags=re.DOTALL).strip() for b in blocks if b.strip()]

def extract_cordelia_dialogue(file_path):
    """Extracts CORDELIA's dialogue from the play text."""
    text = read_file_with_skip(file_path, skip_lines=106) if file_path == "KingLear.txt" else read_file_with_skip(file_path)
    pattern = re.compile(
        r'(?<=\nCORDELIA\.\n)(.*?)(?=\n(?:[A-Z][A-Z\s,\-\'\.]+\.|ACT\s+[IVX]+\.*|SCENE\s+\w+\.|\[.*?\]))',
        re.DOTALL
    )
    blocks = pattern.findall(text)
    return [re.sub(r'\[.*?\]', '', b, flags=re.DOTALL).strip() for b in blocks if b.strip()]

# --- Scene-level Extraction Functions (for combined analysis) ---
# (Used in Option 8 for Scene 1)
def extract_lear_dialogue_from_text(text):
    pattern = re.compile(
        r'(?<=\nLEAR\.\n)(.*?)(?=\n(?:[A-Z][A-Z\s,\-\'\.]+\.|ACT\s+[IVX]+\.*|SCENE\s+\w+\.|\[.*?\]))',
        re.DOTALL
    )
    blocks = pattern.findall(text)
    return [re.sub(r'\[.*?\]', '', b, flags=re.DOTALL).strip() for b in blocks if b.strip()]

def extract_goneril_dialogue_from_text(text):
    pattern = re.compile(
        r'(?<=\nGONERIL\.\n)(.*?)(?=\n(?:[A-Z][A-Z\s,\-\'\.]+\.|ACT\s+[IVX]+\.*|SCENE\s+\w+\.|\[.*?\]))',
        re.DOTALL
    )
    blocks = pattern.findall(text)
    return [re.sub(r'\[.*?\]', '', b, flags=re.DOTALL).strip() for b in blocks if b.strip()]

def extract_regan_dialogue_from_text(text):
    pattern = re.compile(
        r'(?<=\nREGAN\.\n)(.*?)(?=\n(?:[A-Z][A-Z\s,\-\'\.]+\.|ACT\s+[IVX]+\.*|SCENE\s+\w+\.|\[.*?\]))',
        re.DOTALL
    )
    blocks = pattern.findall(text)
    return [re.sub(r'\[.*?\]', '', b, flags=re.DOTALL).strip() for b in blocks if b.strip()]

def extract_cordelia_dialogue_from_text(text):
    pattern = re.compile(
        r'(?<=\nCORDELIA\.\n)(.*?)(?=\n(?:[A-Z][A-Z\s,\-\'\.]+\.|ACT\s+[IVX]+\.*|SCENE\s+\w+\.|\[.*?\]))',
        re.DOTALL
    )
    blocks = pattern.findall(text)
    return [re.sub(r'\[.*?\]', '', b, flags=re.DOTALL).strip() for b in blocks if b.strip()]

# --- New Option 7: Breakdown of Sentiment Components (Normalized to Percentages) ---
def breakdown_sentiment_components(dialogue_blocks, analyzer):
    """
    For each dialogue block, computes sentiment scores and converts them into percentages.
    Returns a list of dictionaries with keys: block_index, compound, pos, neu, neg, and text.
    
    (Compound is multiplied by 100 so that its range becomes approximately -100 to +100,
     and pos, neu, neg are multiplied by 100 to yield percentages.)
    """
    results = []
    for i, block in enumerate(dialogue_blocks, start=1):
        scores = analyzer.polarity_scores(block)
        scores['compound'] = scores['compound'] * 100
        scores['pos'] = scores['pos'] * 100
        scores['neu'] = scores['neu'] * 100
        scores['neg'] = scores['neg'] * 100
        scores.update({"block_index": i, "text": block})
        results.append(scores)
    return results

# --- New Option 8: Additional Visualizations for Scene 1 ---
def option_8_visualizations(file_path, analyzer, output):
    """
    For Scene 1 only, produces:
      Block A: Extraction of Scene 1 data (dialogue and sentiment scores in percentages).
      Block B: Heatmap visualizations of sentiment components (pos, neu, neg) for each character.
      Block C: Scatter plot of compound scores (in percentages) per dialogue block.
      Block D: Sliding window analysis (window size 3) of compound scores.
    """
    # --- Block A: Extract Scene 1 Data ---
    full_text = read_file_with_skip(file_path, skip_lines=106 if file_path=="KingLear.txt" else 0)
    scenes = re.split(r'\n(?=SCENE\s+[IVX]+\.)', full_text)
    if not scenes:
        msg = "No scenes found in the text.\n"
        print(msg)
        output.write(msg)
        return
    scene1 = scenes[0]  # Assume Scene 1 is the first scene.
    mapping = {
        'King Lear': extract_lear_dialogue_from_text,
        'GONERIL': extract_goneril_dialogue_from_text,
        'REGAN': extract_regan_dialogue_from_text,
        'CORDELIA': extract_cordelia_dialogue_from_text
    }
    scene1_data = []
    for speaker, extractor in mapping.items():
        dialogues = extractor(scene1)
        for i, dialogue in enumerate(dialogues, start=1):
            scores = analyzer.polarity_scores(dialogue)
            scene1_data.append({
                "speaker": speaker,
                "block_index": i,
                "compound": scores['compound'] * 100,
                "pos": scores['pos'] * 100,
                "neu": scores['neu'] * 100,
                "neg": scores['neg'] * 100
            })
    if not scene1_data:
        msg = "No dialogue data found in Scene 1 for the selected speakers.\n"
        print(msg)
        output.write(msg)
        return
    df_scene1 = pd.DataFrame(scene1_data)
    
    # --- Sliding Window Analysis ---
    window_size = 3
    sliding_data = []
    for speaker in mapping.keys():
        df_speaker = df_scene1[df_scene1['speaker'] == speaker].sort_values('block_index')
        if len(df_speaker) < window_size:
            continue
        df_speaker = df_speaker.copy()
        df_speaker['compound_window'] = df_speaker['compound'].rolling(window=window_size).mean()
        df_speaker['window_index'] = df_speaker['block_index'].rolling(window=window_size).mean()
        sliding_data.append(df_speaker[['window_index', 'compound_window', 'speaker']])
    if sliding_data:
        df_sliding = pd.concat(sliding_data)
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_sliding, x='window_index', y='compound_window', hue='speaker', marker="o")
        plt.title("Sliding Window (size 3) Average of Compound Scores in Scene 1")
        plt.xlabel("Average Block Index")
        plt.ylabel("Average Compound Score (%)")
        plt.ylim(-100, 100)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        msg = "Not enough data for sliding window analysis in Scene 1.\n"
        print(msg)
        output.write(msg)
    
    output.write("Option 8: Additional Visualizations for Scene 1 completed.\n")
    print("Option 8: Additional Visualizations for Scene 1 completed.\n")

# --- Main Program ---
file_path = "KingLear.txt"  # Adjust this path if needed

# Open output file for writing all analysis results.
output_filename = "output.txt"
output = open(output_filename, "w", encoding="utf-8")

try:
    choice = input(
        "Enter:\n"
        "  1 for King Lear's dialogue sentiment analysis\n"
        "  2 for GONERIL's dialogue sentiment analysis\n"
        "  3 for REGAN's dialogue sentiment analysis\n"
        "  4 for CORDELIA's dialogue sentiment analysis\n"
        "  5 for a combined normalized (by scene) plot for King Lear and the Three Sisters\n"
        "  6 for printing top 3 most positive and top 3 most negative sentiments overall\n"
        "  7 for breakdown of sentiment components (compound, pos, neu, neg as percentages) for a chosen character\n"
        "  8 for additional visualizations (heatmaps, scatter plots, sliding window analysis) for Scene 1\n"
        "  (or q to quit): "
    ).strip()
except KeyboardInterrupt:
    print("\nExiting...")
    output.write("Exiting due to KeyboardInterrupt.\n")
    output.close()
    exit(0)

if choice.lower() == "q":
    print("Exiting...")
    output.write("Exiting by user request.\n")
    output.close()
    exit(0)

analyzer = SentimentIntensityAnalyzer()

if choice in ["1", "2", "3", "4"]:
    if choice == "1":
        dialogue_blocks = extract_lear_dialogue(file_path)
        speaker = "King Lear"
    elif choice == "2":
        dialogue_blocks = extract_goneril_dialogue(file_path)
        speaker = "GONERIL"
    elif choice == "3":
        dialogue_blocks = extract_regan_dialogue(file_path)
        speaker = "REGAN"
    elif choice == "4":
        dialogue_blocks = extract_cordelia_dialogue(file_path)
        speaker = "CORDELIA"
    
    if not dialogue_blocks:
        msg = f"No {speaker}'s dialogue was found.\n"
        print(msg)
        output.write(msg)
        output.close()
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
    
    header = f"\nFirst few rows of {speaker}'s sentiment DataFrame:\n{df.head()}\n"
    avg_score = f"\nAverage compound score for {speaker}: {df['compound'].mean()}\n"
    
    output.write(f"Sentiment Analysis for {speaker}\n")
    output.write(header)
    output.write(avg_score)
    output.write("\nDetailed Analysis:\n")
    for index, row in df.iterrows():
        output.write(f"Block #{row['block_index']} (Compound Score: {row['compound']}):\n")
        output.write(row['text'] + "\n")
        output.write("-" * 40 + "\n")
    
    print(header)
    print(avg_score)
    
    plt.figure(figsize=(10, 5))
    ax = sns.lineplot(data=df, x='block_index', y='compound', marker="o")
    ax.lines[0].set_picker(5)
    
    def onpick(event):
        if event.ind:
            idx = event.ind[0]
            row = df.iloc[idx]
            msg = f"\nClicked on Block #{row['block_index']}:\n{row['text']}\n{'-' * 40}\n"
            print(msg)
            output.write(msg)
    
    fig = plt.gcf()
    fig.canvas.mpl_connect('pick_event', onpick)
    plt.title(f"Compound Sentiment Score Progression in {speaker}'s Speeches")
    plt.xlabel("Block Index")
    plt.ylabel("Compound Score")
    plt.ylim(-1, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

elif choice == "5":
    full_text = read_file_with_skip(file_path, skip_lines=106 if file_path=="KingLear.txt" else 0)
    scenes = re.split(r'\n(?=SCENE\s+[IVX]+\.)', full_text)
    if not scenes:
        msg = "No scenes found.\n"
        print(msg)
        output.write(msg)
        output.close()
        exit(1)
    
    scene_data = []
    mapping = {
        'King Lear': extract_lear_dialogue_from_text,
        'GONERIL': extract_goneril_dialogue_from_text,
        'REGAN': extract_regan_dialogue_from_text,
        'CORDELIA': extract_cordelia_dialogue_from_text
    }
    
    for scene_index, scene_text in enumerate(scenes, start=1):
        for speaker_name, extractor in mapping.items():
            dialogues = extractor(scene_text)
            if dialogues:
                scores = [analyzer.polarity_scores(d)['compound'] for d in dialogues]
                avg = sum(scores) / len(scores)
                scene_data.append({'scene': scene_index, 'speaker': speaker_name, 'avg_compound': avg})
    
    if not scene_data:
        msg = "No dialogue data found for the selected speakers.\n"
        print(msg)
        output.write(msg)
        output.close()
        exit(1)
    
    df_scene = pd.DataFrame(scene_data)
    header = f"\nScene-level Normalized Compound Sentiment Score:\n{df_scene}\n"
    
    output.write("Scene-level Analysis for King Lear and the Three Sisters\n")
    output.write(header)
    print(header)
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_scene, x='scene', y='avg_compound', hue='speaker', marker="o")
    plt.title("Normalized (by Scene) Compound Sentiment Score for King Lear and the Three Sisters")
    plt.xlabel("Scene Number")
    plt.ylabel("Average Compound Score")
    plt.ylim(-1, 1)
    plt.grid(True)
    plt.legend(title="Speaker")
    plt.tight_layout()
    plt.show()

elif choice == "6":
    def print_top_bottom_sentiments(file_path, analyzer, output):
        """
        Collects dialogue for King Lear, GONERIL, REGAN, and CORDELIA,
        then prints out the top 3 most positive and top 3 most negative statements
        overall, including compound, positive, neutral, and negative scores.
        """
        speakers = {
            "King Lear": extract_lear_dialogue,
            "GONERIL": extract_goneril_dialogue,
            "REGAN": extract_regan_dialogue,
            "CORDELIA": extract_cordelia_dialogue
        }
        all_results = []
        for speaker, extractor in speakers.items():
            dialogues = extractor(file_path)
            for i, dialogue in enumerate(dialogues, start=1):
                sentiment = analyzer.polarity_scores(dialogue)
                all_results.append({
                    "speaker": speaker,
                    "block_index": i,
                    "text": dialogue,
                    "compound": sentiment['compound'],
                    "pos": sentiment['pos'],
                    "neu": sentiment['neu'],
                    "neg": sentiment['neg']
                })
        # Sort by compound score for negative and positive extremes
        negative_sorted = sorted(all_results, key=lambda x: x["compound"])
        positive_sorted = sorted(all_results, key=lambda x: x["compound"], reverse=True)
        
        msg = "\nTop 3 Most Negative Statements Overall:\n"
        output.write(msg)
        print(msg)
        for result in negative_sorted[:3]:
            msg = (f"Speaker: {result['speaker']}, Block #{result['block_index']}:\n"
                   f"Compound Score: {result['compound']}, Positive: {result['pos']}, Neutral: {result['neu']}, Negative: {result['neg']}\n"
                   f"{result['text']}\n" + "-"*40 + "\n")
            output.write(msg)
            print(msg)

        print("\n" * 30)
    
        msg = "\nTop 3 Most Positive Statements Overall:\n"
        output.write(msg)
        print(msg)
        for result in positive_sorted[:3]:
            msg = (f"Speaker: {result['speaker']}, Block #{result['block_index']}:\n"
                   f"Compound Score: {result['compound']}, Positive: {result['pos']}, Neutral: {result['neu']}, Negative: {result['neg']}\n"
                   f"{result['text']}\n" + "-"*40 + "\n")
            output.write(msg)
            print(msg)
    
    print_top_bottom_sentiments(file_path, analyzer, output)


elif choice == "7":
    char_choice = input("Enter: 1 for King Lear, 2 for GONERIL, 3 for REGAN, 4 for CORDELIA: ").strip()
    if char_choice == "1":
        dialogue_blocks = extract_lear_dialogue(file_path)
        speaker = "King Lear"
    elif char_choice == "2":
        dialogue_blocks = extract_goneril_dialogue(file_path)
        speaker = "GONERIL"
    elif char_choice == "3":
        dialogue_blocks = extract_regan_dialogue(file_path)
        speaker = "REGAN"
    elif char_choice == "4":
        dialogue_blocks = extract_cordelia_dialogue(file_path)
        speaker = "CORDELIA"
    else:
        msg = "Invalid character choice. Exiting.\n"
        print(msg)
        output.write(msg)
        output.close()
        exit(1)
    
    if not dialogue_blocks:
        msg = f"No {speaker}'s dialogue was found.\n"
        print(msg)
        output.write(msg)
        output.close()
        exit(1)
    
    breakdown_results = breakdown_sentiment_components(dialogue_blocks, analyzer)
    df_breakdown = pd.DataFrame(breakdown_results)
    
    header = f"\nDetailed Sentiment Breakdown for {speaker} (Values as percentages):\n{df_breakdown[['block_index','compound','pos','neu','neg']].head()}\n"
    avg_stats = (
        f"\nAverage Scores for {speaker}:\n"
        f"Compound: {df_breakdown['compound'].mean():.2f}%\n"
        f"Positive: {df_breakdown['pos'].mean():.2f}%\n"
        f"Neutral: {df_breakdown['neu'].mean():.2f}%\n"
        f"Negative: {df_breakdown['neg'].mean():.2f}%\n"
    )
    
    output.write(f"Sentiment Breakdown for {speaker}\n")
    output.write(header)
    output.write(avg_stats)
    
    print(header)
    print(avg_stats)
    
    exuberant = df_breakdown[(df_breakdown['pos'] > 80) & (df_breakdown['neg'] < 5)]
    if not exuberant.empty:
        msg = f"\nExuberant Blocks for {speaker} (High positive, low negative):\n{exuberant[['block_index','compound','pos','neg']]}\n"
        output.write(msg)
        print(msg)
    
    plt.figure(figsize=(12, 6))
    plt.plot(df_breakdown['block_index'], df_breakdown['compound'], marker="o", label="Compound")
    plt.plot(df_breakdown['block_index'], df_breakdown['pos'], marker="o", label="Positive")
    plt.plot(df_breakdown['block_index'], df_breakdown['neu'], marker="o", label="Neutral")
    plt.plot(df_breakdown['block_index'], df_breakdown['neg'], marker="o", label="Negative")
    plt.title(f"Sentiment Component Scores for {speaker} (Percentages)")
    plt.xlabel("Block Index")
    plt.ylabel("Score (%)")
    plt.ylim(0, 110)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

elif choice == "8":
    option_8_visualizations(file_path, analyzer, output)

else:
    msg = "Invalid choice. Please run the program again and enter a valid option (1-8 or q to quit).\n"
    print(msg)
    output.write(msg)

output.close()

# King Lear Dialogue Sentiment Analysis

This repository contains Python scripts for performing sentiment analysis on the dialogue extracted from Shakespeare's *King Lear*. Using regular expressions and the VADER sentiment analysis tool, the scripts extract dialogue blocks for various characters (such as King Lear, GONERIL, REGAN, CORDELIA, Fool, and Edgar) and then compute sentiment scores. The analysis includes interactive visualizations, scene-based normalization, and detailed breakdowns of sentiment components.

## Features

- **Dialogue Extraction:**  
  Extracts dialogue blocks for individual characters from the play text using regular expressions.

- **Sentiment Analysis:**  
  Utilizes VADER to compute compound, positive, neutral, and negative sentiment scores for each dialogue block.

- **Scene-level Analysis:**  
  Supports normalization of sentiment scores by scene, allowing comparisons across scenes.

- **Visualization:**  
  Generates interactive plots (line plots, scatter plots, heatmaps, and sliding window analyses) with Matplotlib and Seaborn to visualize sentiment progression and block-level sentiment details.

- **User Interaction:**  
  Offers a menu-driven command-line interface with multiple options, including:
  - Individual character analysis
  - Combined scene-level sentiment analysis
  - Detailed sentiment breakdowns (with percentages)
  - Extraction of top positive and negative statements

## Repository Structure

- **Dialogue Extraction Functions:**  
  Contains functions to extract dialogue from the *King Lear* text file for characters such as King Lear, GONERIL, REGAN, CORDELIA, Fool, and Edgar. There are also scene-level extraction functions for combined analyses.

- **Sentiment Analysis Functions:**  
  Includes functions that apply the VADER sentiment analyzer to dialogue blocks and convert scores into percentages when needed.

- **Visualization Functions:**  
  Provides options for plotting sentiment progression, interactive clickable plots, heatmaps, and sliding window analyses for Scene 1 and across scenes.

- **Main Programs:**  
  - **paper1sentiment.py:** General sentiment analysis for the entire play.
  - **SentimentTest.py:** Focused sentiment analysis for the first scene only.

- **Input File:**  
  The scripts expect the play text file (`KingLear.txt`) to be present in the repository (or you can adjust the file path accordingly).

## Findings & Research Insights

- The "mad" characters (Lear, the Fool, and Edgar) show oscillating sentiment with dramatic fluctuations in positive and negative scores.
- In the general analysis (`paper1sentiment.py`), the highest compound sentiment scores were found in love confessions, with Cordelia’s confession ranking at the top.

## Requirements

- Python 3.12
- [nltk](https://www.nltk.org/)
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)

You can install the required packages using pip:

```bash
pip install nltk vaderSentiment pandas matplotlib seaborn
```
## How to Run

### Prepare the Text File:
Ensure that `KingLear.txt` (the full text of *King Lear*) is in the same directory as the Python scripts. (If needed, adjust the file path within the scripts.)

### Execute the Script:
Run the main Python file in your terminal:

```bash
python your_script_name.py

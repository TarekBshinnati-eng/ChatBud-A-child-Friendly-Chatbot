# ChatBud: A Child-Friendly Chatbot

A child-friendly Large Language Model (LLM) chatbot designed to provide safe, educational, and age-appropriate conversations for children aged 9-11 years old.

**Course:** EECE 490
**Team:** Tarek, Nour, Moataz, Ahmad

## Project Overview

ChatBud is an AI-powered chatbot specifically designed for children, focusing on:
- Safe and appropriate responses
- Educational content delivery
- Child-friendly language and tone
- Prosocial behavior modeling

The project uses **Gemma3** as the base model with fine-tuning for child-appropriate responses.

## Repository Structure

```
ChatBud-A-child-Friendly-Chatbot/
├── docs/                              # Project documentation
│   ├── project_report.pdf             # Final project report
│   └── project_poster.pdf             # Project presentation poster
│
├── src/                               # Source code
│   ├── chatbot/                       # ChatBud chatbot application
│   │   ├── ChatBud.ipynb              # Main chatbot notebook
│   │   └── chatBudUI.html             # Web UI interface
│   │
│   ├── cosine_similarity/             # Semantic similarity analysis
│   │   ├── cosineSim_gemma3.ipynb     # Cosine similarity with Gemma3
│   │   ├── cosineSimilarity_gemma3.ipynb
│   │   ├── child_qa_export.ipynb      # Q&A data export
│   │   ├── prompts.csv                # Test prompts
│   │   └── cosine_similarity_visualization.png
│   │
│   ├── fine_tuning/                   # Model fine-tuning
│   │   └── fine_tuning_gemma3.ipynb   # Gemma3 fine-tuning notebook
│   │
│   └── testing/                       # Model testing & inference
│       └── testing_gemma3.ipynb       # Testing fine-tuned vs base model
│
├── evaluation/                        # Model evaluation & comparison
│   ├── judge_analysis/                # LLM-as-judge evaluation
│   │   ├── gemini_judge_analysis.ipynb    # Analysis notebook
│   │   ├── gemini_judge_prompt.txt        # Judge prompt template
│   │   ├── gemini_judge_results.csv       # Evaluation results
│   │   ├── gemini_judge_raw_output.txt    # Raw judge outputs
│   │   └── prompts_gemini3pro_judge.csv   # Judge prompts
│   │
│   ├── results/                       # Evaluation metrics (CSV)
│   │   ├── overall_model_summary.csv
│   │   ├── category_overall_means.csv
│   │   ├── score_comparison_means.csv
│   │   ├── statistical_tests.csv
│   │   └── ... (16 CSV files total)
│   │
│   └── visualizations/                # Evaluation charts
│       ├── 01_overall_winner_pie.png
│       ├── 02_overall_winner_bar.png
│       ├── 03_mean_scores_by_dimension.png
│       └── ... (15 PNG files total)
│
├── eda/                               # Exploratory Data Analysis
│   ├── data/                          # Analysis datasets
│   │   ├── combined_eda.csv           # Combined dataset analysis
│   │   ├── cai_eda.csv                # CAI dataset
│   │   ├── child_qa_eda.csv           # Child Q&A dataset
│   │   ├── kidschatbot_eda.csv        # Kids chatbot dataset
│   │   ├── prosocial_eda.csv          # Prosocial dataset
│   │   ├── sahar_eda.csv              # Sahar dataset
│   │   └── eda_summary_by_source.csv  # Summary statistics
│   │
│   └── visualizations/                # EDA charts (24 PNG files)
│       ├── source_pie_chart.png       # Dataset distribution
│       ├── *_length_hist.png          # Message length histograms
│       └── *_length_ratio_hist.png    # Length ratio distributions
│
├── requirements.txt                   # Python dependencies
└── README.md
```

## Key Components

### 1. Fine-Tuning (`src/fine_tuning/`)
Fine-tuning Gemma3 model using LoRA adapters for child-friendly responses.

### 2. ChatBud Application (`src/chatbot/`)
- **ChatBud.ipynb**: Main chatbot logic and inference
- **chatBudUI.html**: Interactive web interface

### 3. Evaluation (`evaluation/`)
Comprehensive evaluation comparing fine-tuned vs base model:
- **LLM-as-Judge**: Using Gemini to evaluate response quality
- **Metrics**: Fluency, Accuracy, Child-Language, Age-Appropriateness, Safety, Helpfulness, Engagement, Brevity

### 4. EDA (`eda/`)
Analysis of training datasets from multiple sources:
- CAI, Child Q&A, KidsChatbot, Prosocial, Sahar

## Datasets

| Dataset | Description |
|---------|-------------|
| CAI | Conversational AI dataset |
| Child Q&A | Child question-answering pairs |
| KidsChatbot | Kids chatbot conversation logs |
| Prosocial | Prosocial behavior dialogue dataset |
| Sahar | Sahar conversational dataset |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. **Fine-tune the model**: Run `src/fine_tuning/fine_tuning_gemma3.ipynb`
2. **Test the model**: Run `src/testing/testing_gemma3.ipynb`
3. **Launch ChatBud**: Open `src/chatbot/chatBudUI.html` in browser

## Evaluation Results

The evaluation compares the fine-tuned model against the base Gemma3 model across multiple dimensions. See `evaluation/visualizations/` for detailed charts.

## License

This project was created for educational purposes as part of EECE 490 course.

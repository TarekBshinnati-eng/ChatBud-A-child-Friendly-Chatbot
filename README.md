# ChatBud: A Child-Friendly Chatbot

A child-friendly Large Language Model (LLM) chatbot designed to provide safe, educational, and age-appropriate conversations for children aged 9-11 years old.

**Course:** EECE 490
**Team:** Tarek Bshinnati, Nour Hamieh, Moataz Maarouf, Ahmad Isber

## Project Overview

ChatBud is an AI-powered chatbot specifically designed for children, focusing on:
- Safe and appropriate responses
- Educational content delivery
- Child-friendly language and tone
- Prosocial behavior modeling

The project uses **Gemma3** as the base model with LoRA fine-tuning for child-appropriate responses.

## Repository Structure

```
ChatBud-A-child-Friendly-Chatbot/
├── data/                                  # Training datasets
│   └── combined_dataset.csv               # Combined training dataset
│
├── docs/                                  # Project documentation
│   ├── project_report.pdf                 # Final project report
│   └── project_poster.pdf                 # Project presentation poster
│
├── src/                                   # Source code
│   ├── chatbot/                           # ChatBud chatbot application
│   │   ├── ChatBud.ipynb                  # Main chatbot notebook
│   │   ├── chatBudUI.html                 # Web UI interface
│   │   └── chatbud_system_prompt.txt      # System prompt for ChatBud
│   │
│   ├── cosine_similarity/                 # Semantic similarity analysis
│   │   ├── cosineSimilarity_gemma3.ipynb  # Cosine similarity evaluation
│   │   ├── child_qa_export.ipynb          # Q&A data export utility
│   │   ├── prompts.csv                    # Test prompts for similarity
│   │   └── cosine_similarity_visualization.png
│   │
│   ├── fine_tuning/                       # Model fine-tuning
│   │   └── fine_tuning_gemma3.ipynb       # Gemma3 LoRA fine-tuning
│   │
│   ├── testing/                           # Model testing & inference
│   │   └── testing_gemma3.ipynb           # Testing fine-tuned vs base
│   │
│   └── utilities/                         # Utility scripts
│       └── gemma3_drive_download.ipynb    # Download model to Drive
│
├── evaluation/                            # Model evaluation & comparison
│   ├── judge_analysis/                    # LLM-as-judge evaluation
│   │   ├── gemini_judge_analysis.ipynb    # Judge analysis notebook
│   │   ├── gemini_judge_prompt.txt        # Judge prompt template
│   │   ├── gemini_judge_results.csv       # Evaluation results
│   │   ├── gemini_judge_raw_output.txt    # Raw judge outputs
│   │   └── prompts_gemini3pro_judge.csv   # Judge prompts
│   │
│   ├── length_readability/                # Length & readability metrics
│   │   ├── length_readability_eval_gemma3.ipynb  # Evaluation notebook
│   │   ├── length_readability_results.csv        # Results data
│   │   ├── evaluation_prompts.csv                # Test prompts
│   │   ├── evaluation_dataset_claude.csv         # Claude-generated dataset
│   │   ├── evaluation_dataset_from_hf.csv        # HuggingFace dataset
│   │   ├── generate_evaluation_dataset.ipynb     # Dataset generation
│   │   ├── evaluation_dataset_generation_prompt.txt  # Generation prompt
│   │   └── visualizations/                # Length/readability charts
│   │       ├── average_words.png          # Average word count comparison
│   │       ├── average_fkg.png            # Flesch-Kincaid grade comparison
│   │       ├── type_token_ratio.png       # TTR comparison
│   │       └── scatter_base_vs_finetuned.png  # Base vs FT scatter
│   │
│   ├── toxicity/                          # Toxicity evaluation
│   │   ├── toxicity_eval_gemma3_strict.ipynb     # Toxicity evaluation
│   │   ├── deepeval_toxicity_gemini_judge.ipynb  # DeepEval with Gemini
│   │   ├── harmeval_gemma3_toxicity_strict_logs.csv  # Toxicity logs
│   │   ├── harmeval/                      # HarmEval benchmark
│   │   │   ├── harmeval_prompt_sampler.ipynb     # Prompt sampling
│   │   │   ├── getting_the_answers_harmeval_gemma3.ipynb  # Generate answers
│   │   │   ├── harmeval_prompts_labeled.csv      # Labeled prompts
│   │   │   ├── harmeval_gemma3_model_answers.csv # Model responses
│   │   │   └── harmeval_topics.png               # Topic distribution
│   │   └── visualizations/                # Toxicity charts
│   │       └── mean_toxicity_bar_chart.png
│   │
│   ├── results/                           # Evaluation metrics (CSV)
│   │   ├── overall_model_summary.csv
│   │   ├── category_overall_means.csv
│   │   ├── score_comparison_means.csv
│   │   ├── statistical_tests.csv
│   │   └── ... (16 CSV files total)
│   │
│   └── visualizations/                    # General evaluation charts
│       ├── 01_overall_winner_pie.png
│       ├── 02_overall_winner_bar.png
│       ├── 03_mean_scores_by_dimension.png
│       ├── 04_radar_score_comparison.png
│       └── ... (15 PNG files total)
│
├── eda/                                   # Exploratory Data Analysis
│   ├── data/                              # Analysis datasets
│   │   ├── combined_eda.csv               # Combined dataset analysis
│   │   ├── cai_eda.csv                    # CAI dataset
│   │   ├── child_qa_eda.csv               # Child Q&A dataset
│   │   ├── kidschatbot_eda.csv            # Kids chatbot dataset
│   │   ├── prosocial_eda.csv              # Prosocial dataset
│   │   ├── sahar_eda.csv                  # Sahar dataset
│   │   └── eda_summary_by_source.csv      # Summary statistics
│   │
│   └── visualizations/                    # EDA charts
│       ├── source_pie_chart.png           # Dataset source distribution
│       ├── dataset_distribution_pie.png   # Overall distribution
│       ├── *_length_hist.png              # Message length histograms
│       ├── *_length_ratio_hist.png        # Length ratio distributions
│       └── ... (26 PNG files total)
│
├── requirements.txt                       # Python dependencies
└── README.md
```

## Key Components

### 1. Fine-Tuning (`src/fine_tuning/`)
Fine-tuning Gemma3 model using LoRA adapters for child-friendly responses. The fine-tuned model learns to:
- Use simpler vocabulary
- Give shorter, more concise responses
- Avoid inappropriate content
- Redirect safety-critical questions to trusted adults

### 2. ChatBud Application (`src/chatbot/`)
- **ChatBud.ipynb**: Main chatbot logic and inference
- **chatBudUI.html**: Interactive web interface
- **chatbud_system_prompt.txt**: System prompt defining ChatBud's behavior

### 3. Evaluation (`evaluation/`)
Comprehensive evaluation comparing fine-tuned vs base model across multiple dimensions:

- **LLM-as-Judge** (`judge_analysis/`): Using Gemini to evaluate response quality
- **Length & Readability** (`length_readability/`): Word count, Flesch-Kincaid grade, TTR
- **Toxicity** (`toxicity/`): Safety evaluation using HarmEval benchmark

**Metrics Evaluated:**
- Fluency
- Accuracy
- Child-Language
- Age-Appropriateness
- Safety
- Helpfulness
- Engagement
- Brevity

### 4. Semantic Similarity (`src/cosine_similarity/`)
Measures how closely model responses match reference answers using sentence embeddings.

### 5. EDA (`eda/`)
Exploratory data analysis of training datasets from multiple sources:
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
4. **Run evaluation**: Use notebooks in `evaluation/` folder

## Evaluation Results

The evaluation compares the fine-tuned model against the base Gemma3 model across multiple dimensions. See `evaluation/visualizations/` for detailed charts.

Key findings:
- Fine-tuned model produces shorter, more child-appropriate responses
- Improved safety handling for sensitive topics
- Better readability scores (lower Flesch-Kincaid grade)



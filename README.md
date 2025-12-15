# ChatBud: A Child-Friendly Chatbot

A child-friendly Large Language Model (LLM) chatbot designed to provide safe, educational, and age-appropriate conversations for children.

**Course:** EECE 490
**Team:** Tarek, Nour, Moataz, Ahmad

## Project Overview

ChatBud is an AI-powered chatbot specifically designed for children, focusing on:
- Safe and appropriate responses
- Educational content delivery
- Child-friendly language and tone
- Prosocial behavior modeling

## Repository Structure

```
ChatBud-A-child-Friendly-Chatbot/
├── docs/                          # Project documentation
│   ├── project_report.pdf         # Final project report
│   └── project_poster.pdf         # Project presentation poster
├── eda/                           # Exploratory Data Analysis
│   ├── data/                      # Analysis datasets
│   │   ├── cai_eda.csv            # CAI dataset analysis
│   │   ├── child_qa_eda.csv       # Child Q&A dataset analysis
│   │   ├── combined_eda.csv       # Combined datasets analysis
│   │   ├── eda_summary_by_source.csv  # Summary statistics by source
│   │   ├── kidschatbot_eda.csv    # Kids chatbot dataset analysis
│   │   ├── prosocial_eda.csv      # Prosocial dataset analysis
│   │   └── sahar_eda.csv          # Sahar dataset analysis
│   └── visualizations/            # Analysis charts and graphs
│       ├── *_length_hist.png      # Message length histograms
│       ├── *_length_ratio_hist.png # Length ratio distributions
│       ├── source_pie_chart.png   # Dataset source distribution
│       └── ...                    # Other visualizations
├── src/                           # Source code
│   ├── chatbot/                   # ChatBud UI and chatbot logic
│   └── cosine_similarity/         # Cosine similarity analysis
└── README.md
```

## Datasets Analyzed

The project analyzes multiple child-friendly conversation datasets:

| Dataset | Description |
|---------|-------------|
| CAI | Conversational AI dataset |
| Child Q&A | Child question-answering pairs |
| KidsChatbot | Kids chatbot conversation logs |
| Prosocial | Prosocial behavior dialogue dataset |
| Sahar | Sahar conversational dataset |

## Key Features

- **Exploratory Data Analysis (EDA):** Comprehensive analysis of multiple child-friendly conversation datasets
- **Cosine Similarity Analysis:** Semantic similarity evaluation using Gemma3 model
- **ChatBud UI:** User-friendly HTML interface for the chatbot

## Visualizations

The EDA includes various visualizations:
- Message length distributions (user and assistant)
- Length ratio histograms
- Assistant response quality metrics (FKG readability, TTR)
- Safety label distributions
- Source distribution pie charts

## Getting Started

*Documentation for running the chatbot will be added once source files are uploaded.*

## License

This project was created for educational purposes as part of EECE 490 course.

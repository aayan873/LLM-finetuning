# LLM Classification Finetuning on Kaggle

## This project is part of the LLM Classification Finetuning competition on Kaggle.
---

**The goal of the competition is to predict which chatbot response (A or B) a human judge prefers, or if it’s a tie.**

I built and fine-tuned a BERT-based classification model using PyTorch Lightning for this task.
The model takes a prompt along with two responses, extracts embeddings from BERT, and passes the combined representations through a custom feed-forward head to predict one of three outcomes — A wins, B wins, or Tie.

## Approach

- Fine-tuned a pretrained BERT model with most layers frozen for efficiency.  
- Added a lightweight classification head to predict human preference.  
- Generated a `submission.csv` for Kaggle leaderboard submission.

## Model Architecture

- **Embeddings**: `[CLS]` token used for each input (prompt, response A, response B).  
- **Concatenation**: Combines embeddings to preserve comparison context.  
- **Classification Head**: Feed-forward network with two hidden layers + dropout outputs probabilities for A/B/Tie.  
- **Training**: Cross-entropy loss, AdamW optimizer, StepLR scheduler.

Competition link

[LLM Classification Finetuning (Kaggle)](https://www.kaggle.com/competitions/llm-classification-finetuning)

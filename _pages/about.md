---
permalink: /
title: "Wentao`s Page"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

I'm a final year master student from [AMA](https://www.polyu.edu.hk/ama/), [The Hong Kong Polytechnic University](https://www.polyu.edu.hk/). My research interest includes Large language models, machine learning, and Multimodal Learning .

Personal info
======
[Email](2598536686@qq.com) / [Github](https://github.com/NONGFUYULANG) 

Papers
======
1. Interpretability Analysis of Pre-trained Convolutional Neural Networks for Medical Diagnosis(https://ieeexplore.ieee.org/document/10104339)
2. GenesisScene: Procedural Content Generation Guided by Large Vision LanguageModels

Experience
------
I. Microsoft Research Asia – Large Model Application Research: Model Training and Optimization Based on ARC-AGI Business
September 2024 – Present
------
Responsible for pre-training the code large model for ARC-AGI, dynamically adjusting data ratios based on statistical and programmatic sampling, conducting data cleaning, and comparing the performance of multiple versions of the trained model (5 million dataset, 1B model pre-training).
Participated in building a code generation model based on encoder and decoder architectures, encoding and comparing training effectiveness using 2D-ROPE and relative position encoding.
Evaluated pre-trained models and LLaMA architectures using perplexity (PPL) and other metrics, analyzing the impact of different initialization strategies and tokenization methods on perplexity. The pre-trained code model demonstrated an 8% accuracy improvement over LLaMA 3 8B.
Designed ARC-AGI experiments based on the multimodal large model Flamingo, exploring the performance of multimodal models utilizing a grid encoder in ARC-AGI tasks.

II. Tencent – IEG-NLP Application Research: Model Training and Optimization Based on UGC + AI Business – AAAI Paper Submission
March 2024 – September 2024
------
Information Extraction Model Development and Training: Built an information extraction model based on GPT-2-MOE, enabling precise extraction of quantitative numerical information. Evaluated the effects of different attention mechanisms and activation functions, refined the output layer for parameter type recognition, and incorporated adversarial learning to enhance model performance.
LLM Fine-Tuning: Applied multi-stage training techniques using LoRA, DoRA, and DPO to fine-tune the Qwen-VL model, significantly improving generation quality compared to models such as Gemma 2B, Phi, LLaMA 3, and LLaVA. Integrated LMDeploy for model deployment and production.
Multi-Level Agent Generation and Multimodal Classification Model Development: Designed a multi-level parameter generation framework based on LLM, significantly improving accuracy in information extraction tasks for 7B models.
Image-Text Classification Enhancement with LLaVA 1.5 and Qwen-VL: Developed an LVLM-based multi-classification model, achieving a 3% improvement in precision for traditional and multi-task classification tasks. Contributed to the development of an RAG framework utilizing FAISS and CLIP.

III. ByteDance – TikTok Commercialization – Algorithm Intern: Model Training and Deployment for Video Brand Recognition
March 2023 – July 2023
------
Computer Vision Model Training and Deployment: Trained and deployed YOLOv7 and DINO object detection models for TikTok brand recognition in videos. Conducted distributed training across multiple GPUs and launched multiple model versions. Integrated ByteDance’s internal framework to build Docker images and deployed inference models using PySpark, Hive, and HDFS, processing millions of data points daily with high-concurrency inference.
LLM Fine-Tuning: Deployed ChatGLM-6B locally for named entity recognition and sentiment classification. Utilized PEFT techniques (LoRA, P-Tuning, P-Tuning v2) for model fine-tuning and built a vector database for domain-specific fine-tuning using LangChain, integrating vLLM for inference acceleration.
CLIP Model Optimization: Fine-tuned OpenAI’s CLIP model, implementing a PyTorch-based CLIP-Adapter model. Local data fine-tuning improved precision by 5% and recall by 3%. Explored zero-shot capabilities using the Tip-Adapter model.

IV. SenseTime – Research Institute – Large Model Backend Intern: RAG Project Development Based on SenChat
January 2024 – March 2024
------
RAG (Retrieval-Augmented Generation): Fine-tuned the embedding model and developed the backend for the model knowledge base. Implemented knowledge ingestion, retrieval, and indexing functionalities. Explored different IK segmentation strategies in Elasticsearch to enhance recall performance. Developed a hybrid retrieval framework combining vector search and term frequency-based recall.
AI-Agent Development: Deployed LAgent locally, developing tool functionalities based on the React Agent framework. Gained expertise in AI agent execution chains and principles.
Model Deployment: Integrated TGI framework, PostgreSQL, and Milvus to deploy multiple inference models across GPUs.
High-Concurrency LLM Request Framework Development Based on Celery: Developed and deployed a high-concurrency request framework integrating Celery, Docker, Redis, and LLM APIs. Implemented dynamic worker allocation and load balancing for optimized performance.

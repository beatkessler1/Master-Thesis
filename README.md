# GenAI Supply Chain Analytics and Knowledge Platform

## Project Overview
This project demonstrates the design and implementation of an enterprise-grade GenAI platform for supply chain analytics and knowledge retrieval in the pharmaceutical industry.

The solution combines structured SAP data with unstructured documentation using Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs) to support data-driven decision-making and improve operational transparency.

---

## Business Problem
Supply chain data and operational knowledge are often distributed across multiple systems and document repositories.  
This fragmentation makes it difficult for decision-makers to quickly access relevant information and derive insights efficiently.

Manual searches, disconnected analytics, and limited visibility reduce operational efficiency and slow down decision-making processes.

---

## Solution Overview
The platform integrates structured and unstructured data into a unified analytics and knowledge retrieval workflow.

Key capabilities include:
- Retrieval of relevant documents using vector-based search
- Analytics on structured SAP data
- Automated data ingestion and preprocessing
- LLM-based processing and summarization of supply chain documentation

---

## Repository Structure

This repository is structured into modular components reflecting the main parts of the solution:

- ingestion_pipeline/  
  Python-based data ingestion and preprocessing pipelines used to prepare structured and unstructured data.

- rag_pipeline/  
  Simplified components of Retrieval-Augmented Generation workflows, including document chunking and vector retrieval logic.

- sql_analytics/  
  Example SQL queries and analytics logic used for structured SAP data analysis.

- architecture/  
  Architecture diagrams and system design documentation.

- presentation/  
  Project presentation summarizing the business problem, solution design, and results.

---

## Architecture
See the architecture diagrams in the `/architecture` folder for a high-level overview of the system design and data flows.

---

## Tech Stack
Python  
SQL  
AWS  
Dataiku DSS  
Large Language Models (LLMs)  
Retrieval-Augmented Generation (RAG)  

---

## Results and Impact
The solution enabled:
- Improved accessibility of operational knowledge
- Faster retrieval of relevant supply chain information
- Reduction of manual search and analysis effort
- Enhanced transparency for data-driven decision-making

---

## Disclaimer
This repository contains a simplified and anonymized version of the project architecture and selected components.  
All proprietary data, internal infrastructure details, and confidential business logic have been removed or replaced with illustrative examples.

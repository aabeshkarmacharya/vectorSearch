# Vector Search Application

A Python-based vector search application using Qdrant vector database for semantic search capabilities.

## Overview

This project implements a vector search system that allows you to index and search documents using semantic similarity. It uses Qdrant as the vector database backend and provides a REST API interface for search operations.

## Specification

- **Vector Database**: Qdrant
- **Embedding Model**: bge-base-en-v1.5
- **API**: REST API using FastAPI

## Quick Start

### Using Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd vectorSearch
   ```

2. **Start the application**
   ```bash
   docker-compose up -d
   ```

3. **Access the application**
   - Application API: http://localhost:8000
   - Qdrant Dashboard: http://localhost:6333/dashboard


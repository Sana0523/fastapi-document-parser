# HackRX Document Query API

An intelligent document Q&A system designed for insurance, legal, HR, and compliance documents. This FastAPI application uses advanced NLP techniques including semantic embeddings and Google's Gemini AI to provide accurate answers to questions about uploaded documents.

## Features

- **Multi-format Document Support**: PDF, DOCX, and plain text files
- **Intelligent Document Processing**: Advanced text extraction with fallback methods
- **Semantic Search**: Vector-based similarity search using sentence transformers
- **AI-Powered Answers**: Integration with Google Gemini for comprehensive responses
- **Document Chunking**: Smart text segmentation with overlap for better context
- **Multiple Query Types**: Support for yes/no, explanatory, definition, and procedural questions
- **Comprehensive Error Handling**: Robust error management with detailed logging
- **Caching**: Document caching for improved performance
- **Authentication**: Bearer token authentication for secure access

## Architecture

```
Document URL → Download → Text Extraction → Semantic Chunking → Vector Embeddings → Search → AI Answer Generation
```

## Requirements

- Python 3.8+
- Google Gemini API Key
- SpaCy English model (`en_core_web_sm`)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd hackrx-document-query-api
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download SpaCy model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Download NLTK data** (will be done automatically on first run):
   ```bash
   python -c "import nltk; nltk.download('stopwords')"
   ```

## Configuration

1. **Create a `.env` file** in the root directory:
   ```env
   GEMINI_API_KEY=your_google_gemini_api_key_here
   ```

2. **Get a Gemini API Key**:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Add it to your `.env` file

## Usage

### Starting the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, visit:
- Interactive API docs: `http://localhost:8000/docs`
- ReDoc documentation: `http://localhost:8000/redoc`

### Authentication

All API endpoints (except `/` and `/health`) require Bearer token authentication:

```bash
Authorization: Bearer 6613ee9a3bcb0925802224950bfad9d70f8be3907dc22442d035ae7798dbe14b
```

### Main Endpoint

**POST** `/hackrx/run`

Process a document and answer questions about it.

**Request Body**:
```json
{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "Does this policy cover knee surgery?",
    "What is the waiting period for pre-existing conditions?",
    "What are the exclusions mentioned in this document?"
  ]
}
```

**Response**:
```json
{
  "answers": [
    {
      "question": "Does this policy cover knee surgery?",
      "answer": "Yes, this policy covers knee surgery under the orthopedic procedures benefit...",
      "confidence_score": 0.87,
      "reasoning": "Answer based on 3 relevant document sections with average similarity score of 0.78",
      "clause_references": [
        {
          "chunk_id": "chunk_15",
          "page_number": 12,
          "similarity_score": 0.89,
          "text_snippet": "Orthopedic procedures including knee surgery are covered under this policy..."
        }
      ],
      "decision_rationale": "Based on semantic search and yes_no analysis"
    }
  ],
  "processing_time": 15.2,
  "document_info": {
    "url": "https://example.com/document.pdf",
    "total_chunks": 45,
    "total_pages": 20,
    "text_length": 15420,
    "processing_method": "semantic_chunking_with_embeddings"
  }
}
```

### Debug Endpoint

**GET** `/debug/document?url=<document_url>`

Inspect how a document is processed without answering questions.

### Health Check

**GET** `/health`

Check the status of all loaded models and services.

## Example Usage with cURL

```bash
# Health check
curl -X GET "http://localhost:8000/health"

# Process document and answer questions
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 6613ee9a3bcb0925802224950bfad9d70f8be3907dc22442d035ae7798dbe14b" \
  -d '{
    "documents": "https://example.com/insurance-policy.pdf",
    "questions": [
      "What is the maximum coverage amount?",
      "Are pre-existing conditions covered?",
      "What is the claim process?"
    ]
  }'
```

## Example Usage with Python

```python
import requests

url = "http://localhost:8000/hackrx/run"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer 6613ee9a3bcb0925802224950bfad9d70f8be3907dc22442d035ae7798dbe14b"
}

data = {
    "documents": "https://example.com/policy.pdf",
    "questions": [
        "What are the covered benefits?",
        "What is the deductible amount?"
    ]
}

response = requests.post(url, json=data, headers=headers)
result = response.json()

for answer in result["answers"]:
    print(f"Q: {answer['question']}")
    print(f"A: {answer['answer']}")
    print(f"Confidence: {answer['confidence_score']:.2f}")
    print("---")
```

## Supported Document Types

- **PDF**: Both text-based and scanned documents
- **DOCX**: Microsoft Word documents with text and tables
- **Plain Text**: UTF-8, Latin-1, CP1252, ISO-8859-1 encodings

## Document Processing Features

- **Smart Text Extraction**: Multiple fallback methods for robust text extraction
- **Table Processing**: Extraction of tabular data from PDFs and DOCX files
- **Semantic Chunking**: Intelligent text segmentation based on document structure
- **Page Number Tracking**: Maintains page references for citations
- **Caching**: Automatic document caching for repeated requests

## Query Types Supported

- **Coverage Questions**: "Is X covered?", "Does this include Y?"
- **Waiting Periods**: "How long is the waiting period?"
- **Conditions**: "What are the requirements?"
- **Definitions**: "What does X mean?", "Define Y"
- **Benefits**: "What is the maximum amount?"
- **Exclusions**: "What is not covered?"
- **Procedures**: "How do I file a claim?"
- **Eligibility**: "Who is eligible?"

## Performance Considerations

- **Batch Processing**: Embeddings are processed in batches for memory efficiency
- **Caching**: Documents are cached locally to avoid re-download
- **Chunking Strategy**: Optimized chunk sizes with overlapping for better context
- **Vector Search**: FAISS for fast similarity search
- **Memory Management**: Streaming document download with size limits

## Error Handling

The API includes comprehensive error handling for:
- Document download failures
- Text extraction errors
- API rate limiting
- Model initialization failures
- Authentication errors
- Malformed requests

## Monitoring and Logging

- Detailed logging at INFO level
- Request/response timing
- Error tracking with stack traces
- Model loading status
- Processing statistics

## Security Features

- Bearer token authentication
- Input validation with Pydantic
- Rate limiting considerations
- Safe document processing
- Error message sanitization

## Limitations

- Document size limit: 50MB
- Supported formats: PDF, DOCX, plain text
- Requires internet connection for document download
- Gemini API rate limits apply
- Memory usage scales with document size

## Troubleshooting

### Common Issues

1. **"SpaCy model not found"**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **"GEMINI_API_KEY not found"**:
   - Ensure `.env` file exists with valid API key
   - Check environment variable loading

3. **Document download fails**:
   - Verify URL accessibility
   - Check network connectivity
   - Ensure document is publicly accessible

4. **Memory errors**:
   - Reduce document size
   - Check available system memory
   - Consider processing smaller chunks

### Debug Mode

Enable debug logging by setting environment variable:
```bash
export LOG_LEVEL=DEBUG
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request



## Support

For support and questions:
- Check the [API documentation](http://localhost:8000/docs)
- Review the logs for error details
- Ensure all dependencies are properly installed
- Verify environment configuration

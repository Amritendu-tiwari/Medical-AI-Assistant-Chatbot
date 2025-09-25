# Medical AI Assistant Chatbot - RAG 
A Retrieval-Augmented Generation (RAG) prototype for a Medical AI Assistant Chatbot built with Langchain. This chatbot provides general medical information while maintaining strict safety boundaries and ethical guidelines.

## 🏥 Features

- **Safe Medical Information**: Provides general medical information based on authoritative sources
- **Ethical Boundaries**: Refuses to provide diagnoses, prescriptions, or specific medical advice
- **Source Attribution**: Shows retrieved sources for transparency
- **Medical Disclaimer**: Includes persistent medical disclaimers in all responses
- **Out-of-Scope Handling**: Politely declines non-medical or inappropriate queries
- **Interactive UI**: User-friendly Streamlit interface

## 📁 Project Structure

```
medical-ai-assistant/
├── data/                           # Medical documents and processed data
│   ├── cdc_viral_hemorrhagic_fevers.md
│   ├── cdc_infection_control_recommendations.md
│   ├── cdc_micronutrient_facts.md
│   ├── redcross_first_aid_steps.md
│   ├── who_vitamin_mineral_requirements.pdf
│   ├── who_vitamin_mineral_requirements.txt
│   └── processed_chunks.json       # Processed document chunks
├── src/                           # Source code
│   ├── ingestion.py              # Document loading and chunking
│   ├── embed_index.py            # Embedding generation and vector store
│   ├── retriever.py              # Retrieval function
│   ├── qa.py                     # LLM integration and QA chain
│   └── streamlit_app.py          # Streamlit web interface
├── examples/                      # Sample responses and demonstrations
│   └── sample_responses.md       # Example chatbot interactions
├── faiss_index/                  # FAISS vector database
├── .env                          # Environment variables (API keys)
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key
- Required Python packages (see Installation)

### Installation

1. **Clone or download the project files**

2. **Install required packages:**
   ```bash
   pip install langchain langchain-community langchain-openai pypdf tiktoken faiss-cpu openai streamlit python-dotenv
   ```

3. **Set up environment variables:**
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

### Running the Application

1. **Process the medical documents:**
   ```bash
   cd src
   python ingestion.py
   ```

2. **Generate embeddings and create vector store:**
   ```bash
   python embed_index.py
   ```

3. **Test the retrieval system:**
   ```bash
   python retriever.py
   ```

4. **Test the QA chain:**
   ```bash
   python qa.py
   ```

5. **Launch the Streamlit interface:**
   ```bash
   streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
   ```

6. **Access the application:**
   Open your browser and navigate to `http://localhost:8501`

## 📊 Data Sources

The chatbot is trained on authoritative medical documents from:

- **CDC (Centers for Disease Control and Prevention)**
  - Viral Hemorrhagic Fevers information
  - Infection control recommendations
  - Micronutrient facts and nutrition guidelines

- **WHO (World Health Organization)**
  - Vitamin and mineral requirements in human nutrition

- **American Red Cross**
  - First aid guidelines and procedures

## 🔧 Technical Architecture

### 1. Data Collection and Preprocessing
- **Document Loading**: Uses Langchain's document loaders for PDFs and text files
- **Text Chunking**: Splits documents into 400-600 token chunks with 100-token overlap
- **Metadata Preservation**: Maintains source information and page numbers

### 2. Vector Database and Retrieval
- **Embeddings**: OpenAI text-embedding-ada-002 model
- **Vector Store**: FAISS for efficient similarity search
- **Retrieval**: Returns top 5 most relevant chunks for each query

### 3. LLM Integration and Safety
- **Model**: OpenAI GPT-3.5-turbo
- **Prompt Engineering**: Carefully crafted prompts with medical disclaimers
- **Safety Measures**: Built-in refusal mechanisms for inappropriate requests

### 4. User Interface
- **Framework**: Streamlit for interactive web interface
- **Features**: Chat history, source display, clear disclaimers
- **Responsive Design**: Works on desktop and mobile devices

## 🛡️ Safety Features

### Medical Disclaimers
Every response includes a persistent disclaimer:
> "This information is for general knowledge and informational purposes only, and does not constitute medical advice. Please consult a qualified healthcare professional for any medical concerns."

### Ethical Boundaries
The chatbot is programmed to:
- ❌ **Refuse diagnoses**: "I cannot provide medical diagnoses"
- ❌ **Refuse prescriptions**: "I cannot prescribe medications"
- ❌ **Refuse specific medical advice**: Redirects to healthcare professionals
- ❌ **Handle out-of-scope queries**: Politely declines non-medical questions

### Source Transparency
- Shows retrieved document sources for each response
- Provides content snippets for verification
- Maintains traceability to original medical literature

## 📝 Example Interactions

See `examples/sample_responses.md` for detailed examples of:
- ✅ Appropriate medical information queries
- ❌ Diagnosis requests (properly refused)
- ❌ Out-of-scope questions (properly handled)
- 📚 Source attribution examples

## 🧪 Testing and Evaluation

### Test Categories
1. **Retrieval Accuracy (30%)**: Relevant document retrieval
2. **Safe Prompt Design (25%)**: Proper refusal of inappropriate requests
3. **Code Quality (20%)**: Clean, maintainable code structure
4. **Documentation (15%)**: Comprehensive documentation and examples
5. **Bonus Features (10%)**: UI, source citations, conversation support

### Evaluation Criteria
- ✅ Accurate retrieval of relevant medical information
- ✅ Consistent refusal of diagnoses and prescriptions
- ✅ Proper handling of out-of-scope queries
- ✅ Clear medical disclaimers in all responses
- ✅ Source attribution and transparency

## 🔮 Future Enhancements

### Potential Improvements
- **Multi-turn Conversations**: Enhanced conversation memory
- **Advanced Retrieval**: Hybrid search with keyword + semantic matching
- **Specialized Medical Domains**: Focused datasets for specific medical areas
- **Multi-language Support**: Support for multiple languages
- **Voice Interface**: Speech-to-text and text-to-speech capabilities

### Deployment Options
- **Docker Containerization**: Easy deployment and scaling
- **Cloud Deployment**: AWS, GCP, or Azure hosting
- **API Integration**: RESTful API for third-party integrations
- **Mobile App**: Native mobile application development

## ⚠️ Important Disclaimers

### Medical Disclaimer
This AI assistant is designed for educational and informational purposes only. It does not provide medical advice, diagnoses, or treatment recommendations. Always consult qualified healthcare professionals for medical concerns, symptoms, or health-related decisions.

### Limitations
- **Not a Medical Professional**: This system cannot replace human medical expertise
- **General Information Only**: Provides general medical knowledge, not personalized advice
- **Source Dependent**: Quality of responses depends on the quality of source documents
- **Technology Limitations**: AI systems can make errors or provide incomplete information

### Usage Guidelines
- Use for general medical education and information only
- Always verify information with healthcare professionals
- Do not use for emergency medical situations
- Consult doctors for any health concerns or symptoms

## 📄 License

This project is created for educational and demonstration purposes. Please ensure compliance with relevant medical information regulations and guidelines when deploying in production environments.

## 🤝 Contributing

This is a prototype project. For production use, consider:
- Adding more comprehensive medical datasets
- Implementing additional safety measures
- Conducting thorough medical review and validation
- Ensuring compliance with healthcare regulations (HIPAA, etc.)

---

**Demo**
[Video](https://drive.google.com/file/d/1Y1yRebd5jm1csW98LuveFybap47H2IUM/view?usp=sharing)

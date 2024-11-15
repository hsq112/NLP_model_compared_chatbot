# NLP_model_compared_chatbot

# Sentiment Analysis and Chatbot Model with NLP Techniques

This repository contains notebooks for experimenting with various NLP models, focusing on sentiment analysis using the IMDb dataset and building a chatbot using BERT. The project compares traditional RNN-based models, combines them with modern transformer-based architectures, and presents the best-performing fusion models. Here is a brief description of each file:

## Files Overview

1. **RNN_LSTM_GRU_HGRU.ipynb**  
   This notebook compares four different models for sentiment analysis on the IMDb dataset:
   - **RNN** (Recurrent Neural Network)
   - **LSTM** (Long Short-Term Memory)
   - **GRU** (Gated Recurrent Unit)
   - **Hypertuning GRU** (An optimized version of GRU with hyperparameter tuning)

   Each model is trained and evaluated to observe the performance differences among RNN-based architectures.

2. **imdb-bert-lstm.ipynb**  
   This notebook combines the **BERT** (Bidirectional Encoder Representations from Transformers) and **LSTM** models for sentiment analysis on the IMDb dataset. BERT captures contextual word representations, while LSTM focuses on sequential dependencies. The goal is to improve sentiment classification performance by leveraging the strengths of both models.

3. **chatbot_model_test&tarin.ipynb**  
   This notebook demonstrates a **chatbot model** trained on conversational data using the **BERT** model as the backbone. The model has been fine-tuned for chatbot interactions, allowing it to generate responses based on user inputs.

4. **Web-Based Chatbot Version**  
  Due to the large size of the trained model files, the web-based chatbot version is provided as a downloadable `.zip` file. You can download the **CHATBOT_web.zip** file from [Google Drive](https://drive.google.com/drive/folders/1zhXXhSUl-BGPIM09-oEp0lAZo-QzWB39?usp=sharing). Unzipping this file will provide a complete setup for the chatbot's web version.

# Chattie - Web-based Sentiment Chatbot

Chattie is a web-based sentiment analysis chatbot built using Flask (Python) and a fine-tuned BERT model. This chatbot can understand user sentiment and respond with either positive or supportive messages accordingly. The front-end interface is created using HTML, CSS, and JavaScript, providing an easy-to-use and aesthetically pleasing chat experience.

## Features

- **Sentiment Analysis**: Analyzes user input to determine if the sentiment is positive or negative.
- **Personalized Responses**: Responds with positive messages for positive sentiments and supportive messages for negative sentiments.
- **Responsive Front-End**: Provides a simple, responsive chat interface for an intuitive user experience.
- **Customizable**: Easily expand or modify the chatbot's responses to meet your needs.

## Project Structure

```
project_folder/
├── app.py                # Flask backend for handling chat requests and sentiment analysis
├── index.html            # Front-end HTML file for the chat interface
├── config.json           # Model configuration file
├── model.safetensors     # Model weight file
├── special_tokens_map.json # Token mapping for tokenizer
├── tokenizer_config.json # Tokenizer configuration file
└── vocab.txt             # Vocabulary file for tokenizer
```

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Flask
- Flask-CORS
- PyTorch
- Transformers (Hugging Face)

### Installation
1. **Set up Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies**

   Install the required dependencies in the project directory:

   ```bash
   pip install flask flask-cors torch transformers
   ```

3. **Add Model Files**

   Place the following model files in the project root directory:

   - `model.safetensors`
   - `config.json`
   - `tokenizer_config.json`
   - `vocab.txt`
   - `special_tokens_map.json`

## Usage

1. **Run the Flask App**

   Start the Flask server:

   ```bash
   python app.py
   ```

   Once started, you should see output similar to the following in your terminal:

   ```
   * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
   ```

2. **Open the Front-End Interface**

   Open a browser and navigate to `http://127.0.0.1:5000` to start chatting with Chattie.

3. **Chat with the Bot**

   Type a message and press the **Send** button or hit Enter. Chattie will respond with either a positive message if it detects positive sentiment or a supportive message if it detects negative sentiment.

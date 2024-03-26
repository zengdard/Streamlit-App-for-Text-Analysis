# Streamlit App for Text Analysis

This Streamlit app provides various text analysis tools, including text generation, text comparison using a Markov model, and text metrics calculation.

## Features

- **Accueil**: Welcome page with information about the app.
- **Expert**: A page for comparing a reference text with a suspicious text using various text metrics and visualizations.
- **MultipleTextes**: A page for analyzing and comparing multiple texts using text metrics and visualizations.
- **Gogh**: A page for analyzing images (not implemented in the provided code).

## Dependencies

- `streamlit`
- `nltk`
- `PIL`
- `requests`
- `numpy`
- `PyPDF2`
- `matplotlib`
- `opencv-python`

## Usage

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```
1. Run the app:
```bash
streamlit run app.py
```
1. Use the sidebar to navigate between the different pages and features of the app.

## Text Metrics

The app calculates the following text metrics:

- **Lexical richness**: The ratio of unique words to the total number of words in a text.
- **Grammatical richness**: The ratio of unique grammatical categories to the total number of words in a text.
- **Verbal richness**: The ratio of unique verbs to the total number of verbs in a text.

These metrics are used to compare the reference text with the suspicious text in the **Expert** page, and to analyze multiple texts in the **MultipleTextes** page.

## Markov Model

The app uses a Markov model to compare the reference text with the suspicious text in the **Expert** page. The model calculates the probability of transition between bigrams (pairs of words) in the two texts, and compares the probabilities to determine the similarity between the texts.

## Text Generation

The app uses a text generation function to generate a text with a specified number of words and a given description. The function is called when the user enters a reference text in the **Expert** page.

## Limitations

- The text generation function may not always produce accurate results, depending on the given description and the number of words.
- The app currently supports only French texts.
- The **Gogh** page is not implemented in the provided code.

## Future Work

- Implement the **Gogh** page for image analysis.
- Improve the text generation function to produce more accurate results.
- Add support for other languages.
- Implement additional text analysis tools and visualizations.

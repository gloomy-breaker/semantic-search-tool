## What is Semantic Search

Standard search tools like Ctrl + F on Windows and Cmd + F on MacOS are lexical tools, meaning that they rely on matching the user's input character by character to words in a document to find the exact matching word, but what if you don't know what exact word you are looking for, what if you only have an idea for what you're looking for? For example in a document you might be looking for the sentence describing the weather, but the sentence might not explicitly mention the word "weather", meaning that lexical searching tools will struggle to help you locate what you are trying to find while a semantic search tool matches the inputted word to words or sentences from the document through conceptual understanding.

**TLDR:** Semantic search tools focus on matching by conceptual understanding, meaning and the context rather than literal spelling.

## How the code works

First the script opens and reads the text in the sample.txt file and stores it in a variable which is then, split into individual sentences using the re.split function, and then the sentences in the list have the surrounding white space and blank lines removed. Then the model is loaded using the SentenceTransformer() function. A while loop(infinite loop) keeps prompting the user for the words they want to look for, while a nested if statement checks for whether the user wants to break out of the programming by checking for "/exit", and "/quit". After that the user's input and the list of sentences are placed into a matcher function, which essentially converts the user input and list of sentences into numerical values and compares each sentence from the list to the numerical value of the user's input and assigns the sentences with a similarity score and then compares the similarity scores for each sentence through a for loop and a nested if statement compares the similarity scores of all the sentences and stores the highest one as the best score and it also stores which sentence has the best score and returns the best sentence and best score into the variables. The best score is then compared to a threshold value and it is greater than the threshold value the sentence is printed as the best match, if the score is lower, then the sentence is printed as a best guess.

## How to set it up

1. Clone the repository and navigate into it:

    ```bash
    git clone https://github.com/gloomy-breaker/semantic-search-tool.git
    cd semantic-search-tool
    ```

2. Create a virtual environment where all the packages can be installed:

    ```bash
    python3 -m venv .venv
    ```

3. Activate the virtual environment:

    ```bash
    source .venv/bin/activate
    ```

4. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

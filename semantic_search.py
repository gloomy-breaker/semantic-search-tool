import os

# 1. Points python to the mirror of huggingface so that it works in China, comment out the line by adding a "# " (dont forget to add the space behind) if this is not needed
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Force sentence-transformers to use local files and never check the internet, switch the 1 in the HF_HUB_OFFLINE line to 0 to temporarily allow it to check the internet and update the model
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import re  # Used to split the text according to punctuation into sentences
from sentence_transformers import (
    SentenceTransformer,
    util,
)  # Used to match the cleaned up sentences to numerical values and then compared to derive a similarity score.


def main():
    print("Welcome to Semantic Search")
    print("Please paste all the text into the sample.txt file in the current directory")
    print()
    # Reads all text from a file
    with open("sample.txt", "r") as file:
        source_text = file.read()
    # Splits the text that is read according to punctuation
    sentences = re.split(r"[.!?]", source_text)
    clean_sentences = []
    for s in sentences:
        if s.strip():
            clean_sentences.append(s.strip())

    # Load AI model
    print("AI model is loading...")
    model = SentenceTransformer("all-mpnet-base-v2")

    while True:
        search_word = input("Key word or phrase: ")
        if search_word == "/quit" or search_word == "/exit":
            break
        best_match, similarity_score = matcher(search_word, clean_sentences, model)
        print(f"[{similarity_score:.4f}] -> {best_match}")

        if similarity_score >= 0.22:
            print(f" Match Found: {best_match}")
        else:
            # If the score falls below the threshold, a "best guess" is shown
            print("⚠️ No definitive match found. Best mathematical guess:")
            print(f"-> {best_match} (Low confidence score: {similarity_score:.4f})")


def matcher(word, sentence_list, model):

    vector_word = model.encode(word)
    vector_sentence_list = model.encode(sentence_list)
    # calculates the similarity between the vectorized number and list of sentences
    similarity_matrix = util.cos_sim(vector_word, vector_sentence_list)
    sentence_scores = similarity_matrix[0]
    best_sentence = ""
    best_score = 0.0

    for index, score in enumerate(sentence_scores):
        score_float = float(score)
        if word.lower() in sentence_list[index].lower():
            # Artificially add a bonus to the score because it's an exact word match to ensure that the exact word shows up if its the input, no matter the conetxt
            score_float += 0.25
        if score_float > best_score:
            best_score = score_float
            best_sentence = sentence_list[index]
    return best_sentence, best_score


if __name__ == "__main__":
    main()


# To do
# 1. Allow user to use the tool from anywhere on the computer and choose a target file(still a .txt) :
#   - Try to use sys.argv which would allow any custom file to be passed as an arguement in the terminal.
# 2. Add supoort for markdown files
# 3. Let the relevant word in the chosen sentence be highlighted, so that ther user knows which word from the sentence is the semmatically linked to the user's input word.

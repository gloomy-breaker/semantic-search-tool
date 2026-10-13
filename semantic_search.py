import os

# 1. Point Python to the high-speed mirror site for the initial download
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Force sentence-transformers to strictly use local files and never check the internet
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import re
from sentence_transformers import SentenceTransformer, util


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
            print(f"🎯 Match Found: {best_match}")
        else:
            # If it fails the threshold, still show the closest guess but label it clearly!
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
            # Artificially add a bonus to the score because it's an exact word match!
            score_float += 0.25
        if score_float > best_score:
            best_score = score_float
            best_sentence = sentence_list[index]
    return best_sentence, best_score


if __name__ == "__main__":
    main()


# 1. Define a main function:
#   A. Print a welcome message to the user.
#   B. Inform the user that they should paste their text into the sample.txt file in the cuurent directory, this will be advanced upon later
#   C. Break that large text down into individual sentences (a list of sentences) so we can check them one by one.
#   D. Start a loop that runs forever (until stopped):
#       i.   Prompt the user for the "search word" or phrase they want to find.
#       ii.  Check if the user typed an exit command (like 'quit' or 'exit'). If they did, break the loop.
#       iii. Pass the search word and the list of sentences into a helper function to calculate the match.
#       iv.  Receive the best-matching sentence and its similarity score back from that function.
#       v.  If the score is high enough, print the matching sentence.
#           Otherwise, tell the user no close match was found.

# 2. Define a helper function called "find_best_match" that takes (search_word, list_of_sentences):
#   A. Import a library tool that converts text into math/vectors (like scikit-learn or sentence-transformers).
#   B. Turn the user's search word into a vector (a list of numbers representing its meaning).
#   C. Turn every sentence in the list into vectors.
#   D. Use a built-in mathematical formula (like Cosine Similarity) to calculate the "distance" or score between the search word vector and each sentence vector.
#   E. Find which sentence had the highest similarity score.
#   F. Return that best sentence and its score back to the main function.

# 3. Call the main function to start the program.

#%% Imports Libraries
import os
import unicodedata
import regex as re
import pandas as pd
from pyvi import ViTokenizer, ViPosTagger
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle
import numpy as np

#%% Vietnamese Constants Setup
vowel_with_accent = [['a', 'à', 'á', 'ả', 'ã', 'ạ', 'a'],
                     ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 'aw'],
                     ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'aa'],
                     ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'e'],
                     ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ee'],
                     ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị', 'i'],
                     ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'o'],
                     ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'oo'],
                     ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ow'],
                     ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ', 'u'],
                     ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'uw'],
                     ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ', 'y']]

tone_marks = ['', 'f', 's', 'r', 'x', 'j']
vowel_to_ids = {}

for i in range(len(vowel_with_accent)):
    for j in range(len(vowel_with_accent[i]) - 1):
        vowel_to_ids[vowel_with_accent[i][j]] = (i, j)

#%% Unicode & Tone Normalization
def normalize_unicode(text):
    return unicodedata.normalize('NFC', text)

def normalize_vietnamese_tone(word):
    if not is_valid_vietnamese_word(word):
        return word

    chars = list(word)
    tone = 0
    vowel_indices = []
    qu_or_gi = False

    for index, char in enumerate(chars):
        x, y = vowel_to_ids.get(char, (-1, -1))
        if x == -1:
            continue
        elif x == 9:  # check "qu"
            if index != 0 and chars[index - 1] == 'q':
                chars[index] = 'u'
                qu_or_gi = True
        elif x == 5:  # check "gi"
            if index != 0 and chars[index - 1] == 'g':
                chars[index] = 'i'
                qu_or_gi = True
        if y != 0:
            tone = y
            chars[index] = vowel_with_accent[x][0]
        if not qu_or_gi or index != 1:
            vowel_indices.append(index)

    if len(vowel_indices) < 2:
        if qu_or_gi:
            if len(chars) == 2:
                x, y = vowel_to_ids.get(chars[1])
                chars[1] = vowel_with_accent[x][tone]
            else:
                x, y = vowel_to_ids.get(chars[2], (-1, -1))
                if x != -1:
                    chars[2] = vowel_with_accent[x][tone]
                else:
                    chars[1] = vowel_with_accent[5][tone] if chars[1] == 'i' else vowel_with_accent[9][tone]
            return ''.join(chars)
        return word

    for index in vowel_indices:
        x, y = vowel_to_ids[chars[index]]
        if x == 4 or x == 8:  # ê, ơ
            chars[index] = vowel_with_accent[x][tone]
            return ''.join(chars)

    if len(vowel_indices) == 2:
        if vowel_indices[-1] == len(chars) - 1:
            x, y = vowel_to_ids[chars[vowel_indices[0]]]
            chars[vowel_indices[0]] = vowel_with_accent[x][tone]
        else:
            x, y = vowel_to_ids[chars[vowel_indices[1]]]
            chars[vowel_indices[1]] = vowel_with_accent[x][tone]
    else:
        x, y = vowel_to_ids[chars[vowel_indices[1]]]
        chars[vowel_indices[1]] = vowel_with_accent[x][tone]

    return ''.join(chars)

def is_valid_vietnamese_word(word):
    chars = list(word)
    vowel_index = -1
    for index, char in enumerate(chars):
        x, y = vowel_to_ids.get(char, (-1, -1))
        if x != -1:
            if vowel_index == -1:
                vowel_index = index
            else:
                if index - vowel_index != 1:
                    return False
                vowel_index = index
    return True

def normalize_vietnamese_sentence(sentence):
    sentence = sentence.lower()
    words = sentence.split()
    for index, word in enumerate(words):
        cw = re.sub(r'(^\p{P}*)([\p{L}.]*\p{L}+)(\p{P}*$)', r'\1/\2/\3', word).split('/')
        if len(cw) == 3:
            cw[1] = normalize_vietnamese_tone(cw[1])
        words[index] = ''.join(cw)
    return ' '.join(words)

#%% Text Cleaning & Processing
def segment_vietnamese_words(text):
    return ViTokenizer.tokenize(text)

def convert_to_lowercase(text):
    return text.lower()

def clean_text(text):
    text = re.sub(r'\b\d+\b', ' ', text)
    text = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_vietnamese_text(text):
    text = normalize_unicode(text)
    text = normalize_vietnamese_sentence(text)
    text = segment_vietnamese_words(text)
    text = convert_to_lowercase(text)
    text = clean_text(text)
    return text

#%% File I/O
list_name_categories = os.listdir("Test_Full")

def get_list_content(path, stopword_path="stopword.txt"):
	list_content_file = []
	list_name_file = os.listdir(path)

	with open(stopword_path, "r", encoding="utf-8") as f:
		stopwords = set(f.read().splitlines())

	for name_file in list_name_file:
		path_file = path + "/" + name_file
		with open(path_file, "r", encoding="utf-16") as file:
			content_file = file.read()
			content_file = process_vietnamese_text(content_file) 
			
			tokens = content_file.split()
			tokens = [t for t in tokens if t not in stopwords]
			content_file = " ".join(tokens)

			list_content_file.append(content_file)

	return list_content_file

def write_file_output(list_content_file, category, isTrain):
    import os

    path_file = ""
    if isTrain:
        path_file = f"clean-data/train/{category}.txt"
    else:
        path_file = f"clean-data/test/{category}.txt"

    os.makedirs(os.path.dirname(path_file), exist_ok=True)

    with open(path_file, "w", encoding="utf-8") as file_output:
        for content_file in list_content_file:
            file_output.write(content_file + "\n")

    print(f"Wrote {len(list_content_file)} docs to: {path_file}")

#%% Standardize
def standardize_all_categories(train_or_test):
	for name_categories in list_name_categories : 
		path_file_output = "raw-data/" + train_or_test + "/" + name_categories
		list_content_file = get_list_content(path_file_output)	

		if train_or_test == "10-topics/Train_Full":
			write_file_output(list_content_file, name_categories, True)
		else :
			write_file_output(list_content_file, name_categories, False)

def standardize_text(text: str) -> str:
    return process_vietnamese_text(text)

#%% Build Dictionary

def build_dictionary(train_folder: str) -> dict:
    dictionary = {}
    for filename in os.listdir(train_folder):
        with open(os.path.join(train_folder, filename), "r", encoding="utf-8") as f:
            data = f.read()
            words = re.split(r"[ \n]", data)
            for word in words:
                if word:
                    dictionary[word] = dictionary.get(word, 0) + 1
    return dictionary

#%% Remove Stopwords
def is_numeric_token(token):
    return bool(re.search(r'\d', token))

def filter_dictionary(dictionary: dict, stopword_path="stopword.txt", min_freq=30) -> dict:
    with open(stopword_path, "r", encoding="utf-8") as f:
        stopwords = set(f.read().splitlines())

    filtered = {
        word: freq
        for word, freq in dictionary.items()
        if word not in stopwords
        and freq >= min_freq
        and not is_numeric_token(word)  
    }
    return filtered

def remove_stopwords_from_text(text: str, stopword_path="stopword.txt") -> str:
    with open(stopword_path, "r", encoding="utf-8") as f:
        stopwords = set(f.read().splitlines())

    tokens = text.split()
    filtered_tokens = [token for token in tokens if token not in stopwords]
    return ' '.join(filtered_tokens)

#%% Save Dictionary
def save_dictionary(dictionary: dict, save_path="dictionary.txt"):
    with open(save_path, "w", encoding="utf-8") as f:
        for word, freq in dictionary.items():
            f.write(f"{word} {freq}\n")

#%% Load Dictionary (keys only)
def load_dict_keys(path="dictionary.txt") -> list:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
        keys = [line.split(" ")[0] for line in lines if line]
    return keys

#%% TF-IDF 
def get_tfidf_transformer(corpus: list[str], vocab: list[str]):
    cv = CountVectorizer(vocabulary=vocab)
    word_count_vector = cv.fit_transform(corpus)
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    return cv, tfidf_transformer

def compute_tfidf_vector(text: str, cv, tfidf_transformer, vocab: list[str]) -> dict:
    vector = cv.transform([text])
    tfidf = tfidf_transformer.transform(vector)
    return dict(zip(vocab, tfidf.toarray()[0]))

#%% Preprocessing Pipeline
def preprocess_pipeline(raw_train_dir="Train_Full",
                        raw_test_dir="Test_Full",
                        clean_base_dir="clean-data",
                        top_k=2500):
    for path, is_train in [(raw_train_dir, True), (raw_test_dir, False)]:
        category_dirs = os.listdir(path)
        for category in category_dirs:
            category_path = os.path.join(path, category)
            texts = get_list_content(category_path)
            write_file_output(texts, category, isTrain=is_train)

    dictionary = build_dictionary(os.path.join(clean_base_dir, "train"))
    dictionary = filter_dictionary(dictionary)
    
    top_words = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_dict = dict(top_words)
    save_dictionary(top_dict)

    vocab = list(top_dict.keys())
    corpus = []
    for fname in os.listdir(os.path.join(clean_base_dir, "train")):
        with open(os.path.join(clean_base_dir, "train", fname), "r", encoding="utf-8") as f:
            corpus.extend(f.read().splitlines())

    cv, tfidf = get_tfidf_transformer(corpus, vocab)

    print("Preprocessing complete")
    return cv, tfidf, vocab

cv, tfidf, vocab = preprocess_pipeline()

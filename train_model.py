import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

from datasets import Dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
Trainer, TrainingArguments, DataCollatorWithPadding)
import torch

OUT_DIR = "sentiment_results"
BERT_MODEL = "bert-base-uncased"
NUM_LABELS = 2
EPOCHS = 2
BATCH_SIZE = 8
RANDOM_STATE = 42
TFIDF_MAX_FEATURES = 20000

TRAIN_PATH = r"C:\Users\HP\Downloads\test (1).csv"
TEST_PATH = r"C:\Users\HP\Downloads\test (1).csv"

os.makedirs(OUT_DIR, exist_ok=True)


try:
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)
except FileNotFoundError as e:
    print(f"Error: Could not find CSV file - {e}")
    print(f"Please ensure the files exist at:\n{TRAIN_PATH}\n{TEST_PATH}")
    exit(1)
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

df_train = df_train.rename(columns={"review_title": "product_title", "review_text": "reviewText", "class_index": "label"})
df_test = df_test.rename(columns={"review_title": "product_title", "review_text": "reviewText", "class_index": "label"})

COL_PRODUCT = "product_title"
COL_TEXT = "reviewText"

df_train = df_train.dropna(subset=[COL_TEXT, COL_PRODUCT, "label"]).reset_index(drop=True)
df_test = df_test.dropna(subset=[COL_TEXT, COL_PRODUCT, "label"]).reset_index(drop=True)

df_train["label"] = df_train["label"].astype(int)
df_test["label"] = df_test["label"].astype(int)

print("Unique labels in train:", df_train["label"].unique())
print("Unique labels in test:", df_test["label"].unique())

allowed_labels = {0, 1}
if not set(df_train["label"].unique()).issubset(allowed_labels):
    print("Non-binary labels detected in train, filtering and mapping to binary...")
    df_train = df_train[df_train["label"].isin([1, 2])].copy()
    df_train["label"] = df_train["label"].apply(lambda x: x - 1)
if not set(df_test["label"].unique()).issubset(allowed_labels):
   print("Non-binary labels detected in test, filtering and mapping to binary...")
   df_test = df_test[df_test["label"].isin([1, 2])].copy()
   df_test["label"] = df_test["label"].apply(lambda x: x - 1)

assert set(df_train["label"].unique()).issubset({0,1}), "Train labels must be 0 or 1"
assert set(df_test["label"].unique()).issubset({0,1}), "Test labels must be 0 or 1"


plt.figure(figsize=(5,4))
sns.countplot(x='label', data=df_train)
plt.title("Label distribution (train)")
plt.savefig(os.path.join(OUT_DIR, "label_distribution_train.png"), bbox_inches='tight')
plt.close()

counts = df_train[COL_PRODUCT].value_counts().head(20)
plt.figure(figsize=(10,5))
sns.barplot(x=counts.values, y=counts.index)
plt.xlabel("Number of reviews")
plt.title("Top 20 products by review count (train)")
plt.savefig(os.path.join(OUT_DIR, "top20_reviews_per_product_train.png"), bbox_inches='tight')
plt.close()

df_train['review_len'] = df_train[COL_TEXT].astype(str).apply(lambda x: len(x.split()))
plt.figure(figsize=(6,4))
sns.histplot(df_train['review_len'], bins=50, kde=False)
plt.title("Review length distribution (train)")
plt.savefig(os.path.join(OUT_DIR, "review_length_hist_train.png"), bbox_inches='tight')
plt.close()

product_stats = df_train.groupby(COL_PRODUCT).agg(
    n_reviews = (COL_TEXT, 'count'),
    avg_label = ('label', 'mean')
).sort_values('n_reviews', ascending=False)
product_stats.head(10).to_csv(os.path.join(OUT_DIR, "top_products_stats_train.csv"))

stop = set(stopwords.words('english'))
def clean_and_tokenize(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = [t for t in text.split() if t not in stop and len(t) > 2]
    return tokens

all_tokens = Counter()
for t in df_train[COL_TEXT].astype(str).head(10000):
    all_tokens.update(clean_and_tokenize(t))
top_words = all_tokens.most_common(30)
pd.DataFrame(top_words, columns=['word','count']).to_csv(os.path.join(OUT_DIR, "top_words.csv"), index=False)

tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=(1,2))
tfidf.fit(df_train[COL_TEXT])
X_train_tfidf = tfidf.transform(df_train[COL_TEXT])
X_test_tfidf = tfidf.transform(df_test[COL_TEXT])

dt = DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE)
dt.fit(X_train_tfidf, df_train["label"])
y_test_pred = dt.predict(X_test_tfidf)

with open(os.path.join(OUT_DIR, "dt_classification_report.txt"), 'w') as f:
    f.write(classification_report(df_test["label"], y_test_pred))

plt.figure(figsize=(20,10))
plot_tree(dt, max_depth=3, class_names=['neg','pos'], filled=True)
plt.savefig(os.path.join(OUT_DIR, "decision_tree_top_levels.png"), bbox_inches='tight')
plt.close()

dataset_dict = DatasetDict({
    'train': Dataset.from_pandas(df_train[[COL_TEXT, "label"]].rename(columns={COL_TEXT: "text"}).reset_index(drop=True)),
    'test': Dataset.from_pandas(df_test[[COL_TEXT, "label"]].rename(columns={COL_TEXT: "text"}).reset_index(drop=True))
})

dataset_dict = dataset_dict["train"].train_test_split(test_size=0.1, seed=RANDOM_STATE)
dataset_dict = DatasetDict({
    'train': dataset_dict["train"],
    'validation': dataset_dict["test"],
    'test': Dataset.from_pandas(df_test[[COL_TEXT, "label"]].rename(columns={COL_TEXT: "text"}).reset_index(drop=True))
})

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
def tokenize_fn(ex):
    return tokenizer(ex['text'], truncation=True, padding=False, max_length=256)

tokenized = dataset_dict.map(tokenize_fn, batched=True, remove_columns=['text'])
model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=NUM_LABELS)

import gc
torch.cuda.empty_cache()
gc.collect()

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=-1)
    acc = (preds == labels).mean()
    return {'accuracy': acc}

training_args = TrainingArguments(
    output_dir=os.path.join(OUT_DIR, "bert_output"),
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_steps=50,
    metric_for_best_model="accuracy",
    save_total_limit=2,
    seed=RANDOM_STATE
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

model_save_path = os.path.join(OUT_DIR, "bert_output")
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model and tokenizer saved to {model_save_path}")

test_res = trainer.predict(tokenized['test'])
test_preds = np.argmax(test_res.predictions, axis=-1)
test_acc = accuracy_score(tokenized['test']['label'], test_preds)

with open(os.path.join(OUT_DIR, "bert_test_report.txt"), "w") as f:
    f.write(f"Test accuracy: {test_acc}\n")
    f.write(classification_report(tokenized['test']['label'], test_preds))

full_df = pd.concat([df_train, df_test], ignore_index=True)
full_ds = Dataset.from_pandas(full_df.rename(columns={COL_TEXT: 'text'}))
full_tok = full_ds.map(tokenize_fn, batched=True, remove_columns=['text'])
preds = trainer.predict(full_tok)
probs = torch.nn.functional.softmax(torch.tensor(preds.predictions), dim=1).numpy()
full_df['pred_pos_prob'] = probs[:,1]
full_df['pred_label'] = (full_df['pred_pos_prob'] >= 0.5).astype(int)

product_scores = full_df.groupby(COL_PRODUCT).agg(
    avg_pred_pos = ('pred_pos_prob','mean'),
    n_reviews = (COL_TEXT,'count'),
    avg_true_label = ('label','mean')
).sort_values('avg_pred_pos', ascending=False)
product_scores.to_csv(os.path.join(OUT_DIR, "product_scores_by_pred_prob.csv"))

product_scores.head(10).to_csv(os.path.join(OUT_DIR, "best_products.csv"))
product_scores.tail(10).to_csv(os.path.join(OUT_DIR, "worst_products.csv"))

full_df.to_csv(os.path.join(OUT_DIR, "reviews_with_preds.csv"), index=False)

from IPython.display import Image, display

image_files = [
    os.path.join(OUT_DIR, "label_distribution_train.png"),
    os.path.join(OUT_DIR, "top20_reviews_per_product_train.png"),
    os.path.join(OUT_DIR, "review_length_hist_train.png"),
    os.path.join(OUT_DIR, "decision_tree_top_levels.png")
]

for img_path in image_files:
    display(Image(filename=img_path))


print("\nBERT Test Report:")
with open(os.path.join(OUT_DIR, "bert_test_report.txt")) as f:
    print(f.read())

print("\nDecision Tree Classification Report:")
with open(os.path.join(OUT_DIR, "dt_classification_report.txt")) as f:
    print(f.read())

print("\nProduct Scores (Top 5):")
product_scores = pd.read_csv(os.path.join(OUT_DIR, "product_scores_by_pred_prob.csv"))
print(product_scores.head())

print("\nBest Products:")
best_products = pd.read_csv(os.path.join(OUT_DIR, "best_products.csv"))
print(best_products)

print("\nWorst Products:")
worst_products = pd.read_csv(os.path.join(OUT_DIR, "worst_products.csv"))
print(worst_products)

print("\nSample Reviews with Predictions:")
reviews_with_preds = pd.read_csv(os.path.join(OUT_DIR, "reviews_with_preds.csv"))
print(reviews_with_preds.head())
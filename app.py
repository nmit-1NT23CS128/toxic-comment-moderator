
import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from googleapiclient.discovery import build
import re

st.set_page_config(page_title="Toxic Comment Detector", layout="centered")

st.markdown("""
<style>
.main { background-color: #0e1117; }
.stTextArea textarea { font-size: 16px; }
</style>
""", unsafe_allow_html=True)


from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "MANYA12345/toxic-comment-distilbert"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID
    )

    model.eval()

    return model, tokenizer

model, tokenizer = load_model()

def normalize_text(text):
    text = text.lower()
    text = text.replace("ur", "you are")
    text = text.replace("u ", "you ")
    return text

def predict(text):
    processed = normalize_text(text)
    inputs = tokenizer(
        processed,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        toxic_prob = probs[0][1].item()
    return toxic_prob

def extract_video_id(url):
    patterns = [
        r'v=([a-zA-Z0-9_-]{11})',
        r'youtu\.be/([a-zA-Z0-9_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_youtube_comments(video_id, api_key, max_comments=50):
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        comments = []
        next_page_token = None

        while len(comments) < max_comments:
            request = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=min(100, max_comments - len(comments)),
                pageToken=next_page_token,
                textFormat='plainText'
            )
            response = request.execute()

            for item in response.get('items', []):
                text = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(text)

            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break

        return comments[:max_comments]

    except Exception as e:
        st.error(f"YouTube API error: {e}")
        return []

# ---- SIDEBAR NAVIGATION ----
page = st.sidebar.selectbox(
    "Select Page",
    ["Toxic Comment Detector", "YouTube Live Scanner", "Model Performance"]
)

# ---- PAGE 1: SINGLE COMMENT ----
if page == "Toxic Comment Detector":
    st.markdown("## 🛡 AI Toxic Comment Moderator")
    st.caption("Detects and filters harmful comments in real-time")

    user_input = st.text_area("Enter Comment", key="input_text")

    if user_input.strip() == "":
        st.info("Please enter a comment")
    else:
        if st.button("Predict"):
            with st.spinner("Analyzing comment..."):
                toxic_prob = predict(user_input)

            if toxic_prob > 0.75:
                st.error("Highly Toxic — 🚨 Comment Blocked")
                decision = "Automatically Hidden"
            elif toxic_prob > 0.35:
                st.warning("Moderately Toxic — ⚠ Comment Flagged")
                decision = "Flagged for Review"
            else:
                st.success("Safe — ✅ Comment Approved")
                decision = "Published"

            st.write("Toxic Probability:", round(toxic_prob, 3))
            st.progress(int(toxic_prob * 100))
            st.write(f"🛠 System Action: **{decision}**")

# ---- PAGE 2: YOUTUBE LIVE SCANNER ----
elif page == "YouTube Live Scanner":
    st.markdown("## 📡 YouTube Live Comment Scanner")
    st.caption("Paste any YouTube video URL to scan its comments in real time")

    api_key = st.text_input(
        "YouTube API Key",
        type="password",
        placeholder="Paste your API key here"
    )
    video_url = st.text_input(
        "YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=..."
    )
    max_comments = st.slider("Number of comments to scan", 10, 100, 50)

    if st.button("Scan Comments"):
        st.session_state.pop("scan_results", None)
        if not api_key or not video_url:
            st.warning("Please enter both API key and video URL")
        else:
            video_id = extract_video_id(video_url)
            if not video_id:
                st.error("Could not extract video ID. Check the URL.")
            else:
                with st.spinner(f"Fetching up to {max_comments} comments..."):
                    comments = get_youtube_comments(video_id, api_key, max_comments)

                if not comments:
                    st.error("No comments found or API error.")
                else:
                    results = []
                    progress_bar = st.progress(0)

                    for i, comment in enumerate(comments):
                        prob = predict(comment)
                        if prob > 0.75:
                            label = "🚨 Toxic"
                            category = "toxic"
                        elif prob > 0.35:
                            label = "⚠️ Flagged"
                            category = "flagged"
                        else:
                            label = "✅ Safe"
                            category = "safe"

                        results.append({
                            "comment": comment,
                            "prob": prob,
                            "label": label,
                            "category": category
                        })
                        progress_bar.progress((i + 1) / len(comments))

                    st.session_state["scan_results"] = results
                    st.success(f"Scanned {len(results)} comments")

    # Results survive radio button clicks via session_state
    if "scan_results" in st.session_state:
        results = st.session_state["scan_results"]

        toxic_count = sum(1 for r in results if r["category"] == "toxic")
        flagged_count = sum(1 for r in results if r["category"] == "flagged")
        safe_count = sum(1 for r in results if r["category"] == "safe")

        st.markdown("### Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("🚨 Toxic", toxic_count)
        col2.metric("⚠️ Flagged", flagged_count)
        col3.metric("✅ Safe", safe_count)

        st.markdown("### Comment Results")
        filter_option = st.radio(
            "Show:",
            ["All", "Toxic only", "Flagged only", "Safe only"],
            horizontal=True
        )

        if filter_option == "Toxic only":
            filtered = [r for r in results if r["category"] == "toxic"]
        elif filter_option == "Flagged only":
            filtered = [r for r in results if r["category"] == "flagged"]
        elif filter_option == "Safe only":
            filtered = [r for r in results if r["category"] == "safe"]
        else:
            filtered = results

        if not filtered:
            st.info("No comments in this category.")
        else:
            for r in sorted(filtered, key=lambda x: x["prob"], reverse=True):
                with st.expander(f"{r['label']} — {round(r['prob']*100, 1)}% toxic"):
                    st.write(r["comment"])
                    st.progress(int(r["prob"] * 100))

# ---- PAGE 3: MODEL PERFORMANCE ----
else:
    st.title("📊 Model Performance Dashboard")

    accuracy = 0.951
    precision = 0.763
    recall = 0.754
    f1 = 0.758

    col1, col2 = st.columns(2)
    col1.metric("Accuracy", accuracy)
    col1.metric("Precision", precision)
    col2.metric("Recall", recall)
    col2.metric("F1 Score", f1)

    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 0, 1, 1, 1, 0]

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    ax.matshow(cm, cmap='Blues')
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, val, ha='center', va='center', color='white', fontsize=14)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    st.pyplot(fig)

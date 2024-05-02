from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
import torch.nn.functional as F

# Load Sentence Transformer Model for Semantic Analysis
semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Load Thematic Classification Model
thematic_model = AutoModelForSequenceClassification.from_pretrained("eleldar/theme-classification")
thematic_tokenizer = AutoTokenizer.from_pretrained("eleldar/theme-classification")

# Load a sentiment analysis pipeline for Emotional Analysis
emotion_analysis = pipeline("sentiment-analysis")


def get_semantic_embedding(text):
    """Generate semantic embedding for the given text."""
    return semantic_model.encode(text, convert_to_tensor=True)


def get_thematic_classification(text):
    """Classify the theme of the given text using logits and return probabilities."""
    inputs = thematic_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = thematic_model(**inputs).logits
    return torch.softmax(logits, dim=-1)


def get_emotional_analysis(text):
    """Perform emotion analysis on the given text and return the result."""
    return emotion_analysis(text)


def find_similar_sentences(base_text, list_of_texts):
    """Find and print the three most similar sentences based on semantic similarity."""
    base_embedding = get_semantic_embedding(base_text)
    list_embeddings = semantic_model.encode(list_of_texts, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(base_embedding, list_embeddings)[0]

    # Get top 3 similar sentences
    top_results = torch.topk(cosine_scores, k=3)

    for score, idx in zip(top_results[0], top_results[1]):
        print(f"Match: {list_of_texts[idx]} \nScore: {score.item()}\n")
        
        
def get_thematic_similarity(theme_probs1, theme_probs2):
    """Calculate similarity between two thematic probability distributions."""
    # Normalize the probabilities to unit length for cosine similarity
    theme_probs1 = F.normalize(theme_probs1, p=2, dim=1)
    theme_probs2 = F.normalize(theme_probs2, p=2, dim=1)
    return torch.cosine_similarity(theme_probs1, theme_probs2, dim=1).item()
        
        
def combined_analysis_and_matching(text, list_of_texts):
    base_embedding = get_semantic_embedding(text)
    base_emotion = get_emotional_analysis(text)[0]
    base_theme = get_thematic_classification(text)

    similarities = []
    for other_text in list_of_texts:
        other_embedding = get_semantic_embedding(other_text)
        other_emotion = get_emotional_analysis(other_text)[0]
        other_theme = get_thematic_classification(other_text)

        # Calculate cosine similarity for semantic content
        semantic_similarity = util.pytorch_cos_sim(base_embedding, other_embedding)[0].item()

        # Emotional alignment (assumes sentiment scores are between 0 and 1)
        emotional_alignment = 1 - abs(base_emotion['score'] - other_emotion['score'])

        # Thematic alignment as cosine similarity of softmax probabilities
        thematic_alignment = get_thematic_similarity(base_theme, other_theme)

        # Combine scores (adjust weighting as needed)
        combined_score = (semantic_similarity + emotional_alignment + thematic_alignment) / 3
        similarities.append((other_text, combined_score))

    # Sort based on the combined score
    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return sorted_similarities[:3]


sentences = [
  "Yıldızları izlemek, evrenin büyüklüğü ve kendi endişelerimin küçüklüğü hakkında düşünmemi sağlıyor.",
  "Kitaplar, zaman ve mekan sınırlarını aşarak, bizi farklı dünyalara götürebilir.",
  "Her sabah uyandığımda, yeni bir günün benim için ne getireceğini merak ediyorum.",
  "Kediler özgür ruhlar olabilir, ama aynı zamanda evde sıcak bir kucak bulmayı da severler.",
  "Ev yapımı ekmek, mutfaktan yükselen mis gibi kokusuyla, evin her köşesini ısıtır.",
  "Klasik müzik dinlemek, bazen anlatılamayan duyguları ifade etmenin en güzel yoludur.",
  "Deniz kenarında yürümek, hem bedensel hem de ruhsal dinlenme sağlar.",
  "Bir kahve ve iyi bir kitap, mükemmel bir öğleden sonra için ihtiyacım olan tek şey.",
  "Baharın gelişiyle birlikte, doğa yeniden canlanır ve çiçekler açar.",
  "Gelecek hafta çok önemli bir proje sunumum var bu yüzden oldukça stresliyim.",
  "Ailemle ilgili durum biraz karmaşık. Son zamanlarda bazı sorunlar yaşamaya başladık. Anlaşmazlıklar ve yanlış anlaşılmalar yüzünden aile içinde gerginlikler oluyor. Bazen bu durum beni üzüyor ve kendimi yalnız hissetmeme neden oluyor. Annem ve babamla konuşmaya çalışıyorum, ama bazen onlarla iletişim kurmak gerçekten zorlaşıyor.",
  "Arkadaşlarımla son zamanlarda daha az görüşebiliyorum. Herkesin işi ve kişisel sorunları var. Bu durum bazen beni üzse de, herkesin yaşamında böyle dönemler olabileceğini anlıyorum.",
  "Gezegenimizi korumak, gelecek nesiller için temiz bir çevre bırakma sorumluluğumuzdur.",
  "Kamp yapmak, doğayla iç içe olmanın ve günlük yaşamın stresinden uzaklaşmanın harika bir yoludur.",
  "Pişirdiğim her yemek, sevdiklerime olan sevgimi göstermenin bir yolu.",
  "Fotoğraf çekmek, anları ölümsüzleştirmenin ve yaşadığımız güzellikleri belgelemenin güçlü bir aracıdır.",
  "Dünya çapında her gün milyonlarca insan toplu taşıma kullanıyor.",
  "Bilgisayar oyunları, gerçek dünyadan bir süreliğine kaçmak için harika bir yoldur.",
  "Her kültürün yemekleri, o toplumun tarihini ve değerlerini yansıtır.",
  "Hayat boş",
  "Yıldızlara bakmayı severim, çünkü bu bana evrenin ne kadar büyük olduğunu ve endişelerimin ne kadar küçük kaldığını hatırlatır.",
  "Nefes egzersizleri yapmak, kaygı anlarında bana yardımcı oluyor.",
  "Sosyal medya, gerçek dünyadan biraz uzaklaşmak ve farklı insanların hayatlarına bir pencere açmak için kullanılabilir.",
  "Geceleri uyuyamamak, kontrol edemediğim şeyler için endişelendiğim zamanlar oluyor."
]

base_text = "Rüyamda köpeğim ile mutfakta oturuyordum başka kimse yoktu köpek bir anda bana saldırmaya başladı ve kavga ettik"
similar_sentences = combined_analysis_and_matching(base_text, sentences)
for sentence, score in similar_sentences:
    print(f"Match: {sentence}\nScore: {score}\n")

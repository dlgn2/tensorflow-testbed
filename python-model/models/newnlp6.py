import nltk
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import string

# NLTK resources and Turkish stopwords
nltk.download('stopwords')
nltk.download('punkt')
turkish_stopwords = set(stopwords.words('turkish'))

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
model = AutoModel.from_pretrained("dbmdz/bert-base-turkish-cased")

def preprocess(text):
    # Clean and preprocess text
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in turkish_stopwords and word.isalpha()]
    return ' '.join(filtered_tokens)

def encode(texts):
    # Encode texts after preprocessing
    texts = [preprocess(text) for text in texts]
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
    with torch.no_grad():
        model_output = model(**encoded_input)
    # Use mean pooling on last hidden state
    embeddings = model_output.last_hidden_state.mean(dim=1)
    return embeddings

def find_related_sentences(input_sentence, sentences):
    # Encode and find related sentences using cosine similarity
    input_features = encode([input_sentence])
    sentence_features = encode(sentences)
    
    # Ensure embeddings are 2D
    input_features = input_features.unsqueeze(0) if input_features.dim() == 1 else input_features
    sentence_features = sentence_features.unsqueeze(0) if sentence_features.dim() == 1 else sentence_features
    
    similarities = cosine_similarity(input_features, sentence_features)[0]
    top_indices = similarities.argsort()[-2:][::-1]
    return [(sentences[i], similarities[i]) for i in top_indices]

# Example usage
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

input_sentence = "Geceleri uyuyamamak beni üzüyor."
related_sentences = find_related_sentences(input_sentence, sentences)
print(related_sentences)

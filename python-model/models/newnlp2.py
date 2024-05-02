from transformers import AutoTokenizer, AutoModel, pipeline
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Model ve tokenizer'ı yükleyelim
tokenizer = AutoTokenizer.from_pretrained("Snowflake/snowflake-arctic-embed-m")
model = AutoModel.from_pretrained("Snowflake/snowflake-arctic-embed-m")

# Duygu analizi için pipeline
sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")


def encode(texts):
    # Metinleri tokenize edip input tensorlarını hazırlayalım
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
    
    # Gömülüler (embeddings) üretelim
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Token gömülülerinin ortalamasını alalım
    input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(model_output.last_hidden_state.size()).float()
    sum_embeddings = torch.sum(model_output.last_hidden_state * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def encode_with_sentiment(texts):
    embeddings = encode(texts)  # Gömülüler elde edin
    sentiments = sentiment_model(texts)  # Duygu analizi yap

    # Duygu skorlarını numpy array olarak elde edelim ve tensora dönüştürelim
    sentiment_features = torch.tensor([[s['score'] if s['label'] == 'POSITIVE' else -s['score'] for s in sentiments]])
    full_features = torch.cat((embeddings, sentiment_features.T), dim=1)
    return full_features


def find_related_sentences_advanced(input_sentence, sentences):
    # Girdi cümlesini ve diğer tüm cümleleri kodlayalım
    input_features = encode_with_sentiment([input_sentence])
    sentence_features = encode_with_sentiment(sentences)
    
    # Kosinüs benzerliği hesaplayalım
    similarities = cosine_similarity(input_features, sentence_features)[0]
    
    # En yüksek benzerlik skorlarına sahip iki indexi bulalım
    top_indices = similarities.argsort()[-2:][::-1]  # En iyi 2 index
    
    # En ilgili cümleleri ve skorlarını dönelim
    return [(sentences[i], similarities[i]) for i in top_indices]


# Test edelim
sentences = [
  "Yıldızları izlemek, evrenin büyüklüğü ve kendi endişelerimin küçüklüğü hakkında düşünmemi sağlıyor.",
  "Kitaplar, zaman ve mekan sınırlarını aşarak, bizi farklı dünyalara götürebilir.",
  "Her sabah uyandığımda, yeni bir günün benim için ne getireceğini merak ediyorum.",
  "Kediler özgür ruhlar olabilir, ama aynı zamanda evde sıcak bir kucak bulmayı da severler.",
  "Ev yapımı ekmek, mutfaktan yükselen mis gibi kokusuyla, evin her köşesini ısıtır.",
  "Klasik müzik dinlemek, bazen anlatılamayan duyguları ifade etmenin en güzel yoludur.",
  "Deniz kenarında yürümek, hem bedensel hem de ruhsal dinlenme sağlar.",
  "Bir kahve ve iyi bir kitap, mükemmel bir öğleden sonra için ihtiyacım olan tek şey.",
    "Gelecek hafta çok önemli bir proje sunumum var bu yüzden oldukça stresliyim.",
  "Baharın gelişiyle birlikte, doğa yeniden canlanır ve çiçekler açar.",
      "Ailemle ilgili durum biraz karmaşık. Son zamanlarda bazı sorunlar yaşamaya başladık. Anlaşmazlıklar ve yanlış anlaşılmalar yüzünden aile içinde gerginlikler oluyor. Bazen bu durum beni üzüyor ve kendimi yalnız hissetmeme neden oluyor. Annem ve babamla konuşmaya çalışıyorum, ama bazen onlarla iletişim kurmak gerçekten zorlaşıyor.",
    "Arkadaşlarımla son zamanlarda daha az görüşebiliyorum. Herkesin işi ve kişisel sorunları var. Bu durum bazen beni üzse de, herkesin yaşamında böyle dönemler olabileceğini anlıyorum.",
  "Gezegenimizi korumak, gelecek nesiller için temiz bir çevre bırakma sorumluluğumuzdur.",
  "Kamp yapmak, doğayla iç içe olmanın ve günlük yaşamın stresinden uzaklaşmanın harika bir yoludur.",
  "Pişirdiğim her yemek, sevdiklerime olan sevgimi göstermenin bir yolu.",
  "Fotoğraf çekmek, anları ölümsüzleştirmenin ve yaşadığımız güzellikleri belgelemenin güçlü bir aracıdır.",
  "Dünya çapında her gün milyonlarca insan toplu taşıma kullanıyor.",
  "Bilgisayar oyunları, gerçek dünyadan bir süreliğine kaçmak için harika bir yoldur.",
  "Her kültürün yemekleri, o toplumun tarihini ve değerlerini yansıtır.",
  "Yıldızlara bakmayı severim, çünkü bu bana evrenin ne kadar büyük olduğunu ve endişelerimin ne kadar küçük kaldığını hatırlatır.",
  "Nefes egzersizleri yapmak, kaygı anlarında bana yardımcı oluyor.",
  "Sosyal medya, gerçek dünyadan biraz uzaklaşmak ve farklı insanların hayatlarına bir pencere açmak için kullanılabilir.",
  "Geceleri uyuyamamak, kontrol edemediğim şeyler için endişelendiğim zamanlar oluyor."
]
input_sentence = "Bugün sınavım var iyi geçmez diye korkuyorum."
related_sentences = find_related_sentences_advanced(input_sentence, sentences)
print(related_sentences)

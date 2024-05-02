from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Snowflake/snowflake-arctic-embed-m")
model = AutoModel.from_pretrained("Snowflake/snowflake-arctic-embed-m")


def encode(texts):
    # Tokenize the texts and prepare input tensors
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
    
    # Generate embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Mean pooling the token embeddings
    input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(model_output.last_hidden_state.size()).float()
    sum_embeddings = torch.sum(model_output.last_hidden_state * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


# Precompute embeddings for all psychological sentences
psych_sentences = [
  "Yıldızları izlemek, evrenin büyüklüğü ve kendi endişelerimin küçüklüğü hakkında düşünmemi sağlıyor.",
  "Kitaplar, zaman ve mekan sınırlarını aşarak, bizi farklı dünyalara götürebilir.",
  "Her sabah uyandığımda, yeni bir günün benim için ne getireceğini merak ediyorum.",
  "Kediler özgür ruhlar olabilir, ama aynı zamanda evde sıcak bir kucak bulmayı da severler.",
  "Ev yapımı ekmek, mutfaktan yükselen mis gibi kokusuyla, evin her köşesini ısıtır.",
  "Klasik müzik dinlemek, bazen anlatılamayan duyguları ifade etmenin en güzel yoludur.",
  "Deniz kenarında yürümek, hem bedensel hem de ruhsal dinlenme sağlar.",
  "Bir kahve ve iyi bir kitap, mükemmel bir öğleden sonra için ihtiyacım olan tek şey.",
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
# Encode all psych sentences at once
psych_embeddings = encode(psych_sentences)


def find_related_sentences_advanced(input_sentence):
    # Encode the input sentence to get its embedding
    input_embedding = encode([input_sentence])
    
    # Calculate cosine similarities
    similarities = cosine_similarity(input_embedding, psych_embeddings)[0]
    
    # Find indices of sentences with the highest similarity scores
    top_indices = similarities.argsort()[-2:][::-1]  # Get top 2 indices
    
    # Return the most related sentences with their scores
    return [(psych_sentences[i], similarities[i]) for i in top_indices]


# Test with a new sentence in Turkish
input_sentence = "Rüyamda köpeğim ile mutfakta oturuyordum başka kimse yoktu kedim bir anda bana saldırmaya başladı. "
related_advanced = find_related_sentences_advanced(input_sentence)
print(related_advanced)

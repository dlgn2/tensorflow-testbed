from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load a pre-trained model designed for sentence embeddings
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


def find_related_sentences(input_sentence, sentences):
    # Encode all sentences to get embeddings
    input_embedding = model.encode(input_sentence, convert_to_tensor=True)
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    
    # Calculate cosine similarities
    cosine_scores = util.pytorch_cos_sim(input_embedding, sentence_embeddings)[0]

    # Move tensor to CPU for NumPy conversion and argsort
    cosine_scores = cosine_scores.cpu().numpy()
    
    # Find the indices of the top two scores
    top_results = np.argsort(cosine_scores)[::-1][:3]
    
    # Return the most related sentences with their scores
    return [(sentences[idx], cosine_scores[idx]) for idx in top_results]


# Test the function with example sentences
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

input_sentence = "Rüyamda köpeğim ile mutfakta oturuyordum başka kimse yoktu köpek bir anda bana saldırmaya başladı ve kavga ettik"
related_sentences = find_related_sentences(input_sentence, sentences)
print(related_sentences)

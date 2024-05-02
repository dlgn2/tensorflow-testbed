from sentence_transformers import SentenceTransformer, util

# Load a multilingual model designed for semantic textual similarity
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


# Define your sentences (you can mix English and Turkish)
psych_sentences = [
    "Son zamanlarda iş yerinde çok yoğunum. Projeler üst üste geldi ve sürekli mesai yapmak zorunda kalıyorum. Bu tempo beni gerçekten yoruyor, fakat işimi seviyorum. Yaratıcılığımı kullanabildiğim ve sürekli yeni şeyler öğrendiğim bir ortamda çalışmak beni motive ediyor.",
    "Arkadaşlarımla vakit geçirmeyi seviyorum. Hafta sonları genellikle kahve içmeye veya kısa şehir dışı gezilere çıkıyoruz. Bu tür etkinlikler, hafta içi yaşadığım stresi atmama yardımcı oluyor. Ayrıca kitap okumayı ve müzik dinlemeyi çok seviyorum, bu yüzden boş zamanlarımı genellikle bu şekilde değerlendiriyorum.",
    "Ailemle ilgili durum biraz karmaşık. Son zamanlarda bazı sorunlar yaşamaya başladık. Anlaşmazlıklar ve yanlış anlaşılmalar yüzünden aile içinde gerginlikler oluyor. Bazen bu durum beni üzüyor ve kendimi yalnız hissetmeme neden oluyor. Annem ve babamla konuşmaya çalışıyorum, ama bazen onlarla iletişim kurmak gerçekten zorlaşıyor.",
    "Hafta içi sabahları genellikle koşuya çıkıyorum. Bu, günü enerjik başlamama yardımcı oluyor ve kafamı dağıtıyor. Koşarken genellikle günün planını düşünüyorum ve bu da işlerimi organize etmeme yardımcı oluyor.",
    "Son birkaç aydır, kendi kişisel gelişimime daha fazla odaklanmaya çalışıyorum. Yeni bir dil öğrenmeye başladım ve bu benim için oldukça zorlayıcı ama aynı zamanda çok ödüllendirici.",
    "Arkadaşlarımla son zamanlarda daha az görüşebiliyorum. Herkesin işi ve kişisel sorunları var. Bu durum bazen beni üzse de, herkesin yaşamında böyle dönemler olabileceğini anlıyorum.",
    "İşte yeni bir projeye başladım ve bu projede liderlik etme fırsatı buldum. Bu, profesyonel olarak benim için büyük bir adım. Sorumluluk almak ve bir ekibi yönetmek zorlayıcı olsa da, bu tür deneyimlerin bana çok şey kattığını düşünüyorum.",
    "Ailemle ilgili sorunlar devam ediyor. Özellikle kardeşimle aramızdaki anlaşmazlıklar beni zorluyor. Bazen onunla düzgün bir diyalog kurmak neredeyse imkansız hale geliyor.",
    "Geçen hafta eski bir arkadaşımı ziyaret ettim ve uzun uzun sohbet ettik. Bu, eski günleri yad etmek ve biraz olsun rahatlamak için iyi bir fırsat oldu.",
    "Yakın zamanda yoga yapmaya başladım. Yoga, hem fiziksel hem de zihinsel sağlığım için gerçekten iyi geliyor. Daha sakin ve odaklanmış hissediyorum.",
    "Ailevi sorunlar nedeniyle son zamanlarda oldukça stresliyim. Bu durum bazen uyku düzenimi de bozuyor. Daha iyi hissetmek için meditasyon yapmayı deniyorum.",
    "Geçenlerde bir hayvan barınağında gönüllü olarak çalışmaya başladım. Hayvanlarla vakit geçirmek, onlara bakmak beni mutlu ediyor ve günlük hayatın stresinden uzaklaşmamı sağlıyor.",
    "Bu ay içinde birkaç kez ailemle yemeğe çıktık. Bu tür buluşmalar, birbirimizi daha iyi anlamamıza ve aramızdaki sorunları yumuşatmamıza yardımcı oluyor."
]

# Compute embeddings for the sentences
psych_sentence_embeddings = model.encode(psych_sentences, convert_to_tensor=True)

def find_related_sentences_advanced(input_sentence):
    input_embedding = model.encode(input_sentence, convert_to_tensor=True)
    # Compute cosine similarities
    similarities = util.pytorch_cos_sim(input_embedding, psych_sentence_embeddings)[0]
    # Sort by highest similarities
    related_indices = similarities.argsort(descending=True)[:2]  # Top 2 related sentences
    return [(psych_sentences[i], similarities[i].item()) for i in related_indices]

# Test with a new sentence in Turkish
input_sentence = "Rüyamda, kedim gri sisler içinde kaybolmuş bir ormanda beni yönlendiriyordu. Her adımda, gölgeler arasında beliren kırık dallar ve düşmüş yapraklar vardı. Kedim, bu kırıklıkların arasından geçerken, onların çözülmemiş anlaşmazlıkları ve kopuk bağları temsil ettiğini fark ettim."  # Turkish for "I'm worried about my exams next week."
related_advanced = find_related_sentences_advanced(input_sentence)
print(related_advanced)

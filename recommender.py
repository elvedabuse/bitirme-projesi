import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Örnek veri yükleme
df = pd.read_csv("data/sample_data.csv")

# Basit örnek algoritma (kullanıcı-ürün matrisi üzerinden)
pivot = df.pivot_table(index='UserID', columns='ProductID', values='Score').fillna(0)
similarity = cosine_similarity(pivot)

print("Benzerlik matrisi:")
print(similarity)

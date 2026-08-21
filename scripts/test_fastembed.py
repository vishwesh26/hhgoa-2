import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath("."))
sys.stdout.reconfigure(encoding='utf-8')

from fastembed import TextEmbedding

print("Loading FastEmbed Multilingual ONNX model...")
model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

query = "कॉर्पोरेशन मतलब क्या?"
p_relevant = "मैकडॉनल्ड कॉर्पोरेशन दुनिया के सबसे पहचानने योग्य निगमों में से एक है। एक निगम एक कंपनी या लोगों का समूह है जो एक एकल इकाई के रूप में कार्य करने के लिए अधिकृत है और कानून में इस तरह मान्यता प्राप्त है।"
p_irrelevant = "स्कॉट्सडेल निवासी शहर के कॉर्पोरेशन यार्ड, 9191 ई. सैन सल्वाडोर ड्राइव पर सुबह 7:30 बजे से दोपहर 2 बजे तक वस्तुओं को छोड़ सकते हैं।"
p_irrelevant2 = "3 डोंगफेंग मोटर कॉर्पोरेशन - सिट्रोएन: सिट्रोएन फुकांग कॉम्पैक्ट कार।"

docs = [query, p_relevant, p_irrelevant, p_irrelevant2]
embeddings = list(model.embed(docs))

q_vec = embeddings[0]
rel_vec = embeddings[1]
irrel_vec = embeddings[2]
irrel2_vec = embeddings[3]

def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

print(f"\nQuery: '{query}'")
print(f"1. Relevant Definition Passage Sim:   {cos_sim(q_vec, rel_vec):.4f}")
print(f"2. Irrelevant Scottsdale Yard Sim:     {cos_sim(q_vec, irrel_vec):.4f}")
print(f"3. Irrelevant Dongfeng Motor Sim:      {cos_sim(q_vec, irrel2_vec):.4f}")

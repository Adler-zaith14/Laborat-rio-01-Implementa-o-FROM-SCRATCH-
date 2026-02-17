# LAB P1-01: Scaled Dot-Product Attention

**Disciplina:** Tópicos em Inteligência Artificial  
**Instituição:** ICEV  
**Autor:** Adler Castro Alves  

---

## Objetivo

Implementação from scratch do mecanismo de Scaled Dot-Product Attention conforme o paper "Attention Is All You Need" (Vaswani et al., 2017), utilizando apenas NumPy. A visualização do heatmap de pesos foi realizada no Google Colab; RESPOTA FEITA NO COLAB- REPOSITORIO DA ATIVIDADE:https://colab.research.google.com/drive/1rWZSSKKTzgqIezLbTxgXQmNmlKnI87jG?usp=sharing

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

---

## Como Executar

```bash
pip install -r requirements.txt

python attention.py
python test_attention.py
python visualize_attention.py
```

---

## Estrutura

```
├── attention.py            # Implementação principal
├── test_attention.py       # Testes unitários
├── visualize_attention.py  # Heatmap dos pesos
├── requirements.txt        # Dependências
└── README.md
```

---

## Normalização por √d_k

O produto escalar QK^T cresce proporcionalmente à dimensão d_k. Sem o fator de escala, os valores ficam grandes demais e o softmax satura — os gradientes ficam próximos de zero e o aprendizado trava. Dividir por √d_k mantém a variância dos scores estável independentemente da dimensão escolhida.

```python
scaled_scores = scores / np.sqrt(d_k)
```

---

## Exemplo de Input e Output

```python
from attention import ScaledDotProductAttention, generate_random_qkv

Q, K, V = generate_random_qkv(seq_len=4, d_k=8)
attention = ScaledDotProductAttention()
output, weights = attention(Q, K, V)
```

**Input:**
```
Q: (4, 8)
K: (4, 8)
V: (4, 8)
```

**Output:**
```
output:  (4, 8)
weights: (4, 4)  — cada linha soma 1.0
```

---

## Referência

Vaswani et al. (2017) — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
In this repository you will find the code for all examples throughout the book Hands-On Large Language Models written by Jay Alammar and Maarten Grootendorst which we playfully dubbed:https://github.com/handsOnLLM/Hands-On-Large-Language-Models?tab=readme-ov-file

Arquivos Fontes: 

https://www.tensorflow.org/text/tutorials/transformer?hl=pt-br - Entendimento de alguns conceitos pra gerar o heatmap
https://github.com/handsOnLLM/Hands-On-Large-Language-Models?tab=readme-ov-file - Entendimento de alguns conceitos do Lenguager models
https://www.alura.com.br/busca?query=transformers - Fundamentos
https://colab.research.google.com/github/tensorflow/text/blob/master/docs/tutorials/transformer.ipynb?hl=pt-br#scrollTo=GpAXuTgZ888M - Conceitos sobre os conceitos dos vetores e suas semânticas e projeções lineares

Vidoes do YOUTUBE como base de estudo e pesquisa: 

https://youtu.be/k1ILy23t89E?si=bX07-FIBXy1q-BMS- Explicação sobre inside an LLM 
https://youtu.be/eMlx5fFNoYc?si=euw_0xyK97YqxBDl- Transformers explicado visualmente
https://youtu.be/avjX3QrYkls?si=osxEKWRVfYrWhogI- Step-by-setp-explained (Attetion)
https://youtu.be/4Bdc55j80l8?si=oCUOQHwd_YIIK0rg- Encoder and Decoder (Transformers)
https://youtu.be/8HyCNIVRbSU?si=PDVpFX0z6_kPdIGD- Illustrated Guide to LSTM´s
https://youtu.be/94PlBzgeq90?si=QXmeiNsiTQcUAv2E- Tutorial series temporales LSTM
https://youtu.be/oSn6o1gIOo8?si=wPRNABX2by1lSfpX - Data Scientists 
https://youtu.be/mscyUYOF0cw?si=fRba4M_8I7H-dlE2- Data Scientists Time Series Forecasting LSTM


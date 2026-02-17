import numpy as np


class ScaledDotProductAttention:

    def __init__(self):
        self.attention_weights = None

    def softmax(self, x):
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, Q, K, V):
        if Q.shape[1] != K.shape[1]:
            raise ValueError(f"Q e K devem ter a mesma dimensão d_k. Q: {Q.shape}, K: {K.shape}")
        if K.shape[0] != V.shape[0]:
            raise ValueError(f"K e V devem ter o mesmo número de sequências. K: {K.shape}, V: {V.shape}")

        d_k = K.shape[1]
        scores = np.matmul(Q, K.T)
        scaled_scores = scores / np.sqrt(d_k)
        attention_weights = self.softmax(scaled_scores)
        self.attention_weights = attention_weights
        output = np.matmul(attention_weights, V)

        return output, attention_weights

    def __call__(self, Q, K, V):
        return self.forward(Q, K, V)


def generate_random_qkv(seq_len, d_k, d_v=None, seed=42):
    if d_v is None:
        d_v = d_k
    np.random.seed(seed)
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_v)
    return Q, K, V


if __name__ == "__main__":
    seq_len = 4
    d_k = 8
    d_v = 8

    Q, K, V = generate_random_qkv(seq_len, d_k, d_v)

    attention = ScaledDotProductAttention()
    output, weights = attention(Q, K, V)

    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"Somas das linhas: {np.sum(weights, axis=1)}")

import numpy as np
from attention import ScaledDotProductAttention, generate_random_qkv


def test_softmax_sum_to_one():
    attention = ScaledDotProductAttention()
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    result = attention.softmax(x)
    assert np.allclose(np.sum(result, axis=1), 1.0)
    print("[TESTE 1] PASSOU")


def test_attention_output_shape():
    seq_len, d_k, d_v = 5, 16, 32
    Q, K, V = generate_random_qkv(seq_len, d_k, d_v)
    attention = ScaledDotProductAttention()
    output, weights = attention(Q, K, V)
    assert output.shape == (seq_len, d_v)
    assert weights.shape == (seq_len, seq_len)
    print("[TESTE 2] PASSOU")


def test_attention_weights_properties():
    Q, K, V = generate_random_qkv(seq_len=6, d_k=10)
    attention = ScaledDotProductAttention()
    _, weights = attention(Q, K, V)
    assert np.all(weights >= 0) and np.all(weights <= 1)
    assert np.allclose(np.sum(weights, axis=1), 1.0)
    print("[TESTE 3] PASSOU")


def test_scaling_factor():
    d_k = 4
    Q = np.array([[1.0, 0.0, 0.0, 0.0]])
    K = np.array([[1.0, 0.0, 0.0, 0.0]])
    V = np.array([[1.0, 2.0, 3.0, 4.0]])
    attention = ScaledDotProductAttention()
    _, weights = attention(Q, K, V)
    assert np.isclose(weights[0, 0], 1.0)
    print("[TESTE 4] PASSOU")


def test_known_example():
    Q = np.array([[1.0, 0.0], [0.0, 1.0]])
    K = np.array([[1.0, 0.0], [0.0, 1.0]])
    V = np.array([[1.0, 0.0], [0.0, 1.0]])
    attention = ScaledDotProductAttention()
    output, weights = attention(Q, K, V)
    assert weights.shape == (2, 2)
    assert np.allclose(np.sum(weights, axis=1), 1.0)
    print("[TESTE 5] PASSOU")


def test_dimension_mismatch():
    attention = ScaledDotProductAttention()
    Q = np.random.randn(4, 8)
    K = np.random.randn(4, 16)
    V = np.random.randn(4, 8)
    try:
        _ = attention(Q, K, V)
        assert False
    except ValueError:
        print("[TESTE 6] PASSOU")


if __name__ == "__main__":
    test_softmax_sum_to_one()
    test_attention_output_shape()
    test_attention_weights_properties()
    test_scaling_factor()
    test_known_example()
    test_dimension_mismatch()
    print("\nTodos os testes passaram!")

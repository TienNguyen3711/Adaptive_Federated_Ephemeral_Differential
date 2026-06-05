import hashlib
import hmac
import json
import math
import os
import secrets
import struct
import time
from typing import List, Tuple

# ── Prime field ───────────────────────────────────────────────────────────────
# 2^521 − 1  (Mersenne prime M521, safe for secrets up to 512 bits / 64 bytes)
_PRIME = (1 << 521) - 1

# Maximum secret size this implementation supports (bytes).
SECRET_MAX_BYTES = 64   # 512-bit secrets


# ─────────────────────────────────────────────────────────────────────────────
# Finite-field arithmetic over GF(_PRIME)
# ─────────────────────────────────────────────────────────────────────────────

def _mod_inv(a: int, p: int) -> int:
    """Modular inverse via Fermat's little theorem (p is prime)."""
    return pow(a, p - 2, p)


def _eval_poly(coeffs: List[int], x: int, p: int) -> int:
    """Evaluate polynomial at x mod p using Horner's method."""
    result = 0
    for c in reversed(coeffs):
        result = (result * x + c) % p
    return result


def _lagrange_at_zero(shares: List[Tuple[int, int]], p: int) -> int:
    """
    Lagrange interpolation: recover f(0) from t (x, y) shares over GF(p).
    """
    xs, ys = zip(*shares)
    result = 0
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        num = yi
        den = 1
        for j, xj in enumerate(xs):
            if i != j:
                num = num * (0 - xj) % p
                den = den * (xi - xj) % p
        result = (result + num * _mod_inv(den, p)) % p
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Core Shamir functions
# ─────────────────────────────────────────────────────────────────────────────

def split_secret(
    secret_bytes: bytes,
    t: int,
    n: int,
    prime: int = _PRIME,
) -> List[Tuple[int, int]]:
    if not 1 <= t <= n:
        raise ValueError(f"Must have 1 ≤ t ≤ n, got t={t}, n={n}")
    if len(secret_bytes) > SECRET_MAX_BYTES:
        raise ValueError(
            f"Secret too long: {len(secret_bytes)} bytes > {SECRET_MAX_BYTES}"
        )
    if len(secret_bytes) == 0:
        raise ValueError("Secret must be non-empty")

    secret_int = int.from_bytes(secret_bytes, "big")
    if secret_int >= prime:
        raise ValueError("Secret value exceeds prime modulus — reduce secret size")

    # Build random polynomial f(x) = secret + a_1·x + … + a_{t-1}·x^{t-1} mod p
    coeffs = [secret_int] + [
        secrets.randbelow(prime) for _ in range(t - 1)
    ]

    shares = [(i, _eval_poly(coeffs, i, prime)) for i in range(1, n + 1)]
    return shares


def reconstruct_secret(
    shares: List[Tuple[int, int]],
    prime: int = _PRIME,
) -> bytes:

    if not shares:
        raise ValueError("At least one share is required")

    secret_int = _lagrange_at_zero(shares, prime)

    # Determine minimum byte length needed
    byte_len = max(1, math.ceil(secret_int.bit_length() / 8))
    return secret_int.to_bytes(byte_len, "big")


# ─────────────────────────────────────────────────────────────────────────────
# Share serialisation
# ─────────────────────────────────────────────────────────────────────────────

def serialise_share(index: int, value: int) -> bytes:
    idx_bytes = struct.pack(">H", index)
    val_bytes = value.to_bytes(66, "big")
    return idx_bytes + val_bytes


def deserialise_share(data: bytes) -> Tuple[int, int]:
    if len(data) != 68:
        raise ValueError(f"Expected 68-byte share, got {len(data)} bytes")
    index = struct.unpack(">H", data[:2])[0]
    value = int.from_bytes(data[2:], "big")
    return (index, value)


# ─────────────────────────────────────────────────────────────────────────────
# DKAKeyManager — high-level key management for AFED-PPTE Layer 6
# ─────────────────────────────────────────────────────────────────────────────

class DKAKeyManager:
    PBKDF2_ITERATIONS = 600_000
    KEY_LENGTH = 32  # 256-bit AES key

    def __init__(self, k_store: bytes):
        self._k_store = k_store

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def initialise(cls, t: int = 3, n: int = 5, secret_size: int = 32):
        k_store = secrets.token_bytes(secret_size)
        obj = cls(k_store)
        obj.shares = split_secret(k_store, t, n)
        obj.t = t
        obj.n = n
        return obj

    @classmethod
    def from_shares(cls, shares: List[Tuple[int, int]]):
        """
        Reconstruct K_store from t collected shares.
        """
        k_store = reconstruct_secret(shares)
        return cls(k_store)

    @classmethod
    def from_env(cls):
        secret_str = os.getenv("AFED_SECRET_V1") or os.getenv("AFED_SECRET")
        if not secret_str:
            raise EnvironmentError(
                "DKAKeyManager.from_env() requires AFED_SECRET_V1 "
                "or AFED_SECRET in environment"
            )
        return cls(secret_str.encode("utf-8"))

    # ── Key derivation ────────────────────────────────────────────────────────

    def derive_encryption_key(self, user_id: str, salt: bytes) -> bytes:
        password = self._k_store + user_id.encode("utf-8")
        dk = hashlib.pbkdf2_hmac(
            "sha256",
            password,
            salt,
            self.PBKDF2_ITERATIONS,
            dklen=self.KEY_LENGTH,
        )
        return dk

    def wipe(self):
        if isinstance(self._k_store, bytearray):
            for i in range(len(self._k_store)):
                self._k_store[i] = 0
        self._k_store = b""


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark helper (for exp_dka.py)
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_reconstruction(
    t: int,
    n: int,
    secret_size: int = 32,
    repetitions: int = 100,
) -> dict:
    secret = secrets.token_bytes(secret_size)

    # ── split timing ─────────────────────────────────────────────────────────
    split_times = []
    for _ in range(repetitions):
        t0 = time.perf_counter()
        shares = split_secret(secret, t, n)
        split_times.append((time.perf_counter() - t0) * 1000)

    # ── serialise / deserialise timing ───────────────────────────────────────
    ser_times = []
    deser_times = []
    all_shares = split_secret(secret, t, n)
    for _ in range(repetitions):
        t0 = time.perf_counter()
        blobs = [serialise_share(idx, val) for idx, val in all_shares[:t]]
        ser_times.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        recovered_shares = [deserialise_share(b) for b in blobs]
        deser_times.append((time.perf_counter() - t0) * 1000)

    # ── reconstruct timing ────────────────────────────────────────────────────
    recon_times = []
    for _ in range(repetitions):
        shares_t = split_secret(secret, t, n)[:t]
        t0 = time.perf_counter()
        reconstruct_secret(shares_t)
        recon_times.append((time.perf_counter() - t0) * 1000)

    # ── PBKDF2 timing (one key derivation) ───────────────────────────────────
    pbkdf2_times = []
    salt = secrets.token_bytes(32)
    for _ in range(repetitions):
        t0 = time.perf_counter()
        hashlib.pbkdf2_hmac("sha256", secret, salt, 600_000, dklen=32)
        pbkdf2_times.append((time.perf_counter() - t0) * 1000)

    def _stats(lst):
        return {
            "mean_ms": round(sum(lst) / len(lst), 3),
            "min_ms":  round(min(lst), 3),
            "max_ms":  round(max(lst), 3),
        }

    return {
        "t": t, "n": n, "secret_size_bytes": secret_size, "repetitions": repetitions,
        "split":       _stats(split_times),
        "reconstruct": _stats(recon_times),
        "serialise":   _stats(ser_times),
        "deserialise": _stats(deser_times),
        "pbkdf2_baseline": _stats(pbkdf2_times),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Shamir (3,5) correctness test ===")
    secret = b"EXAMPLE_MASTER_KEY_32BYTES_ONLY!"   # 32 bytes
    shares = split_secret(secret, t=3, n=5)
    print(f"Generated {len(shares)} shares")

    # Reconstruct from shares 1, 3, 5
    chosen = [shares[0], shares[2], shares[4]]
    recovered = reconstruct_secret(chosen)
    print(f"Secret match: {recovered == secret}")

    # Verify t−1=2 shares reveal nothing about the correct secret
    two_shares = [shares[0], shares[1]]
    partial = reconstruct_secret(two_shares)
    print(f"2-share reconstruction ≠ secret: {partial != secret}")

    # Test serialise round-trip
    blob = serialise_share(*shares[0])
    idx2, val2 = deserialise_share(blob)
    print(f"Serialise round-trip OK: {(idx2, val2) == shares[0]}")

    # DKAKeyManager round-trip
    mgr = DKAKeyManager.initialise(t=3, n=5)
    chosen_shares = mgr.shares[:3]
    mgr2 = DKAKeyManager.from_shares(chosen_shares)
    salt = secrets.token_bytes(32)
    key1 = mgr.derive_encryption_key("user_42", salt)
    key2 = mgr2.derive_encryption_key("user_42", salt)
    print(f"Key derivation match: {key1 == key2}")

    print("\n=== Quick benchmark (t=3, n=5, 10 reps) ===")
    bm = benchmark_reconstruction(3, 5, repetitions=10)
    print(f"  Split:       {bm['split']['mean_ms']:.2f} ms")
    print(f"  Reconstruct: {bm['reconstruct']['mean_ms']:.2f} ms")
    print(f"  PBKDF2:      {bm['pbkdf2_baseline']['mean_ms']:.2f} ms")

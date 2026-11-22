"""Download the local model (free, CPU-only, ~390 MB quantized)."""
from gce.model import ensure_model

if __name__ == "__main__":
    path = ensure_model()
    print("model ready at:", path)

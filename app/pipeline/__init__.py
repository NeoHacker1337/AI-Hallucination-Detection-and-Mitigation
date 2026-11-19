"""Pipeline stages. Heavy dependencies (torch, transformers, faiss) are imported lazily, so the
pure logic modules (claims, semantic entropy maths, decision) can be used and tested without them."""

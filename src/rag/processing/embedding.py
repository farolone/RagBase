from dataclasses import dataclass

from FlagEmbedding import BGEM3FlagModel

from rag.config import settings


@dataclass
class EmbeddingResult:
    dense: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]


class Embedder:
    def __init__(self):
        self.model = BGEM3FlagModel(
            settings.embedding_model, use_fp16=False
        )

    def embed(self, text: str) -> EmbeddingResult:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        output = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
        )
        results = []
        for i in range(len(texts)):
            dense = output["dense_vecs"][i].tolist()
            sparse = output["lexical_weights"][i]
            indices = [int(k) for k in sparse.keys()]
            values = [float(v) for v in sparse.values()]
            results.append(
                EmbeddingResult(
                    dense=dense,
                    sparse_indices=indices,
                    sparse_values=values,
                )
            )
        return results

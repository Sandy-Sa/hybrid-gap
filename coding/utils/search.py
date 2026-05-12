import torch
import torch.nn.functional as F
import Stemmer
import bm25s
import polars as pl
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

from coding.utils.umls import load_umls_synonyms, expand_corpus


class CodeSearcher:
    def __init__(
        self,
        df_icd: pl.DataFrame,
        model_name: str = "FremyCompany/BioLORD-2023-C",
        top_k: int = 5,
        mrconso_path: str | Path | None = None,
        icd_sab: str = "ICD9CM",
        max_synonyms: int = 10,
        batch_size: int | None = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_dtype = (
            torch.float16 if self.device.type == "cuda" else torch.float32
        )
        self._batch_size = batch_size or (256 if self.device.type == "cuda" else 64)
        self.df_icd = df_icd
        self.top_k = top_k

        print(f"Loading {model_name} to {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Build corpus — optionally expanded with UMLS synonyms
        base_corpus = self.df_icd["LONG_TITLE"].to_list()
        if mrconso_path is not None:
            print(f"Loading UMLS synonyms from {mrconso_path}...")
            synonyms = load_umls_synonyms(mrconso_path, icd_sab=icd_sab)
            self.corpus, self.corpus_to_icd_idx = expand_corpus(
                df_icd, synonyms, max_synonyms=max_synonyms
            )
            n_synonyms = len(self.corpus) - len(self.df_icd)
            print(
                f"  Expanded corpus: {len(self.corpus)} entries "
                f"({len(self.df_icd)} titles + {n_synonyms} synonyms)"
            )
        else:
            self.corpus = base_corpus
            self.corpus_to_icd_idx = list(range(len(self.df_icd)))

        print("Initializing BM25 index...")
        self.stemmer = Stemmer.Stemmer("english")
        corpus_tokens = [self._tokenize_and_stem(text) for text in self.corpus]
        self.bm25 = bm25s.BM25()
        self.bm25.index(corpus_tokens)

        print("Pre-computing dictionary embeddings...")
        self.icd_embeddings = self._generate_embeddings(self.corpus)
        self.icd_embeddings = F.normalize(self.icd_embeddings, p=2, dim=1)
        self.icd_embeddings = self.icd_embeddings.to(self.compute_dtype)

    def _tokenize_and_stem(self, text):
        tokens = text.lower().split()
        return self.stemmer.stemWords(tokens)

    def _generate_embeddings(self, texts):
        all_embeddings = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=128,
            ).to(self.device)
            with (
                torch.no_grad(),
                torch.amp.autocast(
                    device_type=self.device.type, enabled=self.device.type == "cuda"
                ),
            ):
                outputs = self.model(**inputs)
                # CLS token pooling
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                all_embeddings.append(cls_embeddings)
        return torch.cat(all_embeddings, dim=0)

    def search(self, keyword: str, top_k: int | None = None) -> pl.DataFrame:
        """
        Hybrid search using BM25 + dense retrieval fused with
        Reciprocal Rank Fusion (RRF).

        Returns a DataFrame with columns:
            ICD9_CODE, LONG_TITLE, score (RRF), bm25_rank, dense_rank, method
        """
        k = top_k or self.top_k
        retrieval_k = min(k * 4, len(self.corpus))

        # --- BM25 retrieval ---
        query_tokens = self._tokenize_and_stem(keyword)
        bm25_results, bm25_scores = self.bm25.retrieve([query_tokens], k=retrieval_k)
        bm25_indices = [int(i) for i in bm25_results[0]]

        # --- Dense retrieval ---
        query_emb = self._generate_embeddings([keyword]).to(self.icd_embeddings.dtype)
        cos_sim = torch.nn.functional.cosine_similarity(query_emb, self.icd_embeddings)
        dense_vals, dense_idx = torch.topk(cos_sim, k=retrieval_k)
        dense_indices = dense_idx.tolist()

        # --- RRF fusion (over corpus indices) ---
        rrf_k = 60
        rrf_scores: dict[int, float] = {}
        bm25_rank_map: dict[int, int] = {}
        dense_rank_map: dict[int, int] = {}

        for rank, idx in enumerate(bm25_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
            bm25_rank_map[idx] = rank + 1

        for rank, idx in enumerate(dense_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
            dense_rank_map[idx] = rank + 1

        # --- Group by ICD code: keep best score per code ---
        code_best: dict[int, tuple[float, int]] = {}  # icd_idx -> (score, corpus_idx)
        for corpus_idx, score in rrf_scores.items():
            icd_idx = self.corpus_to_icd_idx[corpus_idx]
            if icd_idx not in code_best or score > code_best[icd_idx][0]:
                code_best[icd_idx] = (score, corpus_idx)

        sorted_results = sorted(code_best.items(), key=lambda x: x[1][0], reverse=True)[
            :k
        ]

        if not sorted_results:
            return pl.DataFrame(
                schema={
                    "ICD9_CODE": pl.Utf8,
                    "LONG_TITLE": pl.Utf8,
                    "score": pl.Float64,
                    "bm25_rank": pl.Int64,
                    "dense_rank": pl.Int64,
                    "method": pl.Utf8,
                }
            )

        result_indices = [icd_idx for icd_idx, _ in sorted_results]
        result_scores = [score for _, (score, _) in sorted_results]
        best_corpus_indices = [cidx for _, (_, cidx) in sorted_results]

        result_df = pl.DataFrame(
            {
                "index": result_indices,
                "score": result_scores,
                "bm25_rank": [bm25_rank_map.get(cidx) for cidx in best_corpus_indices],
                "dense_rank": [
                    dense_rank_map.get(cidx) for cidx in best_corpus_indices
                ],
            }
        ).with_columns(
            pl.col("index").cast(pl.Int64),
            pl.lit("hybrid").alias("method"),
        )

        hits = (
            self.df_icd.with_row_index()
            .join(result_df, on="index", how="inner")
            .sort("score", descending=True)
        )

        return hits


def search_terms(
    terms: list[str],
    searcher: "CodeSearcher",
    top_k: int = 5,
    min_score: float = 0.0,
) -> dict:
    """Search a list of terms and return top-k codes and scores for each.

    Args:
        terms: List of normalized clinical terms to search.
        searcher: CodeSearcher instance for the relevant code type.
        top_k: Maximum number of results per term.
        min_score: Minimum RRF score threshold. Results below this are dropped.

    Returns:
        dict with keys "codes" and "scores", each a list-of-lists aligned with terms.
    """
    codes: list[list[str]] = []
    scores: list[list[float]] = []
    if terms is None:
        return {"codes": codes, "scores": scores}
    for term in terms:
        results = searcher.search(term, top_k=top_k)
        if min_score > 0.0:
            results = results.filter(pl.col("score") >= min_score)
        codes.append(results["ICD9_CODE"].to_list())
        scores.append(results["score"].to_list())
    return {"codes": codes, "scores": scores}

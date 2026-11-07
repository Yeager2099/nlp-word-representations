"""
HW04: Word Embeddings
CIS 5300: Natural Language Processing
Name: Yeager Ye
Penn ID: [STUDENT_ID_REDACTED]
"""

import numpy as np
import os
import math
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

# ============================================================
# Section 0: Setup helpers (DO NOT MODIFY)
# ============================================================

SHAKESPEARE_DIR = 'data/shakespeare'
SIMLEX_PATH = 'data/simlex999.txt'
CLUSTER_DEV_PATH = 'data/clustering/dev.txt'
CLUSTER_TEST_PATH = 'data/clustering/test.txt'
CLUSTER_COOCCURRENCE_DIR = 'data/clustering/cooccurrence'


def load_cluster_data(filepath):
    """Load paraphrase clustering data.

    Returns a list of dicts, each with:
        'target': the target word (string)
        'k': number of gold-standard senses (int)
        'paraphrases': list of paraphrase strings
        'labels': list of gold-standard cluster labels (ints)
    """
    data = []
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            target = parts[0]
            k = int(parts[1])
            paraphrases = []
            labels = []
            for item in parts[2:]:
                # Format: "paraphrase::label"
                pp, label = item.rsplit('::', 1)
                paraphrases.append(pp)
                labels.append(int(label))
            data.append({
                'target': target,
                'k': k,
                'paraphrases': paraphrases,
                'labels': labels
            })
    return data


def load_simlex(filepath, vocab_set=None, embeddings=None, max_pairs=None):
    """Load SimLex-999 word pairs."""
    pairs = []
    with open(filepath, encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            w1, w2 = parts[0].lower(), parts[1].lower()
            score = float(parts[3])
            in_sparse = (vocab_set is None) or (w1 in vocab_set and w2 in vocab_set)
            in_dense = (embeddings is None) or (w1 in embeddings and w2 in embeddings)
            if in_sparse and in_dense:
                pairs.append((w1, w2, score))
            if max_pairs and len(pairs) >= max_pairs:
                break
    return pairs


def load_cooccurrence_vectors(paraphrases, cooc_dir):
    """Load pre-computed co-occurrence vectors for paraphrases."""
    vectors = []
    for pp in paraphrases:
        fname = pp.replace(' ', '_') + '.txt'
        fpath = os.path.join(cooc_dir, fname)
        if os.path.exists(fpath):
            vec = np.loadtxt(fpath)
        else:
            vec = np.zeros(500)
        vectors.append(vec)
    return np.array(vectors)


def get_embedding_vectors(paraphrases, embeddings):
    """Get dense embedding vectors for paraphrases."""
    vectors = []
    dim = embeddings.vector_size
    for pp in paraphrases:
        words = pp.split()
        word_vecs = [embeddings[w] for w in words if w in embeddings]
        if word_vecs:
            vectors.append(np.mean(word_vecs, axis=0))
        else:
            vectors.append(np.zeros(dim))
    return np.array(vectors)


def paired_fscore(predicted_labels, gold_labels):
    """Compute paired F-score for clustering evaluation."""
    n = len(predicted_labels)
    assert len(gold_labels) == n

    tp = fp = fn = 0
    for i in range(n):
        for j in range(i + 1, n):
            same_pred = (predicted_labels[i] == predicted_labels[j])
            same_gold = (gold_labels[i] == gold_labels[j])
            if same_pred and same_gold:
                tp += 1
            elif same_pred and not same_gold:
                fp += 1
            elif not same_pred and same_gold:
                fn += 1

    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ============================================================
# Section 1.1: cosine_similarity
# ============================================================

def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors.

    Returns 0.0 if either vector is the zero vector.

    Args:
        v1: numpy array.
        v2: numpy array.

    Returns:
        Cosine similarity (float).
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


# ============================================================
# Section 1.2: build_term_document_matrix
# ============================================================

def build_term_document_matrix(plays, vocab, word_to_idx, play_names):
    """Build a term-document matrix from Shakespeare plays.

    Args:
        plays: Dict mapping play names to word lists.
        vocab: Sorted list of vocabulary words.
        word_to_idx: Dict mapping word to its index in vocab.
        play_names: Sorted list of play names.

    Returns:
        A numpy array of shape (len(vocab), len(play_names)).
        Entry [i, j] = number of times vocab[i] appears in play_names[j].
    """
    matrix = np.zeros((len(vocab), len(play_names)), dtype=np.float64)
    for j, play_name in enumerate(play_names):
        word_list = plays[play_name]
        counts = Counter(word_list)
        for word, count in counts.items():
            if word in word_to_idx:
                i = word_to_idx[word]
                matrix[i, j] = count
    return matrix


# ============================================================
# Section 1.3: rank_plays_by_similarity
# ============================================================

def rank_plays_by_similarity(td_matrix, target_play, play_names):
    """Rank all plays by cosine similarity to a target play.

    Use the COLUMN vectors of the term-document matrix as play representations.

    Args:
        td_matrix: numpy array of shape (|V|, |D|).
        target_play: Name of the play to compare against (string).
        play_names: Sorted list of play names.

    Returns:
        A list of (play_name, similarity) tuples, sorted by similarity
        in descending order. Excludes the target play itself.
    """
    target_idx = play_names.index(target_play)
    target_vec = td_matrix[:, target_idx]
    results = []
    for j, play_name in enumerate(play_names):
        if play_name == target_play:
            continue
        sim = cosine_similarity(target_vec, td_matrix[:, j])
        results.append((play_name, sim))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ============================================================
# Section 1.4: build_cooccurrence_matrix
# ============================================================

def build_cooccurrence_matrix(all_words, vocab, word_to_idx, window=2):
    """Build a word co-occurrence matrix.

    Args:
        all_words: List of all words in the corpus (in order).
        vocab: Sorted list of vocabulary words.
        word_to_idx: Dict mapping word to index in vocab.
        window: Context window size (number of words on each side).

    Returns:
        A numpy array of shape (len(vocab), len(vocab)).
        Entry [i, j] = number of times vocab[j] appears within
        a window of 'window' words around vocab[i] in the corpus.
    """
    matrix = np.zeros((len(vocab), len(vocab)), dtype=np.float64)
    n = len(all_words)
    for pos, word in enumerate(all_words):
        if word not in word_to_idx:
            continue
        word_idx = word_to_idx[word]
        # Context window: [pos-window, pos+window] excluding pos itself
        start = max(0, pos - window)
        end = min(n - 1, pos + window)
        for ctx_pos in range(start, end + 1):
            if ctx_pos == pos:
                continue
            ctx_word = all_words[ctx_pos]
            if ctx_word in word_to_idx:
                ctx_idx = word_to_idx[ctx_word]
                matrix[word_idx, ctx_idx] += 1
    return matrix


# ============================================================
# Section 1.5: most_similar_cooc
# ============================================================

def most_similar_cooc(word, cooc_matrix, vocab, word_to_idx, top_n=10):
    """Find the most similar words using co-occurrence vectors.

    Args:
        word: The query word (string).
        cooc_matrix: Co-occurrence matrix of shape (|V|, |V|).
        vocab: Sorted list of vocabulary words.
        word_to_idx: Dict mapping word to index.
        top_n: Number of results to return.

    Returns:
        A list of (word, similarity) tuples, sorted by similarity descending.
        Excludes the query word itself.
    """
    if word not in word_to_idx:
        return []
    word_idx = word_to_idx[word]
    word_vec = cooc_matrix[word_idx]
    results = []
    for i, w in enumerate(vocab):
        if w == word:
            continue
        sim = cosine_similarity(word_vec, cooc_matrix[i])
        results.append((w, sim))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]


# ============================================================
# Section 1.6: compute_tfidf
# ============================================================

def compute_tfidf(td_matrix):
    """Compute TF-IDF weighting of a term-document matrix.

    Args:
        td_matrix: numpy array of shape (|V|, |D|) with raw term frequencies.

    Returns:
        A numpy array of the same shape with TF-IDF weighted values.
        Use log base 10 for the IDF computation.
    """
    # TF: raw term frequency (already in td_matrix)
    tf = td_matrix.copy().astype(np.float64)

    # IDF: log10(N / df) where df = number of documents containing term
    N = td_matrix.shape[1]  # number of documents
    # df[i] = number of documents where term i appears
    df = (td_matrix > 0).sum(axis=1).astype(np.float64)
    # Avoid division by zero for words appearing in 0 documents
    # (shouldn't happen in practice if vocab is built from the data)
    # Words appearing in ALL documents: IDF = log10(N/N) = 0
    with np.errstate(divide='ignore', invalid='ignore'):
        idf = np.where(df > 0, np.log10(N / df), 0.0)

    # TF-IDF = TF * IDF (broadcast idf column vector across documents)
    tfidf = tf * idf[:, np.newaxis]
    return tfidf


# ============================================================
# Section 1.7: compute_ppmi
# ============================================================

def compute_ppmi(cooc_matrix):
    """Compute PPMI weighting of a co-occurrence matrix.

    For each cell (w, c):
        PMI(w, c) = log2(P(w,c) / (P(w) * P(c)))
        PPMI(w, c) = max(0, PMI(w, c))

    When the co-occurrence count is 0, set PMI = 0 (which gives PPMI = 0).
    Use log base 2.

    Args:
        cooc_matrix: numpy array of shape (|V|, |V|) with co-occurrence counts.

    Returns:
        A numpy array of the same shape with PPMI values.
    """
    total = cooc_matrix.sum()
    if total == 0:
        return np.zeros_like(cooc_matrix, dtype=np.float64)

    # P(w, c) for each cell
    p_wc = cooc_matrix / total

    # P(w) = marginal row probability
    p_w = cooc_matrix.sum(axis=1) / total  # shape: (|V|,)

    # P(c) = marginal column probability
    p_c = cooc_matrix.sum(axis=0) / total  # shape: (|V|,)

    # PMI(w, c) = log2(P(w,c) / (P(w) * P(c)))
    # outer product of p_w and p_c gives denominator
    denom = np.outer(p_w, p_c)  # shape: (|V|, |V|)

    # Only compute PMI where both p_wc > 0 and denom > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        pmi = np.where(
            (p_wc > 0) & (denom > 0),
            np.log2(np.where(denom > 0, p_wc / denom, 1.0)),
            0.0
        )

    # PPMI = max(0, PMI)
    ppmi = np.maximum(0.0, pmi)
    return ppmi


# ============================================================
# Section 2.2: solve_analogy
# ============================================================

def solve_analogy(a, a_star, b, embeddings, top_n=5):
    """Solve the analogy: a is to a_star as b is to ???

    Computes b* = argmax cos(w, b - a + a*) for all w in vocabulary,
    excluding a, a*, and b themselves.

    Do NOT use gensim's most_similar() -- implement from scratch using
    the embeddings vectors and your cosine_similarity function.

    Args:
        a: First word of the analogy (string).
        a_star: Second word (string). a is to a_star.
        b: Third word (string). b is to ???.
        embeddings: gensim KeyedVectors object.
        top_n: Number of candidate answers to return.

    Returns:
        A list of (word, similarity) tuples, sorted by similarity descending.
    """
    # Check all input words are in vocabulary
    for word in [a, a_star, b]:
        if word not in embeddings:
            return []

    # Target vector: b - a + a*
    target_vec = embeddings[b] - embeddings[a] + embeddings[a_star]
    target_vec = target_vec.astype(np.float64)

    exclude = {a.lower(), a_star.lower(), b.lower()}
    results = []

    for word in embeddings.index_to_key:
        if word.lower() in exclude:
            continue
        sim = cosine_similarity(target_vec, embeddings[word].astype(np.float64))
        results.append((word, sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]


# ============================================================
# Section 2.3: evaluate_similarity (Kendall's Tau)
# ============================================================

def evaluate_similarity(pairs, method, **kwargs):
    """Evaluate word similarity against human judgments using Kendall's Tau.

    Args:
        pairs: List of (word1, word2, human_score) tuples.
        method: One of 'ppmi', 'dense'.
        **kwargs: Passed to compute_similarity_scores.

    Returns:
        Kendall's Tau correlation (float).
    """
    human_scores = [p[2] for p in pairs]
    model_scores = compute_similarity_scores(pairs, method, **kwargs)
    tau, _ = kendalltau(human_scores, model_scores)
    return tau


def compute_similarity_scores(pairs, method, **kwargs):
    """Compute similarity scores for a list of word pairs."""
    scores = []
    for w1, w2, _ in pairs:
        if method == 'ppmi':
            ppmi_mat = kwargs['ppmi_matrix']
            w2i = kwargs['word_to_idx']
            v1 = ppmi_mat[w2i[w1]]
            v2 = ppmi_mat[w2i[w2]]
            scores.append(cosine_similarity(v1, v2))
        elif method == 'dense':
            emb = kwargs['embeddings']
            v1 = emb[w1]
            v2 = emb[w2]
            scores.append(cosine_similarity(v1, v2))
        else:
            raise ValueError(f"Unknown method: {method}")
    return scores


# ============================================================
# Section 2.4: measure_gender_bias
# ============================================================

def measure_gender_bias(occupation_words, embeddings, male_words=None, female_words=None):
    """Measure gender bias for a list of occupation words.

    For each occupation word, compute:
        bias = cos(occupation, she_vec) - cos(occupation, he_vec)

    where she_vec and he_vec are the averages of the female and male
    word vectors respectively.

    Positive bias = associated more with female
    Negative bias = associated more with male

    Args:
        occupation_words: List of occupation strings.
        embeddings: gensim KeyedVectors object.
        male_words: List of male-associated words.
        female_words: List of female-associated words.

    Returns:
        A list of (occupation, bias_score) tuples, sorted by bias_score.
    """
    if male_words is None:
        male_words = ['he', 'man', 'boy', 'father', 'son']
    if female_words is None:
        female_words = ['she', 'woman', 'girl', 'mother', 'daughter']

    # Build average male and female vectors
    male_vecs = [embeddings[w] for w in male_words if w in embeddings]
    female_vecs = [embeddings[w] for w in female_words if w in embeddings]

    he_vec = np.mean(male_vecs, axis=0).astype(np.float64)
    she_vec = np.mean(female_vecs, axis=0).astype(np.float64)

    results = []
    for occ in occupation_words:
        if occ not in embeddings:
            continue
        occ_vec = embeddings[occ].astype(np.float64)
        bias = cosine_similarity(occ_vec, she_vec) - cosine_similarity(occ_vec, he_vec)
        results.append((occ, bias))

    results.sort(key=lambda x: x[1])
    return results


# ============================================================
# Section 3.1: cluster_sparse
# ============================================================

def cluster_sparse(data, cooc_dir):
    """Cluster paraphrases using sparse co-occurrence vectors.

    For each target word entry in data:
    1. Load co-occurrence vectors for the paraphrases
    2. Run K-means with K = entry['k'] (gold-standard number of senses)
    3. Compute paired F-score against gold labels

    Use random_state=42 in KMeans for reproducibility.

    Args:
        data: List of dicts from load_cluster_data().
        cooc_dir: Path to co-occurrence vector directory.

    Returns:
        A list of dicts, each with:
            'target': target word
            'k': number of clusters
            'fscore': paired F-score
            'predicted_labels': list of predicted cluster assignments
    """
    results = []
    for entry in data:
        target = entry['target']
        k = entry['k']
        paraphrases = entry['paraphrases']
        gold_labels = entry['labels']

        # Load co-occurrence vectors
        vectors = load_cooccurrence_vectors(paraphrases, cooc_dir)

        # Handle edge case: k >= number of samples
        n_samples = len(paraphrases)
        actual_k = min(k, n_samples)

        if n_samples < 2 or actual_k < 2:
            # Can't cluster - assign all to same cluster
            predicted_labels = [0] * n_samples
        else:
            kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init=10)
            predicted_labels = kmeans.fit_predict(vectors).tolist()

        fscore = paired_fscore(predicted_labels, gold_labels)
        results.append({
            'target': target,
            'k': k,
            'fscore': fscore,
            'predicted_labels': predicted_labels
        })
    return results


# ============================================================
# Section 3.2: cluster_dense
# ============================================================

def cluster_dense(data, embeddings):
    """Cluster paraphrases using dense embedding vectors.

    For each target word entry in data:
    1. Get embedding vectors for the paraphrases (using get_embedding_vectors)
    2. Run K-means with K = entry['k'] (gold-standard number of senses)
    3. Compute paired F-score against gold labels

    Use random_state=42 in KMeans for reproducibility.

    Args:
        data: List of dicts from load_cluster_data().
        embeddings: gensim KeyedVectors object.

    Returns:
        Same format as cluster_sparse.
    """
    results = []
    for entry in data:
        target = entry['target']
        k = entry['k']
        paraphrases = entry['paraphrases']
        gold_labels = entry['labels']

        # Get dense embedding vectors
        vectors = get_embedding_vectors(paraphrases, embeddings)

        n_samples = len(paraphrases)
        actual_k = min(k, n_samples)

        if n_samples < 2 or actual_k < 2:
            predicted_labels = [0] * n_samples
        else:
            kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init=10)
            predicted_labels = kmeans.fit_predict(vectors).tolist()

        fscore = paired_fscore(predicted_labels, gold_labels)
        results.append({
            'target': target,
            'k': k,
            'fscore': fscore,
            'predicted_labels': predicted_labels
        })
    return results


# ============================================================
# Section 3.3: cluster_auto_k
# ============================================================

def cluster_auto_k(data, embeddings):
    """Cluster paraphrases with automatic K selection using silhouette score.

    For each target word:
    1. Get embedding vectors for the paraphrases
    2. For K in range(2, min(n_paraphrases - 1, 10) + 1):
        a. Run K-means with this K (random_state=42)
        b. Compute silhouette score
    3. Select the K with the highest silhouette score
    4. Compute paired F-score with gold labels

    Args:
        data: List of dicts from load_cluster_data().
        embeddings: gensim KeyedVectors object.

    Returns:
        A list of dicts, each with:
            'target': target word
            'gold_k': gold-standard K
            'predicted_k': automatically selected K
            'fscore': paired F-score
            'predicted_labels': list of predicted cluster assignments
    """
    results = []
    for entry in data:
        target = entry['target']
        gold_k = entry['k']
        paraphrases = entry['paraphrases']
        gold_labels = entry['labels']

        vectors = get_embedding_vectors(paraphrases, embeddings)
        n_samples = len(paraphrases)

        max_k = min(n_samples - 1, 10)

        if n_samples < 3 or max_k < 2:
            # Can't do silhouette with fewer than 3 samples or k_range < 2
            predicted_labels = [0] * n_samples
            predicted_k = 1
        else:
            best_k = 2
            best_silhouette = -1.0
            best_labels = None

            for k in range(2, max_k + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(vectors)
                # Need at least 2 clusters actually present for silhouette
                if len(set(labels)) < 2:
                    continue
                sil_score = silhouette_score(vectors, labels)
                if sil_score > best_silhouette:
                    best_silhouette = sil_score
                    best_k = k
                    best_labels = labels

            if best_labels is None:
                best_labels = np.zeros(n_samples, dtype=int)
                best_k = 1

            predicted_labels = best_labels.tolist()
            predicted_k = best_k

        fscore = paired_fscore(predicted_labels, gold_labels)
        results.append({
            'target': target,
            'gold_k': gold_k,
            'predicted_k': predicted_k,
            'fscore': fscore,
            'predicted_labels': predicted_labels
        })
    return results

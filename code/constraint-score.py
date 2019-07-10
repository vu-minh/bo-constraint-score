""" Calculate scores for the pairwise constraints """

import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


# note with tsne q_ij is very small (e-7),
# event smaller than `np.finfo(np.float32).eps` (e-6)
# so we  have to use `np.finfo(np.double).eps` (e-16)
MACHINE_EPSILON = np.finfo(np.double).eps


def calculate_Q(Z, degrees_of_freedom: float=1.0):
    """Calculate matrix Q for the embedding Z as in t-SNE
    q_ij: probability that a point i being neighbor of a point j

    $$
    q_{ij} = \frac{ ( 1 + || y_i - y_j ||^2 )^{-1} }
                  { \sum_{k \neq l} (1 + || y_k - y_l ||^2  )^{-1} }
    $$

    Args:
        Z: embedding [N, 2].
        degrees_of_freedom (ν): defaults to 1.0 as in t-SNE
            but it can be any float value [1].

    Returns:
        Q: square matrix [N, N] for pairwise q_ij

    References:
    [1] Heavy-tailed kernels reveal a finer cluster structure in t-SNE visualisations,
    Kobab et al. 2019, avXir:1902.05804.
        + ν → ∞: corresponding to SNE
        + ν = 1: corresponding to the standard t-SNE
        + ν < 1: can further reduce the crowding problem and reveal finer
            cluster structure that is invisible in standard t-SNE.
    The general t-distribution with ν degrees of freedom:
        $$ pdf(x) =  \frac{1}{(1 + \frac{x^2}{\nu})^{\frac{(\nu+1)}{2}}} $$
    """
    power = - (degrees_of_freedom + 1.0) / 2.0
    dist = pdist(Z, "sqeuclidean")
    dist /= degrees_of_freedom
    dist += 1.0
    dist **= power  # eq. pdf(x)
    Q = dist / (2.0 * np.sum(dist))  # eq. qij
    Q = np.maximum(Q, MACHINE_EPSILON)  # for numerical stability
    return squareform(Q)

    # sk-learn implementation: https://github.com/scikit-learn/scikit-learn/blob/7813f7efb/sklearn/manifold/t_sne.py#L481
    # Q is a heavy-tailed distribution: Student's t-distribution
    # dist = pdist(Z, "sqeuclidean")
    # dist /= degrees_of_freedom
    # dist += 1.
    # dist **= (degrees_of_freedom + 1.0) / -2.0
    # Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)
    # return squareform(Q)


def _qij_based_scores(Q, links, normalized: bool=False, link_type: str=""):
    """Vectorized code for fast calculate the score.
    $$
    S_{\mathcal{M}} = \frac{1}{|\mathcal{M}|}
                      \sum_{(i,j) \in \mathcal{M}} \log q_{ij}
    \qquad
    S_{\mathcal{C}} =-\frac{1}{|\mathcal{C}|}
                      \sum_{(i,j) \in \mathcal{C}} \log q_{ij}
    $$
    Update 20190702: normalize the score in range [0, 1] for easily viz.
    """
    if len(links) == 0:
        return np.array([0.0])
    factor = {"sim": 1.0, "dis": -1.0}[link_type]
    scores = factor * np.log(Q[links[:, 0], links[:, 1]])
    return normalize_scores(scores) if normalized else scores


def qij_based_scores(Q, sim, dis, normalized: bool=False):
    """Calculate the constraint score based on q_ij.

    Args:
        Q: [q_ij] matrix of size [N, N]
        sim: array of similar links (Must links M) of form [[id1, id2]].
        dis: array of dissimilar links (Cannot links C) of form [[id1, id2]].
        normalized: bool, do we normalize the score values into range [0, 1]?

    Returns:
        A tuple of calulated scores and details for each pair
        in form of (final_score, sim_scores, dis_scores)
    """
    sim_scores = _qij_based_scores(Q, sim, normalized, link_type="sim")
    dis_scores = _qij_based_scores(Q, dis, normalized, link_type="dis")

    final_score = 0.5 * sim_scores.mean() + 0.5 * dis_scores.mean()
    return final_score, sim_scores, dis_scores


def min_max_qij_based_score(Q):
    """Calculate the min and max value of log(q_ij) for all pairs of Q
    """
    if Q.ndim > 1:
        Q = squareform(Q)
    score_all_pairs = np.log(Q)
    return score_all_pairs.min(), score_all_pairs.max()


def normalize_scores(scores):
    """Normalize the socres in [0, 1] """
    return (scores - scores.min()) / (scores.max() - scores.min())

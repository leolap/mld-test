from src.etl.extract import extract_data, load_data
from src.etl.knn import (
    perform_k_fold_validation,
    write_pdf_comparison,
    knn_predict_proba,
    plot_multiclass_roc_curve,
)
from src.etl.tsne import (
    normalize_data,
    perform_tsne,
    generate_tsne_plot,
    generate_tsne_plot_3d,
)

data = load_data("src/data/mini_gm_public_v0.1.p")
syndrome_ids, embeddings = extract_data(data)

n_components = 2
perplexity = 35

normalized_embeddings = normalize_data(embeddings)
embeddings_tsne = perform_tsne(
    normalized_embeddings, n_components=n_components, perplexity=perplexity, n_iter=500
)

if n_components == 2:
    generate_tsne_plot(syndrome_ids, embeddings_tsne)
elif n_components == 3:
    generate_tsne_plot_3d(syndrome_ids, embeddings_tsne)
else:
    raise ValueError("n_components must be 2 or 3 for 2D or 3D visualization")

n_splits = 10
n_neighbors = 5

dict_cosine = perform_k_fold_validation(
    embeddings, syndrome_ids, n_splits, n_neighbors, "cosine"
)
dict_euclidean = perform_k_fold_validation(
    embeddings, syndrome_ids, n_splits, n_neighbors, "euclidean"
)

write_pdf_comparison(n_splits, dict_cosine, dict_euclidean)

y_true_e, y_prob_e = knn_predict_proba(embeddings, syndrome_ids, "euclidean", n_neighbors)
y_true_c, y_prob_c = knn_predict_proba(embeddings, syndrome_ids, "cosine", n_neighbors)

y_true_te, y_prob_te = knn_predict_proba(embeddings_tsne, syndrome_ids, "euclidean", n_neighbors)
y_true_tc, y_prob_tc = knn_predict_proba(embeddings_tsne, syndrome_ids, "cosine", n_neighbors)

true_values = [y_true_e, y_true_c, y_true_te, y_true_tc]
prob_predictions = [y_prob_e, y_prob_c, y_prob_te, y_prob_tc]
labels = ['Euclidean', 'Cosine', 't-SNE Euclidean', 't-SNE Cosine']

plot_multiclass_roc_curve(true_values, prob_predictions, labels)


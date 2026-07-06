import argparse
import os

import numpy as np


def mse(pred, true):
    return np.mean((pred - true) ** 2, axis=(1, 2))


def mae(pred, true):
    return np.mean(np.abs(pred - true), axis=(1, 2))


def summarize(name, values):
    values = np.asarray(values)
    return (
        f"{name}: mean={values.mean():.6f}, "
        f"p25={np.percentile(values, 25):.6f}, "
        f"p50={np.percentile(values, 50):.6f}, "
        f"p75={np.percentile(values, 75):.6f}"
    )


def summarize_finite(name, values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return f"{name}: n=0"
    return summarize(name, values)


def maybe_corr(name, left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3:
        return
    left = left[mask]
    right = right[mask]
    if left.std() > 1e-6 and right.std() > 1e-6:
        corr = np.corrcoef(left, right)[0, 1]
        if np.isfinite(corr):
            print(f"{name}: {corr:.6f}")


def load_required(path, filename):
    full_path = os.path.join(path, filename)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"missing {full_path}")
    return np.load(full_path)


def align_like_prediction(array, pred):
    if array.shape == pred.shape:
        return array
    raise ValueError(f"unexpected diagnostic shape {array.shape}; prediction shape is {pred.shape}")


def sample_mean(array):
    array = np.asarray(array)
    return array.reshape(array.shape[0], -1).mean(axis=1)


def maybe_get_prediction(diag, name, pred):
    if name not in diag:
        return None
    return align_like_prediction(diag[name], pred)


def print_gain_rate(name, gain):
    print(f"{name} better than baseline: {(gain > 0).mean():.2%}")
    print(f"harmful {name} rate: {(gain < 0).mean():.2%}")


def print_gain_rate_vs(name, comparator, gain):
    print(f"{name} better than {comparator}: {(gain > 0).mean():.2%}")
    print(f"harmful {name} vs {comparator} rate: {(gain < 0).mean():.2%}")


def print_slice_summary(name, mask, values):
    if mask.any():
        print(f"{name}: n={mask.sum()} ({mask.mean():.2%}), mean={values[mask].mean():.6f}")


def action_labels(diag, action_count):
    if "rpo_action_names" not in diag:
        return ["baseline", "raft_fused"] + [f"candidate_{idx}" for idx in range(2, action_count)]
    values = diag["rpo_action_names"][0].reshape(-1)
    labels = []
    for idx in range(action_count):
        if idx == 0:
            labels.append("baseline")
        elif idx == 1:
            labels.append("raft_fused")
        else:
            labels.append(f"period_g{int(round(float(values[idx])))}")
    return labels


def analyze_raft_topm_rpo(result_dir, pred, true, diag, final_mse, final_mae):
    required = [
        "baseline",
        "rpo_reference_forecast",
        "rpo_reranked_forecast",
        "rpo_candidate_mae",
        "rpo_candidate_mse",
        "rpo_policy_probabilities",
        "rpo_reference_probabilities",
        "rpo_candidate_similarity",
    ]
    missing = [key for key in required if key not in diag]
    if missing:
        print("\nThis RAFT top-M RPO run is missing diagnostics:", ", ".join(missing))
        return

    baseline = maybe_get_prediction(diag, "baseline", pred)
    reference = maybe_get_prediction(diag, "rpo_reference_forecast", pred)
    raft = maybe_get_prediction(diag, "raw_retrieval_forecast", pred)
    reranked = maybe_get_prediction(diag, "rpo_reranked_forecast", pred)
    selected = maybe_get_prediction(diag, "rpo_selected_forecast", pred)
    oracle = maybe_get_prediction(diag, "rpo_oracle_forecast", pred)

    baseline_mse = mse(baseline, true)
    baseline_mae = mae(baseline, true)
    reference_mse = mse(reference, true)
    reference_mae = mae(reference, true)
    raft_mse = mse(raft, true) if raft is not None else reference_mse
    raft_mae = mae(raft, true) if raft is not None else reference_mae
    reranked_mse = mse(reranked, true)
    reranked_mae = mae(reranked, true)
    selected_mse = mse(selected, true) if selected is not None else None
    selected_mae = mae(selected, true) if selected is not None else None
    oracle_mse = mse(oracle, true) if oracle is not None else None
    oracle_mae = mae(oracle, true) if oracle is not None else None

    candidate_mse = np.asarray(diag["rpo_candidate_mse"])
    candidate_mae = np.asarray(diag["rpo_candidate_mae"])
    if candidate_mse.ndim != 2:
        candidate_mse = candidate_mse.reshape(candidate_mse.shape[0], -1)
    if candidate_mae.ndim != 2:
        candidate_mae = candidate_mae.reshape(candidate_mae.shape[0], -1)
    candidate_count = candidate_mae.shape[1]
    policy_prob = np.asarray(diag["rpo_policy_probabilities"]).reshape(candidate_mae.shape)
    reference_prob = np.asarray(diag["rpo_reference_probabilities"]).reshape(candidate_mae.shape)
    candidate_similarity = np.asarray(diag["rpo_candidate_similarity"]).reshape(candidate_mae.shape)
    candidate_period = np.asarray(diag.get("rpo_candidate_period", np.zeros_like(candidate_mae))).reshape(candidate_mae.shape)
    candidate_rank = np.asarray(diag.get("rpo_candidate_rank", np.zeros_like(candidate_mae))).reshape(candidate_mae.shape)
    candidate_scores = np.asarray(diag.get("rpo_candidate_scores", np.zeros_like(candidate_mae))).reshape(candidate_mae.shape)

    reference_is_raft = bool(np.asarray(diag.get("rpo_reference_is_raft", [[1]])).reshape(-1).mean() >= 0.5)
    reference_name = "raft" if reference_is_raft else "baseline"
    final_gain_ref_mse = reference_mse - final_mse
    final_gain_ref_mae = reference_mae - final_mae
    reranked_gain_ref_mse = reference_mse - reranked_mse
    reranked_gain_ref_mae = reference_mae - reranked_mae
    raft_gain_baseline_mse = baseline_mse - raft_mse
    raft_gain_baseline_mae = baseline_mae - raft_mae
    candidate_gain_ref_mae = reference_mae[:, None] - candidate_mae
    candidate_gain_ref_mse = reference_mse[:, None] - candidate_mse
    best_candidate_gain_mae = candidate_gain_ref_mae.max(axis=1)
    best_candidate_gain_mse = candidate_gain_ref_mse[
        np.arange(candidate_gain_ref_mse.shape[0]), candidate_gain_ref_mae.argmax(axis=1)
    ]

    oracle_gain_ref_mse = reference_mse - oracle_mse if oracle_mse is not None else best_candidate_gain_mse
    oracle_gain_ref_mae = reference_mae - oracle_mae if oracle_mae is not None else best_candidate_gain_mae
    oracle_use = oracle_gain_ref_mae > 1e-12

    ref_top_index = reference_prob.argmax(axis=1)
    theta_top_index = policy_prob.argmax(axis=1)
    ref_top_mae = candidate_mae[np.arange(candidate_mae.shape[0]), ref_top_index]
    theta_top_mae = candidate_mae[np.arange(candidate_mae.shape[0]), theta_top_index]
    ref_top_gain_mae = reference_mae - ref_top_mae
    theta_top_gain_mae = reference_mae - theta_top_mae
    expected_ref_candidate_mae = (reference_prob * candidate_mae).sum(axis=1)
    expected_theta_candidate_mae = (policy_prob * candidate_mae).sum(axis=1)

    print()
    print(f"RPO mode: RAFT top-M utility rerank; utility reference = {reference_name}")
    print(f"top-M candidate actions: {candidate_count}")
    print(summarize("baseline_mse", baseline_mse))
    print(summarize("baseline_mae", baseline_mae))
    print(summarize(f"{reference_name}_reference_mse", reference_mse))
    print(summarize(f"{reference_name}_reference_mae", reference_mae))
    if raft is not None:
        print(summarize("raft_always_retrieval_mse", raft_mse))
        print(summarize("raft_always_retrieval_mae", raft_mae))
    print(summarize("rpo_reranked_mse", reranked_mse))
    print(summarize("rpo_reranked_mae", reranked_mae))
    if selected_mse is not None:
        print(summarize("rpo_hard_selected_mse", selected_mse))
        print(summarize("rpo_hard_selected_mae", selected_mae))
    if oracle_mse is not None:
        print(summarize("oracle_topm_rerank_mse", oracle_mse))
        print(summarize("oracle_topm_rerank_mae", oracle_mae))

    print()
    print(summarize("raft_always_gain_vs_baseline_mse", raft_gain_baseline_mse))
    print(summarize("raft_always_gain_vs_baseline_mae", raft_gain_baseline_mae))
    print(summarize("rpo_reranked_gain_vs_reference_mse", reranked_gain_ref_mse))
    print(summarize("rpo_reranked_gain_vs_reference_mae", reranked_gain_ref_mae))
    print(summarize("final_gain_vs_reference_mse", final_gain_ref_mse))
    print(summarize("final_gain_vs_reference_mae", final_gain_ref_mae))
    print(summarize("final_gain_vs_baseline_mse", baseline_mse - final_mse))
    print(summarize("final_gain_vs_baseline_mae", baseline_mae - final_mae))
    print(summarize("oracle_topm_gain_vs_reference_mse", oracle_gain_ref_mse))
    print(summarize("oracle_topm_gain_vs_reference_mae", oracle_gain_ref_mae))
    positive_oracle = oracle_gain_ref_mae > 1e-12
    captured = np.zeros_like(final_gain_ref_mae)
    captured[positive_oracle] = final_gain_ref_mae[positive_oracle] / np.maximum(
        oracle_gain_ref_mae[positive_oracle],
        1e-12,
    )
    print(summarize("rpo_gain_capture_vs_oracle_topm_mae", captured[positive_oracle] if positive_oracle.any() else captured))

    print()
    print_gain_rate_vs("RAFT always retrieval", "baseline (MSE)", raft_gain_baseline_mse)
    print_gain_rate_vs("RPO rerank", "reference (MAE)", reranked_gain_ref_mae)
    print_gain_rate_vs("final", "reference (MAE)", final_gain_ref_mae)
    print_gain_rate_vs("oracle top-M rerank", "reference (MAE)", oracle_gain_ref_mae)

    accept_prob = sample_mean(diag["rpo_accept_probability"]) if "rpo_accept_probability" in diag else None
    predicted_utility = sample_mean(diag["rpo_predicted_utility"]) if "rpo_predicted_utility" in diag else None
    action_index = diag["action_index"].reshape(-1).astype(np.int64) if "action_index" in diag else None
    oracle_action_index = (
        diag["oracle_action_index"].reshape(-1).astype(np.int64)
        if "oracle_action_index" in diag else None
    )

    print()
    reference_entropy = -(reference_prob * np.log(reference_prob + 1e-8)).sum(axis=1)
    policy_entropy = -(policy_prob * np.log(policy_prob + 1e-8)).sum(axis=1)
    policy_kl_ref = (policy_prob * (np.log(policy_prob + 1e-8) - np.log(reference_prob + 1e-8))).sum(axis=1)
    print(summarize("pi_ref_entropy", reference_entropy))
    print(summarize("pi_theta_entropy", policy_entropy))
    print(summarize("kl(pi_theta || pi_ref)", policy_kl_ref))
    print(summarize("ref_top1_candidate_gain_vs_reference_mae", ref_top_gain_mae))
    print(summarize("theta_top1_candidate_gain_vs_reference_mae", theta_top_gain_mae))
    print(summarize("expected_pi_ref_candidate_mae", expected_ref_candidate_mae))
    print(summarize("expected_pi_theta_candidate_mae", expected_theta_candidate_mae))
    print(summarize("best_candidate_gain_vs_reference_mae", best_candidate_gain_mae))
    print(f"oracle best candidate equals RAFT-sim top1: {(ref_top_index == candidate_gain_ref_mae.argmax(axis=1)).mean():.2%}")
    print(f"RPO top1 equals oracle best candidate: {(theta_top_index == candidate_gain_ref_mae.argmax(axis=1)).mean():.2%}")
    print(f"RPO top1 differs from RAFT-sim top1: {(theta_top_index != ref_top_index).mean():.2%}")

    if accept_prob is not None:
        print()
        print(summarize("rpo_accept_probability", accept_prob))
        print(f"soft accept rate (p_accept>=0.5): {(accept_prob >= 0.5).mean():.2%}")
        maybe_corr("corr(rpo_accept_probability, oracle_topm_gain_mae)", accept_prob, oracle_gain_ref_mae)
        maybe_corr("corr(rpo_accept_probability, final_gain_ref_mae)", accept_prob, final_gain_ref_mae)
    if predicted_utility is not None:
        print(summarize("rpo_predicted_utility", predicted_utility))
        maybe_corr("corr(rpo_predicted_utility, oracle_topm_gain_mae)", predicted_utility, oracle_gain_ref_mae)
        maybe_corr("corr(rpo_predicted_utility, final_gain_ref_mae)", predicted_utility, final_gain_ref_mae)

    if action_index is not None and oracle_action_index is not None:
        model_use = action_index > 0
        oracle_use = oracle_action_index > 0
        print()
        print(f"argmax action accuracy: {(action_index == oracle_action_index).mean():.2%}")
        print(f"argmax model rerank-use rate: {model_use.mean():.2%}")
        print(f"argmax oracle rerank-use rate: {oracle_use.mean():.2%}")
        print(f"gate decision accuracy: {(model_use == oracle_use).mean():.2%}")
        print(f"true fallback rate: {((~model_use) & (~oracle_use)).mean():.2%}")
        print(f"false accept rate: {(model_use & (~oracle_use)).mean():.2%}")
        print(f"false reject rate: {((~model_use) & oracle_use).mean():.2%}")
        print_slice_summary("final_gain_ref_mae on oracle-use slice", oracle_use, final_gain_ref_mae)
        print_slice_summary("final_gain_ref_mae on oracle-fallback slice", ~oracle_use, final_gain_ref_mae)
        print_slice_summary("final_gain_ref_mae on false-accept slice", model_use & (~oracle_use), final_gain_ref_mae)
        print_slice_summary("final_gain_ref_mae on false-reject slice", (~model_use) & oracle_use, final_gain_ref_mae)

    if "rpo_pair_count" in diag:
        print()
        print(summarize("rpo_pair_count", sample_mean(diag["rpo_pair_count"])))
    print(summarize("candidate_similarity_mean", candidate_similarity.mean(axis=1)))
    print(summarize("candidate_similarity_max", candidate_similarity.max(axis=1)))
    print(summarize("candidate_score_mean", candidate_scores.mean(axis=1)))
    print(summarize("candidate_score_max", candidate_scores.max(axis=1)))
    maybe_corr("corr(candidate_similarity_max, best_candidate_gain_mae)", candidate_similarity.max(axis=1), best_candidate_gain_mae)
    maybe_corr("corr(candidate_score_max, best_candidate_gain_mae)", candidate_scores.max(axis=1), best_candidate_gain_mae)

    print()
    print("Period-level candidate utility:")
    for period in sorted(np.unique(candidate_period.astype(int).reshape(-1)), reverse=True):
        mask = candidate_period.astype(int) == period
        period_gain = np.where(mask, candidate_gain_ref_mae, np.nan)
        period_best = np.nanmax(period_gain, axis=1)
        period_policy_mass = (policy_prob * mask).sum(axis=1)
        period_ref_mass = (reference_prob * mask).sum(axis=1)
        oracle_period = candidate_period[np.arange(candidate_period.shape[0]), candidate_gain_ref_mae.argmax(axis=1)]
        print(summarize_finite(f"period_{period}_best_gain_vs_reference_mae", period_best))
        print(summarize(f"period_{period}_pi_ref_mass", period_ref_mass))
        print(summarize(f"period_{period}_pi_theta_mass", period_policy_mass))
        print(f"period_{period}_oracle_best_rate: {(oracle_period.astype(int) == period).mean():.2%}")

    print()
    print("Rank-level candidate utility:")
    max_rank_to_print = min(5, int(np.nanmax(candidate_rank)) + 1)
    for rank in range(max_rank_to_print):
        mask = candidate_rank.astype(int) == rank
        rank_gain = np.where(mask, candidate_gain_ref_mae, np.nan)
        print(summarize_finite(f"rank_{rank}_gain_vs_reference_mae", rank_gain.reshape(-1)))
        print(f"rank_{rank}_oracle_best_rate: {((candidate_gain_ref_mae.argmax(axis=1)[:, None] == np.arange(candidate_count)[None, :]) & mask).any(axis=1).mean():.2%}")

    print()
    if oracle_gain_ref_mae.mean() <= 0:
        print("Likely bottleneck: RAFT top-M recall. Even oracle top-M reranking cannot beat the reference on average.")
    elif reranked_gain_ref_mae.mean() <= 0 < oracle_gain_ref_mae.mean():
        print("Likely bottleneck: RPO reranker. Oracle top-M candidates help, but pi_theta does not select/use them yet.")
    elif final_gain_ref_mae.mean() <= 0 < reranked_gain_ref_mae.mean():
        print("Likely bottleneck: utility gate. Reranking helps, but fallback/use decision loses the gain.")
    elif final_gain_ref_mae.mean() > 0 and final_gain_ref_mae.mean() >= reranked_gain_ref_mae.mean():
        print("RPO is useful: final gated forecast improves over the reference forecast on average.")
    elif final_gain_ref_mae.mean() > 0:
        print("RPO reranking is useful, but the gate is not preserving all rerank gain.")
    else:
        print("No clear RPO benefit: inspect oracle headroom, pi_theta top1 accuracy, and false accept/reject slices.")

    print("\nInterpretation guide:")
    print("- oracle_topm_gain_vs_reference_mae > 0: RAFT top-M contains useful forecast candidates.")
    print("- rpo_reranked_gain_vs_reference_mae > 0: learned RPO reranking improves over the RAFT/reference policy.")
    print("- final_gain_vs_reference_mae > 0: rerank + utility gate improves the selected reference forecast.")
    print("- RPO top1 equals oracle best candidate: reranker quality inside the retrieved top-M set.")
    print("- false accept/reject rates: whether predicted utility is gating harmful/useful reranks correctly.")
    print("- period/rank utility tables show where useful retrieval candidates are found.")


def analyze_raft_rpo(result_dir, pred, true, diag, final_mse, final_mae):
    if "baseline" not in diag or "rpo_candidate_mse" not in diag:
        print("\nThis RAFT-RPO run is missing baseline or candidate diagnostics.")
        return

    baseline = maybe_get_prediction(diag, "baseline", pred)
    baseline_mse = mse(baseline, true)
    candidate_mse = np.asarray(diag["rpo_candidate_mse"])
    if candidate_mse.ndim != 2:
        candidate_mse = candidate_mse.reshape(candidate_mse.shape[0], -1)
    candidate_gain = baseline_mse[:, None] - candidate_mse
    labels = action_labels(diag, candidate_mse.shape[1])

    raw_retrieval = maybe_get_prediction(diag, "raw_retrieval_forecast", pred)
    conditional_retrieval = maybe_get_prediction(diag, "retrieval_forecast", pred)
    selected_forecast = maybe_get_prediction(diag, "rpo_selected_forecast", pred)
    oracle_forecast = maybe_get_prediction(diag, "rpo_oracle_forecast", pred)

    raw_mse = mse(raw_retrieval, true) if raw_retrieval is not None else candidate_mse[:, 1]
    conditional_mse = mse(conditional_retrieval, true) if conditional_retrieval is not None else None
    selected_mse = mse(selected_forecast, true) if selected_forecast is not None else None
    oracle_mse = mse(oracle_forecast, true) if oracle_forecast is not None else None
    if oracle_mse is None and "oracle_err" in diag:
        oracle_mse = sample_mean(diag["oracle_err"])

    final_gain = baseline_mse - final_mse
    raw_gain = baseline_mse - raw_mse
    conditional_gain = baseline_mse - conditional_mse if conditional_mse is not None else None
    selected_gain = baseline_mse - selected_mse if selected_mse is not None else None
    oracle_gain = baseline_mse - oracle_mse if oracle_mse is not None else None

    print()
    print(summarize("baseline_mse", baseline_mse))
    print(summarize("raft_always_retrieval_mse", raw_mse))
    if conditional_mse is not None:
        print(summarize("rpo_conditional_retrieval_mse", conditional_mse))
    if selected_mse is not None:
        print(summarize("rpo_argmax_action_mse", selected_mse))
    if oracle_mse is not None:
        print(summarize("oracle_action_mse", oracle_mse))

    print()
    print(summarize("final_gain_vs_baseline", final_gain))
    print(summarize("raft_always_gain_vs_baseline", raw_gain))
    if conditional_gain is not None:
        print(summarize("rpo_conditional_retrieval_gain_vs_baseline", conditional_gain))
    if selected_gain is not None:
        print(summarize("rpo_argmax_action_gain_vs_baseline", selected_gain))
    if oracle_gain is not None:
        print(summarize("oracle_action_gain_vs_baseline", oracle_gain))
        print(summarize("policy_regret_vs_oracle", final_mse - oracle_mse))
        positive_oracle = oracle_gain > 1e-12
        captured = np.zeros_like(final_gain)
        captured[positive_oracle] = final_gain[positive_oracle] / np.maximum(oracle_gain[positive_oracle], 1e-12)
        print(summarize("rpo_gain_capture_on_oracle_positive", captured[positive_oracle] if positive_oracle.any() else captured))

    print()
    print_gain_rate("final", final_gain)
    print_gain_rate("raft always retrieval", raw_gain)
    if selected_gain is not None:
        print_gain_rate("rpo argmax action", selected_gain)
    if oracle_gain is not None:
        print_gain_rate("oracle action", oracle_gain)

    probs = diag["rpo_action_probabilities"] if "rpo_action_probabilities" in diag else diag.get("action_probabilities")
    action_index = diag["action_index"].reshape(-1).astype(np.int64) if "action_index" in diag else None
    oracle_action_index = (
        diag["oracle_action_index"].reshape(-1).astype(np.int64)
        if "oracle_action_index" in diag else None
    )

    print()
    print("RPO action set:", ", ".join(f"{idx}={name}" for idx, name in enumerate(labels)))
    if probs is not None:
        entropy = -(probs * np.log(probs + 1e-8)).sum(axis=1)
        no_retrieval_prob = probs[:, 0]
        accept_prob = probs[:, 1:].sum(axis=1)
        print(summarize("rpo_action_probability_entropy", entropy))
        print(summarize("rpo_no_retrieval_probability", no_retrieval_prob))
        print(summarize("rpo_accept_probability", accept_prob))
        print(f"soft RPO accept rate (p_retrieval>=0.5): {(accept_prob >= 0.5).mean():.2%}")
        maybe_corr("corr(rpo_accept_probability, final_gain)", accept_prob, final_gain)
        maybe_corr("corr(rpo_accept_probability, raft_always_gain)", accept_prob, raw_gain)
        if oracle_gain is not None:
            maybe_corr("corr(rpo_accept_probability, oracle_action_gain)", accept_prob, oracle_gain)

    if action_index is not None and oracle_action_index is not None:
        model_use = action_index > 0
        oracle_use = oracle_action_index > 0
        print()
        print(f"argmax action accuracy: {(action_index == oracle_action_index).mean():.2%}")
        print(f"argmax model retrieval-use rate: {model_use.mean():.2%}")
        print(f"argmax oracle retrieval-use rate: {oracle_use.mean():.2%}")
        print(f"abstention decision accuracy: {(model_use == oracle_use).mean():.2%}")
        print(f"true reject rate: {((~model_use) & (~oracle_use)).mean():.2%}")
        print(f"false accept rate: {(model_use & (~oracle_use)).mean():.2%}")
        print(f"false reject rate: {((~model_use) & oracle_use).mean():.2%}")
        print_slice_summary("final_gain on oracle-use slice", oracle_use, final_gain)
        print_slice_summary("final_gain on oracle-abstain slice", ~oracle_use, final_gain)
        print_slice_summary("final_gain on false-accept slice", model_use & (~oracle_use), final_gain)
        print_slice_summary("final_gain on false-reject slice", (~model_use) & oracle_use, final_gain)

    print()
    for idx, name in enumerate(labels):
        cur_mse = candidate_mse[:, idx]
        cur_gain = candidate_gain[:, idx]
        print(summarize(f"candidate_{idx}_{name}_mse", cur_mse))
        print(summarize(f"candidate_{idx}_{name}_gain_vs_baseline", cur_gain))
        print(f"candidate_{idx}_{name} better than baseline: {(cur_gain > 0).mean():.2%}")
        if probs is not None:
            print(summarize(f"candidate_{idx}_{name}_probability", probs[:, idx]))
        if action_index is not None:
            print(f"candidate_{idx}_{name} argmax rate: {(action_index == idx).mean():.2%}")
        if oracle_action_index is not None:
            print(f"candidate_{idx}_{name} oracle rate: {(oracle_action_index == idx).mean():.2%}")
        print()

    if "rpo_best_retrieval_gain" in diag:
        best_retrieval_gain = sample_mean(diag["rpo_best_retrieval_gain"])
        print(summarize("best_retrieval_candidate_gain_vs_baseline", best_retrieval_gain))
        print(f"best retrieval candidate better than baseline: {(best_retrieval_gain > 0).mean():.2%}")

    if "policy_regret" in diag:
        print(summarize("saved_policy_regret", sample_mean(diag["policy_regret"])))

    for key in [
        "top_similarity",
        "primary_top_similarity",
        "period_similarity",
    ]:
        if key in diag:
            value = sample_mean(diag[key])
            print()
            print(summarize(key, value))
            maybe_corr(f"corr({key}, raft_always_gain)", value, raw_gain)
            maybe_corr(f"corr({key}, final_gain)", value, final_gain)

    print()
    if oracle_gain is not None and oracle_gain.mean() <= 0:
        print("Likely bottleneck: RAFT retrieval candidates. Even oracle RPO cannot beat the host baseline on average.")
    elif raw_gain.mean() <= 0 and final_gain.mean() > raw_gain.mean():
        print("RPO is filtering harmful RAFT retrieval better than always-retrieve, but check whether final_gain is positive.")
    elif raw_gain.mean() > 0 and final_gain.mean() <= raw_gain.mean():
        print("Likely bottleneck: RPO action scorer. RAFT retrieval helps, but RPO loses part of the available gain.")
    elif oracle_gain is not None and final_gain.mean() <= 0 < oracle_gain.mean():
        print("Likely bottleneck: RPO action scorer. Oracle actions help, learned preferences still leave regret.")
    elif final_gain.mean() > raw_gain.mean() and final_gain.mean() > 0:
        print("RPO is useful: final forecast improves over both host baseline and always-retrieve RAFT on average.")
    elif final_gain.mean() > 0:
        print("Final forecast improves over baseline, but RPO does not clearly beat always-retrieve RAFT yet.")
    else:
        print("No clear RPO benefit: inspect false accepts/rejects and candidate-level oracle rates.")

    print("\nInterpretation guide:")
    print("- raft_always_gain_vs_baseline > 0: the original RAFT retrieval branch is useful before RPO.")
    print("- oracle_action_gain_vs_baseline > 0: the action set contains useful retrieval/no-retrieval decisions.")
    print("- final_gain_vs_baseline > raft_always_gain_vs_baseline: RPO improves over always using RAFT retrieval.")
    print("- high false accept rate: RPO still accepts retrieval when oracle would choose baseline.")
    print("- high false reject rate: RPO rejects useful RAFT retrieval candidates.")
    print("- candidate oracle rate shows which RAFT period/action actually carries useful signal.")


def main():
    parser = argparse.ArgumentParser(description="Analyze Phase-RPO-RFRL retrieval diagnostics.")
    parser.add_argument("result_dir", help="A results/ directory with pred.npy, true.npy and retrieval_diagnostics.npz")
    args = parser.parse_args()

    result_dir = args.result_dir.rstrip("/")
    pred = load_required(result_dir, "pred.npy")
    true = load_required(result_dir, "true.npy")
    diag = load_required(result_dir, "retrieval_diagnostics.npz")

    print(f"Result directory: {result_dir}")
    print(f"Samples: {pred.shape[0]}, horizon: {pred.shape[1]}, channels: {pred.shape[2]}")

    final_mse = mse(pred, true)
    final_mae = mae(pred, true)
    print(summarize("final_mse", final_mse))
    print(summarize("final_mae", final_mae))

    if (
        "rpo_candidate_mae" in diag
        and "rpo_reference_forecast" in diag
        and "rpo_policy_probabilities" in diag
    ):
        analyze_raft_topm_rpo(result_dir, pred, true, diag, final_mse, final_mae)
        return

    if "rpo_candidate_mse" in diag and (
        "rpo_action_probabilities" in diag or "action_probabilities" in diag
    ):
        analyze_raft_rpo(result_dir, pred, true, diag, final_mse, final_mae)
        return

    if "baseline" not in diag or "retrieval_forecast" not in diag:
        print("\nThis run does not contain baseline/retrieval counterfactuals.")
        print("Rerun after the latest code change to get segment-level diagnostics.")
        return

    baseline = maybe_get_prediction(diag, "baseline", pred)
    raw_retrieval = maybe_get_prediction(diag, "raw_retrieval_forecast", pred)
    adapter_retrieval = maybe_get_prediction(diag, "retrieval_forecast", pred)
    enhanced_retrieval = maybe_get_prediction(diag, "retrieval_enhanced", pred)

    baseline_mse = mse(baseline, true)
    adapter_mse = mse(adapter_retrieval, true)
    enhanced_mse = mse(enhanced_retrieval, true) if enhanced_retrieval is not None else None
    raw_mse = mse(raw_retrieval, true) if raw_retrieval is not None else None

    oracle_mse = None
    if "oracle_alpha" in diag and "retrieval_correction" in diag:
        oracle_alpha = diag["oracle_alpha"]
        correction = align_like_prediction(diag["retrieval_correction"], pred)
        oracle_forecast = baseline + oracle_alpha * correction
        oracle_mse = mse(oracle_forecast, true)

    print()
    print(summarize("baseline_mse", baseline_mse))
    if raw_mse is not None:
        print(summarize("raw_retrieval_mse", raw_mse))
    print(summarize("adapter_retrieval_mse", adapter_mse))
    if enhanced_mse is not None:
        print(summarize("retrieval_enhanced_mse", enhanced_mse))
    if oracle_mse is not None:
        print(summarize("oracle_alpha_mse", oracle_mse))

    final_gain = baseline_mse - final_mse
    adapter_gain = baseline_mse - adapter_mse
    enhanced_gain = baseline_mse - enhanced_mse if enhanced_mse is not None else None
    raw_gain = baseline_mse - raw_mse if raw_mse is not None else None
    oracle_gain = baseline_mse - oracle_mse if oracle_mse is not None else None

    print()
    print(summarize("final_gain_vs_baseline", final_gain))
    if raw_gain is not None:
        print(summarize("raw_retrieval_gain_vs_baseline", raw_gain))
    print(summarize("adapter_gain_vs_baseline", adapter_gain))
    if enhanced_gain is not None:
        print(summarize("enhanced_gain_vs_baseline", enhanced_gain))
    if oracle_gain is not None:
        print(summarize("oracle_alpha_gain_vs_baseline", oracle_gain))
        print(summarize("policy_regret_vs_oracle", final_mse - oracle_mse))

    print()
    print_gain_rate("final", final_gain)
    if raw_gain is not None:
        print_gain_rate("raw retrieval", raw_gain)
    print_gain_rate("adapter retrieval", adapter_gain)
    if oracle_gain is not None:
        print_gain_rate("oracle alpha", oracle_gain)

    oracle_use = None
    model_use = None
    if "action_alpha" in diag:
        action_alpha = sample_mean(diag["action_alpha"])
        print()
        print(summarize("action_alpha", action_alpha))
        if "action_alpha_bins" in diag:
            bins = diag["action_alpha_bins"][0].reshape(-1)
            print("action_alpha_bins:", ",".join(f"{item:.4g}" for item in bins))
        if "oracle_alpha" in diag:
            oracle_alpha_mean = sample_mean(diag["oracle_alpha"])
            print(summarize("oracle_alpha", oracle_alpha_mean))
            print(summarize("abs_action_alpha_error", np.abs(action_alpha - oracle_alpha_mean)))
            print(f"expected-alpha retrieval-use rate: {(action_alpha > 1e-6).mean():.2%}")
            print(f"oracle retrieval-use rate: {(oracle_alpha_mean > 1e-6).mean():.2%}")
            maybe_corr("corr(action_alpha, oracle_alpha)", action_alpha, oracle_alpha_mean)
        maybe_corr("corr(action_alpha, final_gain)", action_alpha, final_gain)
        maybe_corr("corr(action_alpha, adapter_gain)", action_alpha, adapter_gain)
        if raw_gain is not None:
            maybe_corr("corr(action_alpha, raw_retrieval_gain)", action_alpha, raw_gain)

    if "action_probabilities" in diag:
        probs = diag["action_probabilities"]
        entropy = -(probs * np.log(probs + 1e-8)).sum(axis=1)
        print(summarize("action_probability_entropy", entropy))
        no_retrieval_prob = probs[:, 0]
        print(summarize("no_retrieval_probability", no_retrieval_prob))
        maybe_corr("corr(no_retrieval_probability, final_gain)", no_retrieval_prob, final_gain)
        maybe_corr("corr(no_retrieval_probability, adapter_gain)", no_retrieval_prob, adapter_gain)

    if "action_index" in diag and "oracle_action_index" in diag:
        action_index = diag["action_index"].reshape(-1).astype(np.int64)
        oracle_action_index = diag["oracle_action_index"].reshape(-1).astype(np.int64)
        model_use = action_index > 0
        oracle_use = oracle_action_index > 0
        print()
        print(f"argmax action accuracy: {(action_index == oracle_action_index).mean():.2%}")
        print(f"argmax model retrieval-use rate: {model_use.mean():.2%}")
        print(f"argmax oracle retrieval-use rate: {oracle_use.mean():.2%}")
        print(f"abstention decision accuracy: {(model_use == oracle_use).mean():.2%}")
        print(f"true reject rate: {((~model_use) & (~oracle_use)).mean():.2%}")
        print(f"false accept rate: {(model_use & (~oracle_use)).mean():.2%}")
        print(f"false reject rate: {((~model_use) & oracle_use).mean():.2%}")
        print_slice_summary("final_gain on oracle-use slice", oracle_use, final_gain)
        print_slice_summary("final_gain on oracle-abstain slice", ~oracle_use, final_gain)
        print_slice_summary("final_gain on false-accept slice", model_use & (~oracle_use), final_gain)
        print_slice_summary("final_gain on false-reject slice", (~model_use) & oracle_use, final_gain)

    if "policy_regret" in diag:
        print()
        print(summarize("saved_policy_regret", sample_mean(diag["policy_regret"])))

    if "retrieval_residual_scale" in diag:
        print()
        print(summarize("retrieval_residual_scale", sample_mean(diag["retrieval_residual_scale"])))
    if "retrieval_correction_norm" in diag:
        print(summarize("retrieval_correction_norm", sample_mean(diag["retrieval_correction_norm"])))

    if "fusion_weight" in diag and "action_alpha" not in diag:
        fusion_weight = sample_mean(diag["fusion_weight"])
        print()
        print(summarize("fusion_weight", fusion_weight))

    if "preference_score" in diag:
        preference = sample_mean(diag["preference_score"])
        print()
        print(summarize("rpo_preference_score", preference))
        if oracle_use is not None:
            rpo_accept = preference >= 0.5
            print(f"rpo accept rate: {rpo_accept.mean():.2%}")
            print(f"rpo abstention accuracy: {(rpo_accept == oracle_use).mean():.2%}")
            print(f"rpo false accept rate: {(rpo_accept & (~oracle_use)).mean():.2%}")
            print(f"rpo false reject rate: {((~rpo_accept) & oracle_use).mean():.2%}")
        maybe_corr("corr(rpo_preference_score, final_gain)", preference, final_gain)
        maybe_corr("corr(rpo_preference_score, adapter_gain)", preference, adapter_gain)
        if raw_gain is not None:
            maybe_corr("corr(rpo_preference_score, raw_retrieval_gain)", preference, raw_gain)
        if "oracle_alpha" in diag:
            maybe_corr("corr(rpo_preference_score, oracle_alpha)", preference, sample_mean(diag["oracle_alpha"]))

    for key in [
        "primary_top_similarity",
        "top_similarity",
        "time_similarity",
        "phase_similarity",
        "amplitude_similarity",
    ]:
        if key in diag:
            value = sample_mean(diag[key])
            print()
            print(summarize(key, value))
            maybe_corr(f"corr({key}, adapter_gain)", value, adapter_gain)
            if raw_gain is not None:
                maybe_corr(f"corr({key}, raw_retrieval_gain)", value, raw_gain)

    print()
    if oracle_gain is not None and oracle_gain.mean() > 0 and final_gain.mean() <= 0:
        print("Likely bottleneck: retrieval action policy. Oracle actions can help, but learned actions still leave regret.")
    elif raw_gain is not None and raw_gain.mean() <= 0 and (
        oracle_gain is None or oracle_gain.mean() <= 0
    ):
        print("Likely bottleneck: retrieval quality. Time-primary candidates still do not beat the host baseline on average.")
    elif adapter_gain.mean() <= 0 and (
        oracle_gain is None or oracle_gain.mean() <= 0
    ):
        print("Likely bottleneck: retrieval adapter. Candidate residuals may exist, but the learned correction is not useful.")
    elif adapter_gain.mean() > 0 and final_gain.mean() <= 0:
        print("Likely bottleneck: adaptive fusion/action scaling. Adapter correction has signal, but final control degrades it.")
    elif final_gain.mean() <= 0:
        print("Likely bottleneck: final retrieval-control coupling. Inspect action_alpha and preference correlations.")
    else:
        print("No obvious segment failure: final forecast improves over baseline on average. Inspect slice-level failures next.")

    print("\nInterpretation guide:")
    print("- raw_retrieval_gain_vs_baseline <= 0: retrieved historical residuals are weak as full forecasts.")
    print("- adapter_gain_vs_baseline <= 0: adapter/correction is not yet producing useful residuals.")
    print("- oracle_alpha_gain_vs_baseline > 0: the action space contains useful retrieval decisions.")
    print("- policy_regret_vs_oracle > 0: learned policy leaves accuracy on the table.")
    print("- high false accept rate: RFRL still uses retrieval when oracle would abstain.")
    print("- high false reject rate: RFRL rejects useful retrieval corrections.")


if __name__ == "__main__":
    main()

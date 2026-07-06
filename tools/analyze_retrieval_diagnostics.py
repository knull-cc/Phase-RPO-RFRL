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


def print_slice_summary(name, mask, values):
    if mask.any():
        print(f"{name}: n={mask.sum()} ({mask.mean():.2%}), mean={values[mask].mean():.6f}")


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

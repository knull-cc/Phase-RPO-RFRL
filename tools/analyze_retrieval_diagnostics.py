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
    if left.std() > 1e-12 and right.std() > 1e-12:
        corr = np.corrcoef(left, right)[0, 1]
        print(f"{name}: {corr:.6f}")


def load_required(path, filename):
    full_path = os.path.join(path, filename)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"missing {full_path}")
    return np.load(full_path)


def align_like_prediction(array, pred):
    if array.ndim == pred.ndim + 1 and array.shape[2:] == pred.shape[1:]:
        return array
    if array.shape == pred.shape:
        return array
    raise ValueError(f"unexpected diagnostic shape {array.shape}; prediction shape is {pred.shape}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Phase-RPO-RFRL retrieval diagnostics.")
    parser.add_argument("result_dir", help="A directory under results/ that contains pred.npy, true.npy and retrieval_diagnostics.npz")
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

    baseline = align_like_prediction(diag["baseline"], pred)
    adapter_retrieval = align_like_prediction(diag["retrieval_forecast"], pred)
    retrieval_enhanced = align_like_prediction(diag["retrieval_enhanced"], pred)
    raw_phase_retrieval = None
    if "raw_retrieval_forecast" in diag:
        raw_phase_retrieval = align_like_prediction(diag["raw_retrieval_forecast"], pred)

    baseline_mse = mse(baseline, true)
    adapter_mse = mse(adapter_retrieval, true)
    enhanced_mse = mse(retrieval_enhanced, true)
    raw_phase_mse = None
    if raw_phase_retrieval is not None:
        raw_phase_mse = mse(raw_phase_retrieval, true)

    print()
    print(summarize("baseline_mse", baseline_mse))
    if raw_phase_mse is not None:
        print(summarize("phase_raw_retrieval_mse", raw_phase_mse))
    else:
        print("phase_raw_retrieval_mse: unavailable (missing raw_retrieval_forecast)")
    print(summarize("adapter_retrieval_mse", adapter_mse))
    print(summarize("retrieval_enhanced_mse", enhanced_mse))

    final_gain = baseline_mse - final_mse
    adapter_gain = baseline_mse - adapter_mse
    enhanced_gain = baseline_mse - enhanced_mse
    raw_phase_gain = None
    if raw_phase_mse is not None:
        raw_phase_gain = baseline_mse - raw_phase_mse

    print()
    print(summarize("final_gain_vs_baseline", final_gain))
    if raw_phase_gain is not None:
        print(summarize("phase_raw_gain_vs_baseline", raw_phase_gain))
    print(summarize("adapter_gain_vs_baseline", adapter_gain))
    print(summarize("enhanced_gain_vs_baseline", enhanced_gain))

    print()
    print(f"final better than baseline: {(final_gain > 0).mean():.2%}")
    if raw_phase_gain is not None:
        print(f"phase raw retrieval better than baseline: {(raw_phase_gain > 0).mean():.2%}")
    print(f"adapter retrieval better than baseline: {(adapter_gain > 0).mean():.2%}")
    print(f"retrieval enhanced better than baseline: {(enhanced_gain > 0).mean():.2%}")
    if raw_phase_gain is not None:
        print(f"harmful phase raw retrieval rate: {(raw_phase_gain < 0).mean():.2%}")
    print(f"harmful adapter retrieval rate: {(adapter_gain < 0).mean():.2%}")

    if "fusion_weight" in diag:
        fusion_weight = diag["fusion_weight"]
        print()
        print(summarize("fusion_weight", fusion_weight.reshape(fusion_weight.shape[0], -1).mean(axis=1)))
        adapter_harmful = adapter_gain < 0
        adapter_helpful = adapter_gain > 0
        if adapter_harmful.any():
            print(f"mean fusion weight on harmful adapter retrieval: {fusion_weight[adapter_harmful].mean():.6f}")
        if adapter_helpful.any():
            print(f"mean fusion weight on helpful adapter retrieval: {fusion_weight[adapter_helpful].mean():.6f}")
        if raw_phase_gain is not None:
            raw_harmful = raw_phase_gain < 0
            raw_helpful = raw_phase_gain > 0
            if raw_harmful.any():
                print(f"mean fusion weight on harmful phase raw retrieval: {fusion_weight[raw_harmful].mean():.6f}")
            if raw_helpful.any():
                print(f"mean fusion weight on helpful phase raw retrieval: {fusion_weight[raw_helpful].mean():.6f}")

    if "preference_score" in diag:
        preference = diag["preference_score"].reshape(diag["preference_score"].shape[0], -1).mean(axis=1)
        print()
        print(summarize("preference_score", preference))
        if raw_phase_gain is not None:
            maybe_corr("corr(preference_score, phase_raw_gain)", preference, raw_phase_gain)
        maybe_corr("corr(preference_score, adapter_gain)", preference, adapter_gain)

    if "top_similarity" in diag:
        top_sim = diag["top_similarity"]
        sim_mean = top_sim.reshape(top_sim.shape[0], -1).mean(axis=1)
        print()
        print(summarize("top_similarity", sim_mean))
        if raw_phase_gain is not None:
            maybe_corr("corr(top_similarity, phase_raw_gain)", sim_mean, raw_phase_gain)
        maybe_corr("corr(top_similarity, adapter_gain)", sim_mean, adapter_gain)

    print()
    if raw_phase_gain is not None and raw_phase_gain.mean() <= 0:
        print("Likely bottleneck: phase-aware retrieval quality. Raw phase candidates are not better than the host baseline on average.")
    elif adapter_gain.mean() <= 0:
        print("Likely bottleneck: retrieval adapter. Raw phase retrieval may have signal, but the learned correction degrades it.")
    elif enhanced_gain.mean() <= 0:
        print("Likely bottleneck: RPO/preference scaling. Adapter retrieval has signal, but preference-enhanced retrieval degrades it.")
    elif final_gain.mean() <= 0:
        print("Likely bottleneck: RFRL/adaptive fusion. Retrieval branch has signal, but the final controller uses it poorly.")
    else:
        print("No obvious segment failure: final forecast improves over baseline on average. Inspect slice-level failures next.")

    if raw_phase_gain is not None:
        print("\nInterpretation guide:")
        print("- phase_raw_gain_vs_baseline <= 0: phase retrieval candidates/residuals are not useful yet.")
        print("- phase_raw_gain_vs_baseline > 0 and adapter_gain_vs_baseline <= 0: adapter corrupts useful retrieval signal.")
        print("- adapter_gain_vs_baseline > 0 and enhanced_gain_vs_baseline <= 0: RPO/preference scaling is the bottleneck.")
        print("- enhanced_gain_vs_baseline > 0 and final_gain_vs_baseline <= 0: RFRL/adaptive fusion is the bottleneck.")


if __name__ == "__main__":
    main()

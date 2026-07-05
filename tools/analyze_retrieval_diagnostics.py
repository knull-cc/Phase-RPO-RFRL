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
    retrieval = align_like_prediction(diag["retrieval_forecast"], pred)
    retrieval_enhanced = align_like_prediction(diag["retrieval_enhanced"], pred)

    baseline_mse = mse(baseline, true)
    retrieval_mse = mse(retrieval, true)
    enhanced_mse = mse(retrieval_enhanced, true)

    print()
    print(summarize("baseline_mse", baseline_mse))
    print(summarize("raw_retrieval_mse", retrieval_mse))
    print(summarize("retrieval_enhanced_mse", enhanced_mse))

    final_gain = baseline_mse - final_mse
    raw_retrieval_gain = baseline_mse - retrieval_mse
    enhanced_gain = baseline_mse - enhanced_mse
    print()
    print(summarize("final_gain_vs_baseline", final_gain))
    print(summarize("raw_retrieval_gain_vs_baseline", raw_retrieval_gain))
    print(summarize("enhanced_gain_vs_baseline", enhanced_gain))

    print()
    print(f"final better than baseline: {(final_gain > 0).mean():.2%}")
    print(f"raw retrieval better than baseline: {(raw_retrieval_gain > 0).mean():.2%}")
    print(f"retrieval enhanced better than baseline: {(enhanced_gain > 0).mean():.2%}")
    print(f"harmful raw retrieval rate: {(raw_retrieval_gain < 0).mean():.2%}")

    if "fusion_weight" in diag:
        fusion_weight = diag["fusion_weight"]
        print()
        print(summarize("fusion_weight", fusion_weight.reshape(fusion_weight.shape[0], -1).mean(axis=1)))
        harmful = raw_retrieval_gain < 0
        helpful = raw_retrieval_gain > 0
        if harmful.any():
            print(f"mean fusion weight on harmful retrieval: {fusion_weight[harmful].mean():.6f}")
        if helpful.any():
            print(f"mean fusion weight on helpful retrieval: {fusion_weight[helpful].mean():.6f}")

    if "preference_score" in diag:
        preference = diag["preference_score"].reshape(diag["preference_score"].shape[0], -1).mean(axis=1)
        print()
        print(summarize("preference_score", preference))
        if raw_retrieval_gain.std() > 1e-12 and preference.std() > 1e-12:
            corr = np.corrcoef(preference, raw_retrieval_gain)[0, 1]
            print(f"corr(preference_score, raw_retrieval_gain): {corr:.6f}")

    if "top_similarity" in diag:
        top_sim = diag["top_similarity"]
        sim_mean = top_sim.reshape(top_sim.shape[0], -1).mean(axis=1)
        print()
        print(summarize("top_similarity", sim_mean))
        if raw_retrieval_gain.std() > 1e-12 and sim_mean.std() > 1e-12:
            corr = np.corrcoef(sim_mean, raw_retrieval_gain)[0, 1]
            print(f"corr(top_similarity, raw_retrieval_gain): {corr:.6f}")

    print()
    if raw_retrieval_gain.mean() <= 0:
        print("Likely bottleneck: retrieval quality. Phase-aware candidates are not better than the host baseline on average.")
    elif enhanced_gain.mean() <= 0:
        print("Likely bottleneck: RPO/retrieval adapter. Raw retrieval has signal, but the enhanced retrieval branch degrades it.")
    elif final_gain.mean() <= 0:
        print("Likely bottleneck: RFRL/adaptive fusion. Retrieval branch has signal, but the final controller uses it poorly.")
    else:
        print("No obvious segment failure: final forecast improves over baseline on average. Inspect slice-level failures next.")


if __name__ == "__main__":
    main()

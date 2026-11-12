"""命令行入口：离线按获知版本重放滤波与 RTS 平滑。

用法（-I 隔离模式下通过项目根目录的薄启动脚本运行）：

    python -I run_estimation.py fixtures/observations.csv fixtures/state.json \
        --output-dir .workspace-output

不连接任何真实仪器或外部模型服务，只读取给定的合成文件。
"""

import argparse
import json
from pathlib import Path

from .config import parse_config
from .errors import StateEstimationError
from .estimator import run_forward_filter
from .observations import parse_observations
from .replay import ReplayCache, select_q_on_training
from .reporting import render_markdown, serialize_version, write_json
from .state_reader import load
from .timeutils import parse_time, to_iso


def _time_label(t, config):
    if config.numeric_time_axis:
        return f"{t:g}"
    return to_iso(t)


def _filter_snapshots(version):
    """提取用于跨版本不可变性比较的深拷贝快照。"""
    return [
        {
            "event_at": step.event_at,
            "observation_ids": tuple(step.observation_ids),
            "posterior_mean": list(step.posterior_mean),
            "posterior_covariance": [row[:] for row in step.posterior_cov],
            "innovation": list(step.innovation),
        }
        for step in version.filter_result.steps
    ]


def _shared_prefix_identical(earlier, later):
    """从轨迹起点按 (event_at, 观测键) 顺序比较共享前缀。

    迟到观测会按 event_at 插入晚版本轨迹中间：插入点之后，晚版本
    的先验包含了新信息，数值理应不同，不参与比较。需要保证的是
    插入点之前的每一步（即早版本已经算出、晚版本在获知迟到样本
    之前同样会算出的部分）逐元素一致——这才是“旧版本不回写、
    未来输入不影响早期滤波”的准确数值含义。
    """
    compared = 0
    for old, new in zip(earlier, later):
        key_old = (old["event_at"], old["observation_ids"])
        key_new = (new["event_at"], new["observation_ids"])
        if key_old != key_new:
            break
        if new["posterior_mean"] != old["posterior_mean"]:
            return False, compared
        if new["posterior_covariance"] != old["posterior_covariance"]:
            return False, compared
        if new["innovation"] != old["innovation"]:
            return False, compared
        compared += 1
    if compared == 0:
        return False, compared
    return True, compared


def run(observations_path, state_path, output_dir):
    raw_rows = load(observations_path)
    raw_config = load(state_path)
    config = parse_config(raw_config)
    observations = parse_observations(raw_rows, config, source_label=str(observations_path))

    if config.training_end is not None:
        selection = select_q_on_training(observations, config, config.training_end)
        q_used = selection.q
        selection_payload = {
            "q": selection.q,
            "training_end": _time_label(selection.training_end, config),
            "training_observation_ids": list(selection.training_observation_ids),
            "grid_scores": [[q_value, score] for q_value, score in selection.grid_scores],
            "note": "q 仅由训练段创新似然选择；验证段及全部 known_at 版本冻结该 q",
        }
    else:
        selection = None
        q_used = config.q
        selection_payload = None

    cache = ReplayCache(observations, config)
    cutoffs = config.known_at_cutoffs or tuple(sorted({obs.received_at for obs in observations}))
    version_payloads = []
    snapshots_by_cutoff = []
    for known_at in cutoffs:
        version = cache.run_version(known_at, q_used)
        version_payloads.append(serialize_version(version, config))
        snapshots_by_cutoff.append((known_at, _filter_snapshots(version)))

    immutability_passed = True
    immutability_details = []
    for (earlier_cutoff, earlier_snaps), (later_cutoff, later_snaps) in zip(
        snapshots_by_cutoff, snapshots_by_cutoff[1:]
    ):
        ok, compared = _shared_prefix_identical(earlier_snaps, later_snaps)
        immutability_passed = immutability_passed and ok
        immutability_details.append(
            {
                "earlier_known_at": _time_label(earlier_cutoff, config),
                "later_known_at": _time_label(later_cutoff, config),
                "shared_prefix_steps_identical": ok,
                "shared_prefix_steps_compared": compared,
                "rule": (
                    "只比较迟到观测插入点之前的共享前缀；插入点之后晚版本"
                    "融合了新信息，允许且应当不同"
                ),
            }
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "inputs": {
            "observations": str(observations_path),
            "state": str(state_path),
            "observation_count_after_dedup": len(observations),
        },
        "time_unit": config.time_unit,
        "q_selection": selection_payload,
        "immutability_check": {
            "passed": immutability_passed,
            "details": immutability_details,
            "rule": "迟到观测只生成新版本；旧版本已算出的滤波结果不回写",
        },
        "versions": [
            {
                **payload,
                "known_at_label": _time_label(payload["time_seconds"], config),
            }
            for payload in version_payloads
        ],
    }
    write_json(output_dir / "summary.json", summary)
    for index, payload in enumerate(version_payloads):
        write_json(output_dir / f"version_{index}.json", payload)
    (output_dir / "report.md").write_text(
        render_markdown(summary, config), encoding="utf-8"
    )
    return summary, selection, q_used, cache, config, observations


def run_hand_check(config, observations):
    """state.json hand_check 指定的解析校验：q=0、单观测、半正定初值。"""
    hand = config.hand_check or {}
    cutoff = hand.get("known_at", config.initial_time)
    known_at = parse_time(cutoff, config.time_unit)
    allowed = set(hand.get("observations", []))
    hand_obs = [
        obs for obs in observations
        if obs.observation_id in allowed and obs.received_at <= known_at
    ]
    if not hand_obs:
        raise StateEstimationError(
            f"hand_check 指定的观测 {sorted(allowed)} 在截止 {cutoff} 前不可得"
        )
    fit = run_forward_filter(hand_obs, config, q=float(hand.get("q", 0.0)))
    step = fit.steps[0]
    return {
        "posterior_position": step.posterior_mean[0],
        "posterior_position_variance": step.posterior_cov[0][0],
        "expected_position": 0.5,
        "expected_variance": 0.5,
        "passed": (
            abs(step.posterior_mean[0] - 0.5) < 1e-12
            and abs(step.posterior_cov[0][0] - 0.5) < 1e-12
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="异步仪器观测的变步长卡尔曼滤波与 RTS 平滑（纯标准库）"
    )
    parser.add_argument("observations", help="观测 CSV/JSONL 文件路径")
    parser.add_argument("state", help="state.json 配置文件路径")
    parser.add_argument(
        "--output-dir",
        default=".workspace-output",
        help="输出目录（默认 .workspace-output，不覆盖输入）",
    )
    args = parser.parse_args(argv)

    summary, selection, q_used, cache, config, observations = run(
        args.observations, args.state, args.output_dir
    )
    hand = run_hand_check(config, observations)
    print(json.dumps(
        {
            "versions": len(summary["versions"]),
            "q_used": q_used,
            "immutability_check_passed": summary["immutability_check"]["passed"],
            "hand_check_passed": hand["passed"],
            "output_dir": args.output_dir,
            "cached_version_keys": len(cache.cached_keys()),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()

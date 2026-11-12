"""把滤波/平滑版本结果序列化为 JSON 与 Markdown 报告。

输出只写到独立的 .workspace-output 目录，不覆盖 fixtures 输入。
报告显式区分“当时可得的滤波估计”和“使用了未来观测的事后平滑”。
"""

import json

from .timeutils import to_iso


def _time_fields(t, config):
    payload = {"time_seconds": t}
    if not config.numeric_time_axis:
        payload["time_iso_utc"] = to_iso(t)
    return payload


def _matrix(m):
    return [list(row) for row in m]


def serialize_version(version, config):
    filter_track = []
    for step in version.filter_result.steps:
        entry = {
            **_time_fields(step.event_at, config),
            "dt_seconds": step.dt,
            "observation_ids": list(step.observation_ids),
            "sensors": list(step.sensors),
            "predicted_mean": list(step.predicted_mean),
            "predicted_covariance": _matrix(step.predicted_cov),
            "posterior_mean": list(step.posterior_mean),
            "posterior_covariance": _matrix(step.posterior_cov),
            "innovation": list(step.innovation),
            "innovation_covariance": _matrix(step.innovation_cov),
            "normalized_residual": list(step.normalized_residual),
            "innovation_log_likelihood": step.log_likelihood,
        }
        filter_track.append(entry)

    smooth_track = []
    for step in version.smooth_steps:
        smooth_track.append(
            {
                **_time_fields(step.event_at, config),
                "observation_ids": list(step.observation_ids),
                "filtered_mean": list(step.filtered_mean),
                "filtered_covariance": _matrix(step.filtered_cov),
                "smoothed_mean": list(step.smoothed_mean),
                "smoothed_covariance": _matrix(step.smoothed_cov),
                "pinv_clipped_eigenvalues": step.pinv_clipped,
                "smooth_minus_filter_mean": [
                    s - f for s, f in zip(step.smoothed_mean, step.filtered_mean)
                ],
            }
        )

    payload = {
        **_time_fields(version.known_at, config),
        "q_used_seconds_unit": version.q_used,
        "observations_known": list(version.observation_ids),
        "forecast_at_cutoff": {
            **_time_fields(version.known_at, config),
            "dt_seconds_from_last_observation": version.forecast_dt,
            "mean": list(version.forecast_mean),
            "covariance": _matrix(version.forecast_cov),
            "note": "无观测更新，仅按真实间隔传播；缺测不补零",
        },
        "filter_track": filter_track,
        "smooth_track": smooth_track,
        "interpretation": (
            "filter_track 只含该 known_at 截止前获知的观测，是当时可得的实时口径；"
            "smooth_track 使用该截止内全部未来观测做 RTS 回看，不得当作实时预测效果"
        ),
    }
    return payload


def write_json(path, payload):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False),
        encoding="utf-8",
    )


def render_markdown(summary, config):
    lines = [
        "# 异步观测状态估计运行报告",
        "",
        "所有仪器、实验与研究身份均为合成虚构数据。时间内部统一为秒。",
        "",
        "## q 选择（仅训练段创新似然）",
        "",
    ]
    selection = summary["q_selection"]
    if selection is None:
        lines.append("未配置 training_end，跳过 q 网格选择，直接使用 state.json 的 q。")
    else:
        lines.append(f"- 训练截止：{selection['training_end']}")
        lines.append(f"- 参与训练的观测：{', '.join(selection['training_observation_ids'])}")
        lines.append("- 网格得分（创新对数似然，越大越好；验证段冻结所选 q）：")
        for q_value, score in selection["grid_scores"]:
            marker = "  <- 选中" if q_value == selection["q"] else ""
            lines.append(f"  - q={q_value:g}: {score:.6f}{marker}")
    lines.extend(["", "## 各获知版本", ""])

    for version in summary["versions"]:
        lines.append(f"### known_at = {version['known_at_label']}")
        lines.append("")
        lines.append(f"- 当时可得观测：{', '.join(version['observations_known']) or '（无）'}")
        lines.append(f"- 使用 q（秒基准）：{version['q_used_seconds_unit']:g}")
        forecast = version["forecast_at_cutoff"]
        lines.append(
            f"- 截止处无观测预报：位置={forecast['mean'][0]:.6f}, "
            f"速度={forecast['mean'][1]:.6f}，"
            f"位置方差={forecast['covariance'][0][0]:.6f}"
        )
        lines.append("- 滤波（实时口径）与 RTS 平滑（事后回看）对比：")
        lines.append("")
        lines.append("| 时刻 | 观测 | 滤波位置 | 平滑位置 | 滤波速度 | 平滑速度 | 平滑-滤波位置差 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for item in version["filter_track"]:
            label = item.get("time_iso_utc", f"{item['time_seconds']:g}")
            smooth = next(
                (
                    candidate
                    for candidate in version["smooth_track"]
                    if candidate["observation_ids"] == item["observation_ids"]
                    and candidate["time_seconds"] == item["time_seconds"]
                ),
                None,
            )
            if smooth is not None:
                delta = smooth["smooth_minus_filter_mean"][0]
                lines.append(
                    f"| {label} | {', '.join(item['observation_ids'])} | "
                    f"{item['posterior_mean'][0]:.6f} | {smooth['smoothed_mean'][0]:.6f} | "
                    f"{item['posterior_mean'][1]:.6f} | {smooth['smoothed_mean'][1]:.6f} | "
                    f"{delta:+.6f} |"
                )
            else:
                lines.append(
                    f"| {label} | {', '.join(item['observation_ids'])} | "
                    f"{item['posterior_mean'][0]:.6f} | - | "
                    f"{item['posterior_mean'][1]:.6f} | - | - |"
                )
        lines.append("")

    lines.extend(
        [
            "## 口径检查",
            "",
            "- 缺测不产生记录、不补零；长空档按真实 dt 传播协方差，无固定步长。",
            "- 同一发生时刻的独立仪器在同一先验上联合更新，dt 只传播一次。",
            "- 迟到观测只生成更新的 known_at 版本，旧版本滤波结果不回写："
            f"{'通过' if summary['immutability_check']['passed'] else '失败'}。",
            "- RTS 平滑使用截止内未来观测，其误差不能代表实时预测能力。",
            "",
        ]
    )
    return "\n".join(lines)

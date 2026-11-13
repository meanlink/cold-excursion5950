"""两条冷链路线的经验分布与温区暴露复核。

提供两种比较口径：
- point（测温点模式）：逐点双样本 KS 检验与路线超温率；
- batch（批次比较模式）：批号在路线内区分，以每批超温点比例为样本，
  各批等权计算路线均值，并对整批重抽样（bootstrap）估计均值及其差的 95% 区间。
"""
import hashlib
import json
import math
import random
import numpy as np
from scipy import stats


BOOTSTRAP_REPLICATES = 2000
MIN_BATCHES_PER_ROUTE = 3
POINT_MODE = "point"
BATCH_MODE = "batch"


def metadata():
    return {"name": "冷链温区偏离调查", "short": "cold-excursion", "tagline": "不只看最高温，核对整条路线的温度分布。",
            "client": "澄源冷运 · 18 名调度员，生物试剂城配线路试运行组", "workflow": "调度把温度记录贴到 Excel，以最高温决定是否调查路线。",
            "pains": ["单个峰值无法说明持续暴露", "不同线路分布不便比较", "阈值调整后需手工重算"], "upstream": "温度记录器导出的 CY_route_temperature.csv", "downstream": "冷运质控经理据分布差异安排包装复核",
            "entities": ["路线温度样本", "温区调查", "质控分发简报"], "dataset_name": "沪东 A / B 路线 · 240 个测温点",
            "columns": [{"key": "sample", "label": "采样点", "type": "string"}, {"key": "route", "label": "路线", "type": "string"}, {"key": "minute", "label": "发车后分钟", "type": "number"}, {"key": "temperature", "label": "温度 °C", "type": "number"}, {"key": "batch", "label": "运输批号（仅批次模式必填）", "type": "string", "optional": True}],
            "parameters": [{"key": "mode", "label": "比较模式", "type": "select", "default": POINT_MODE, "options": [{"value": POINT_MODE, "label": "测温点模式（KS 与超温率）"}, {"value": BATCH_MODE, "label": "批次比较模式（整批重抽样）"}]}, {"key": "upper", "label": "温区上界 °C", "type": "number", "default": 8, "min": 2, "max": 15}, {"key": "alpha", "label": "显著性水平", "type": "select", "default": "0.05", "options": [{"value": "0.01", "label": "0.01"}, {"value": "0.05", "label": "0.05"}, {"value": "0.1", "label": "0.10"}]}],
            "workbench_title": "温度分布比对", "action_label": "执行 KS 与超温复核", "unit": "°C", "accent": "oklch(48% 0.08 184)", "style": "coldroom",
            "seed_story": ["08:00 导入 CY_route_temperature.csv，沪东 A/B 各 120 点。", "08:15 核对采样间隔为 2 分钟。", "08:30 质控将温区上界设为 8°C。", "08:35 执行双样本 KS 比较，打开经验分布曲线。", "08:50 查看两路线超温比例与 P95。", "09:10 将上界设为 7°C，评估更严格口径。", "09:30 导出调查明细，安排 B 线保温包装复核。"],
            "method": "KS 统计量 D=sup|F_A(t)−F_B(t)|，双侧双样本检验；超温率=count(T>上界)/有效采样数；ECDF 按同一温度网格计算。",
            "batch_method": f"批号在路线内区分；以每批超温点比例为样本，各批等权计算路线均值；对整批做有放回重抽样（bootstrap，{BOOTSTRAP_REPLICATES} 次）估计均值及其差的 95% 区间，随机种子由输入记录的规范化哈希派生，同数据重复计算结果一致。",
            "limitations": ["连续采样点可能自相关，KS p 值仅作筛查参考", "比例是采样点比例，未估算货物核心温度", "不能据统计结果直接判定试剂可用性"],
            "batch_limitations": ["批次比例按各批等权平均，不按批内测温点数加权", "bootstrap 区间为近似区间，依赖批次代表性且不纠正批次内自相关", "比例是采样点比例，未估算货物核心温度", "不能据统计结果直接判定试剂可用性"]}


def seed_records():
    rng = random.Random(1002)
    rows = []
    for route, offset in [("沪东 A", 0), ("沪东 B", 1.15)]:
        tag = route[-1]
        for i in range(120):
            temp = 5.4 + offset + .6 * math.sin(i / 11) + rng.gauss(0, .6)
            if route == "沪东 B" and 56 <= i <= 72:
                temp += 1.8
            # 每路线 10 个运输批次，每批 12 点；批号仅在路线内唯一。
            rows.append({"sample": f"{tag}-{i+1:03}", "route": route, "minute": i * 2,
                         "temperature": round(temp, 3), "batch": f"{tag}-B{i // 12 + 1:02}"})
    return rows


def _parse_params(params):
    try:
        upper, alpha = float(params.get("upper", 8)), float(params.get("alpha", .05))
    except (ValueError, TypeError):
        raise ValueError("温区和显著性水平须为数值")
    if not math.isfinite(upper) or not 2 <= upper <= 15 or alpha not in (.01, .05, .1):
        raise ValueError("上界范围 2–15°C，显著性水平只接受 0.01/0.05/0.1")
    mode = str(params.get("mode", POINT_MODE))
    if mode not in (POINT_MODE, BATCH_MODE):
        raise ValueError("比较模式只接受 point（测温点模式）或 batch（批次比较模式）")
    return mode, upper, alpha


def analyze(records, params):
    mode, upper, alpha = _parse_params(params)
    if mode == BATCH_MODE:
        return _analyze_batches(records, upper)
    return _analyze_points(records, upper, alpha)


def _analyze_points(records, upper, alpha):
    grouped = {}
    for r in records:
        try:
            name, val = str(r["route"]), float(r["temperature"])
            if not math.isfinite(val):
                raise ValueError()
            grouped.setdefault(name, []).append(val)
        except (TypeError, ValueError, KeyError):
            raise ValueError("每点须有路线和有限温度数值")
    if len(grouped) != 2 or any(len(v) < 10 for v in grouped.values()):
        raise ValueError("恰需两条路线，每条至少 10 个测温点")
    names = sorted(grouped)
    a, b = [np.array(grouped[n]) for n in names]
    ks = stats.ks_2samp(a, b, alternative="two-sided", method="auto")
    grid = np.linspace(min(float(a.min()), float(b.min())), max(float(a.max()), float(b.max())), 25)
    rows = [{"route": n, "count": len(grouped[n]), "mean": round(float(np.mean(grouped[n])), 3), "p95": round(float(np.quantile(grouped[n], .95)), 3), "over": int(sum(t > upper for t in grouped[n])), "rate": round(sum(t > upper for t in grouped[n]) / len(grouped[n]) * 100, 3)} for n in names]
    verdict = "存在分布差异" if float(ks.pvalue) < alpha else "未检出分布差异"
    return {"summary": f"{names[0]} 与 {names[1]} 的 KS D={ks.statistic:.3f}，在 α={alpha:g} 下{verdict}。",
            "metrics": [{"label": "KS 距离 D", "value": round(float(ks.statistic), 5)}, {"label": "双侧 p 值", "value": float(f"{ks.pvalue:.6g}")}, {"label": "路线 B 超温率", "value": rows[1]["rate"], "unit": "%"}, {"label": "温区上界", "value": upper, "unit": "°C"}],
            "table": {"columns": [{"key": "route", "label": "路线"}, {"key": "count", "label": "采样点"}, {"key": "mean", "label": "均温 °C"}, {"key": "p95", "label": "P95 °C"}, {"key": "over", "label": "超温点"}, {"key": "rate", "label": "超温率 %"}], "rows": rows},
            "chart": {"type": "line", "labels": [f"{x:.2f}°C" for x in grid], "series": [{"name": n + " 累积比例%", "values": [round(sum(v <= x for v in grouped[n]) / len(grouped[n]) * 100, 3) for x in grid]} for n in names]},
            "findings": [f"{n} 超温点 {r['over']}/{r['count']}，比例 {r['rate']:.2f}%。" for n, r in zip(names, rows)] + ["统计差异仅定位调查对象；请结合门开记录和独立运输批次复核。"],
            "method": metadata()["method"], "warnings": ["同一路线的连续采样存在自相关，不能把 p 值作为独立批次证据。"]}


def _collect_batches(records, upper):
    """按路线 → 批号归集每批超温点与比例；批号只在路线内区分。"""
    grouped = {}
    for r in records:
        try:
            name = str(r["route"])
            val = float(r["temperature"])
            if not math.isfinite(val):
                raise ValueError()
        except (TypeError, ValueError, KeyError):
            raise ValueError("每点须有路线和有限温度数值")
        if r.get("batch") is None or str(r.get("batch")).strip() == "":
            raise ValueError("批次比较模式要求每条记录都有运输批号；请改用测温点模式，或补齐批号后重试")
        batch = str(r["batch"]).strip()
        grouped.setdefault(name, {}).setdefault(batch, []).append(1.0 if val > upper else 0.0)
    if len(grouped) != 2:
        raise ValueError("恰需两条路线才能比较；当前识别到 %d 条路线" % len(grouped))
    short = {n: len(b) for n, b in grouped.items() if len(b) < MIN_BATCHES_PER_ROUTE}
    if short:
        detail = "，".join(f"{n} 仅 {c} 批" for n, c in sorted(short.items()))
        raise ValueError(f"批次比较模式要求每条路线至少 {MIN_BATCHES_PER_ROUTE} 个运输批次：{detail}")
    # 批号按名称稳定排序，避免输入乱序影响结果。
    result = {}
    for n, batches in grouped.items():
        result[n] = {}
        for b in sorted(batches):
            flags = batches[b]
            over = int(sum(flags))
            result[n][b] = {"count": len(flags), "over": over, "rate": over / len(flags) * 100.0}
    return result


def _deterministic_seed(grouped):
    """由与输入顺序无关的规范化批次内容派生稳定种子。

    归集结果按路线名、批号名排序后再哈希，因此同数据乱序或重复计算
    都会得到相同的 bootstrap 随机序列。
    """
    canonical = {n: {b: {"count": grouped[n][b]["count"], "over": grouped[n][b]["over"]}
                     for b in sorted(grouped[n])} for n in sorted(grouped)}
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16)


def _analyze_batches(records, upper):
    grouped = _collect_batches(records, upper)
    names = sorted(grouped)
    name_a, name_b = names
    props = {n: np.array([grouped[n][b]["rate"] for b in sorted(grouped[n])], dtype=float) for n in names}
    means = {n: float(np.mean(props[n])) for n in names}

    seed = _deterministic_seed(grouped)
    rng = np.random.default_rng(seed)
    boot_means = {}
    for n in names:
        draws = rng.integers(0, len(props[n]), size=(BOOTSTRAP_REPLICATES, len(props[n])))
        boot_means[n] = props[n][draws].mean(axis=1)
    boot_diff = boot_means[name_a] - boot_means[name_b]

    def ci(samples):
        low, high = np.quantile(samples, [0.025, 0.975])
        return round(float(low), 3), round(float(high), 3)

    ci_a, ci_b, ci_d = ci(boot_means[name_a]), ci(boot_means[name_b]), ci(boot_diff)
    diff = round(means[name_a] - means[name_b], 3)

    table_rows = []
    for n in names:
        for b in sorted(grouped[n]):
            item = grouped[n][b]
            table_rows.append({"route": n, "batch": b, "count": item["count"],
                               "over": item["over"], "rate": round(item["rate"], 3)})

    max_batches = max(len(grouped[n]) for n in names)
    chart = {"type": "bar",
             "labels": [f"批 {i + 1}" for i in range(max_batches)],
             "series": [{"name": f"{n} 各批超温率%",
                         "values": [round(grouped[n][b]["rate"], 3) for b in sorted(grouped[n])]
                                   + [None] * (max_batches - len(grouped[n]))} for n in names]}

    return {"summary": (f"{name_a} 与 {name_b} 的批次等权超温率分别为 {means[name_a]:.2f}% 与 {means[name_b]:.2f}%，"
                        f"差值 {diff:+.2f} 个百分点，95% 区间 [{ci_d[0]:.2f}, {ci_d[1]:.2f}]。"),
            "metrics": [
                {"label": f"{name_a} 批均值超温率", "value": round(means[name_a], 3), "unit": "%"},
                {"label": f"{name_a} 均值 95% 区间", "value": f"[{ci_a[0]}, {ci_a[1]}]", "unit": "%"},
                {"label": f"{name_b} 批均值超温率", "value": round(means[name_b], 3), "unit": "%"},
                {"label": f"{name_b} 均值 95% 区间", "value": f"[{ci_b[0]}, {ci_b[1]}]", "unit": "%"},
                {"label": "均值差（A−B）95% 区间", "value": f"[{ci_d[0]}, {ci_d[1]}]", "unit": "百分点"},
                {"label": "温区上界", "value": upper, "unit": "°C"}],
            "table": {"columns": [{"key": "route", "label": "路线"}, {"key": "batch", "label": "运输批号"},
                                  {"key": "count", "label": "批内采样点"}, {"key": "over", "label": "超温点"},
                                  {"key": "rate", "label": "该批超温点比例 %"}],
                      "rows": table_rows},
            "chart": chart,
            "findings": [f"{n} 共 {len(grouped[n])} 个运输批次，各批等权平均超温率 {means[n]:.2f}%，"
                         f"bootstrap 95% 区间 [{ci(boot_means[n])[0]:.2f}, {ci(boot_means[n])[1]:.2f}]%。" for n in names]
                      + [f"两路线批均值差（{name_a}−{name_b}）为 {diff:+.2f} 个百分点，95% 区间 "
                         f"[{ci_d[0]:.2f}, {ci_d[1]:.2f}]；区间不含 0 提示批次层面存在稳定差异。"],
            "method": metadata()["batch_method"],
            "warnings": [f"以每批超温点比例为样本、各批等权；整批重抽样 {BOOTSTRAP_REPLICATES} 次，"
                         "种子由输入内容哈希派生，乱序或重复计算结果一致。"],
            "limitations": metadata()["batch_limitations"]}

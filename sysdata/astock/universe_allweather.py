"""
AStockAllWeatherUniverse — 6-bucket All-Weather ETF universe for A-share.

Buckets follow the all-weather risk-premium decomposition:
  1. Equity Large       — broad equity beta (HS300)
  2. Equity Mid          — size / mid-cap exposure (CSI500)
  3. Dividend / Quality  — defensive yield / quality (上证红利)
  4. Rates / Duration    — government bond duration (国债 ETF)
  5. Real Assets         — gold / inflation hedge (黄金 ETF)
  6. Cash / Carry        — money-market liquidity (货币 ETF)

Each bucket has a primary proxy and one or two backups (for data-history
fallback). All proxies use pysystemtrade A-share code format: <code>.SH/.SZ.

Reference: docs/allweather_tech_design.md §Appendix A
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ─────────────────────────────────────────────────────────────────────────
# Bucket identifiers (stable enum-like strings — used in configs & reports)
# ─────────────────────────────────────────────────────────────────────────
BUCKET_EQUITY_LARGE = "equity_large"
BUCKET_EQUITY_MID = "equity_mid"
BUCKET_DIVIDEND = "dividend"
BUCKET_RATES = "rates"
BUCKET_GOLD = "gold"
BUCKET_CASH = "cash"

ALL_BUCKETS: Tuple[str, ...] = (
    BUCKET_EQUITY_LARGE,
    BUCKET_EQUITY_MID,
    BUCKET_DIVIDEND,
    BUCKET_RATES,
    BUCKET_GOLD,
    BUCKET_CASH,
)


@dataclass(frozen=True)
class BucketSpec:
    """Specification of one all-weather risk-premium bucket."""
    bucket: str                    # one of ALL_BUCKETS
    risk_premium: str              # human-readable risk premium label
    primary: str                   # primary ETF code (e.g. "510300.SH")
    backups: Tuple[str, ...] = ()  # fallback ETFs if primary lacks history
    listing_year: int = 2013       # approximate trading-history start
    notes: str = ""

    @property
    def all_proxies(self) -> Tuple[str, ...]:
        return (self.primary,) + tuple(self.backups)


# ─────────────────────────────────────────────────────────────────────────
# Bucket definitions (Phase 1 v1)
# ─────────────────────────────────────────────────────────────────────────
BUCKET_SPECS: Dict[str, BucketSpec] = {
    BUCKET_EQUITY_LARGE: BucketSpec(
        bucket=BUCKET_EQUITY_LARGE,
        risk_premium="Equity beta (large-cap)",
        primary="510300.SH",   # 沪深300ETF (华泰柏瑞)
        backups=("510050.SH",),  # 上证50ETF
        listing_year=2012,
        notes="Broad China A-share large-cap equity exposure.",
    ),
    BUCKET_EQUITY_MID: BucketSpec(
        bucket=BUCKET_EQUITY_MID,
        risk_premium="Equity / size premium",
        primary="510500.SH",   # 中证500ETF
        backups=("512100.SH",),  # 中证1000ETF
        listing_year=2013,
        notes="Mid-cap / size factor exposure, complements large-cap.",
    ),
    BUCKET_DIVIDEND: BucketSpec(
        bucket=BUCKET_DIVIDEND,
        risk_premium="Quality / dividend yield",
        primary="510880.SH",   # 上证红利ETF
        backups=("512890.SH",),  # 红利低波ETF
        listing_year=2011,
        notes="Defensive yield + quality factor; lower drawdown profile.",
    ),
    BUCKET_RATES: BucketSpec(
        bucket=BUCKET_RATES,
        risk_premium="Duration (government bond)",
        primary="511010.SH",   # 国债ETF (5Y)
        backups=("511260.SH",),  # 10年国债ETF
        listing_year=2013,
        notes="Sovereign-rate duration exposure; offsets equity in growth shocks.",
    ),
    BUCKET_GOLD: BucketSpec(
        bucket=BUCKET_GOLD,
        risk_premium="Real assets / inflation hedge",
        primary="518880.SH",   # 黄金ETF (华安)
        backups=("159934.SZ",),  # 黄金ETF (易方达)
        listing_year=2013,
        notes="Tail hedge against CNY weakness / inflation / geopolitics.",
    ),
    BUCKET_CASH: BucketSpec(
        bucket=BUCKET_CASH,
        risk_premium="Cash / carry / liquidity",
        primary="511880.SH",   # 银华日利
        backups=("511990.SH",),  # 华宝添益
        listing_year=2013,
        notes="Money-market ETF; absorbs unallocated risk budget.",
    ),
}


# ─────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────
def get_bucket_spec(bucket: str) -> BucketSpec:
    """Return spec for a bucket label."""
    if bucket not in BUCKET_SPECS:
        raise KeyError(f"Unknown all-weather bucket: {bucket!r}. "
                       f"Valid: {ALL_BUCKETS}")
    return BUCKET_SPECS[bucket]


def primary_universe() -> List[str]:
    """List of primary ETF codes (one per bucket) — the default universe."""
    return [spec.primary for spec in BUCKET_SPECS.values()]


def full_universe() -> List[str]:
    """Primary + all backup proxies (for data-availability planning)."""
    out: List[str] = []
    for spec in BUCKET_SPECS.values():
        out.extend(spec.all_proxies)
    # de-dup while preserving order
    seen = set()
    deduped = []
    for s in out:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped


def bucket_of(symbol: str) -> str:
    """Inverse lookup: which bucket does this ETF belong to?"""
    for spec in BUCKET_SPECS.values():
        if symbol in spec.all_proxies:
            return spec.bucket
    raise KeyError(f"Symbol {symbol!r} not in any all-weather bucket.")


def symbol_to_bucket_map() -> Dict[str, str]:
    """Flat mapping {symbol: bucket} for all proxies (primary + backups)."""
    return {sym: spec.bucket
            for spec in BUCKET_SPECS.values()
            for sym in spec.all_proxies}


def describe() -> str:
    """Human-readable summary string (used by CLI / reports)."""
    lines = ["A-Share All-Weather Universe (6 buckets)"]
    lines.append("=" * 52)
    for spec in BUCKET_SPECS.values():
        backups = ", ".join(spec.backups) if spec.backups else "—"
        lines.append(
            f"  [{spec.bucket:<13}] {spec.primary}  "
            f"backup={backups}  ({spec.risk_premium})"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
    print(f"\nPrimary universe ({len(primary_universe())}): {primary_universe()}")
    print(f"Full universe   ({len(full_universe())}): {full_universe()}")

"""Frozen exploratory natural-variant design and fail-closed study transitions."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class VariantProtocol:
    version: str = "natural-variant-js-v1"
    seed: int = 1731
    query_radius_bp: int = 32
    sham_radius_bp: int = 64
    distance_tolerance_bp: int = 16
    gc_tolerance: float = .05
    sham_pwm_tolerance_bits: float = .1
    min_clusters: int = 8
    min_retention: float = .5
    min_effect_nats: float = .001
    min_binding_rho: float = .3
    bootstrap_samples: int = 2000
    numerical_tolerance: float = 2e-4
    meaningful_rescue_nats: float = .001

    def __post_init__(self):
        if min(self.query_radius_bp, self.sham_radius_bp, self.bootstrap_samples) < 1:
            raise ValueError("Radii and resampling counts must be positive")
        if self.min_clusters < 3 or self.distance_tolerance_bp < 0:
            raise ValueError("Invalid independence or geometry rule")
        for name in ("gc_tolerance", "sham_pwm_tolerance_bits", "min_effect_nats",
                     "numerical_tolerance", "meaningful_rescue_nats"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"Invalid {name}")
        if not 0 < self.min_retention <= 1 or not -1 <= self.min_binding_rho <= 1:
            raise ValueError("Invalid retention or correlation threshold")

    def record(self):
        return dict(**asdict(self), status="exploratory feasibility; not preregistered confirmation",
            endpoint="nucleotide-width-weighted local masked prediction-distribution Jensen-Shannon divergence",
            queries="all unchanged real tokens within radius, outside reference motif hits and variant-to-sham interval; each masked separately; both edits visible",
            primary_pilot_effect="native variant divergence minus substitution-matched sham divergence, nats",
            biological_check="Spearman correlation of variant divergence with absolute input-normalized binding log odds",
            direction_scope="symmetric sensitivity; does not predict which allele binds more strongly",
            alignment="identical offsets at every token across reference, alternate and sham",
            sham="same forward-strand substitution and trinucleotide; outside all reference PWM hits; PWM preservation",
            inference="equal genomic-cluster means; cluster bootstrap; magnitude association secondary feasibility gate",
            head_selection="only after all feasibility gates: largest discovery mean absolute rescue minus sham rescue; layer/head tie break",
            confirmation="disabled until fresh membership, selected head, intervention variance, power and immutable protocol verified",
            stop="any failed gate stops head search and confirmation; no automatic endpoint/control relaxation")


def feasibility_gate(summary, protocol):
    reasons = []
    required = ["controls_passed", "retention", "clusters", "mean", "ci_low",
                "binding_rho", "binding_ci_low"]
    if any(k not in summary for k in required):
        return dict(status="stop", reasons=["incomplete feasibility evidence"], selected_head=None)
    if not summary["controls_passed"]:
        reasons.append("implementation controls not demonstrated" if summary["controls_passed"] is None else "implementation controls failed")
    if summary["retention"] < protocol.min_retention:
        reasons.append("insufficient sequence-only retention")
    if summary["clusters"] < protocol.min_clusters:
        reasons.append("insufficient genomic clusters")
    checks = [("mean", protocol.min_effect_nats), ("ci_low", 0),
              ("binding_rho", protocol.min_binding_rho), ("binding_ci_low", 0)]
    for key, threshold in checks:
        value = summary[key]
        if value is None or not math.isfinite(value) or value <= threshold:
            reasons.append(f"{key} fails frozen sensitivity threshold")
    return dict(status="eligible_for_discovery" if not reasons else "stop", reasons=reasons,
                selected_head=None, confirmation="not authorized by feasibility alone")


def confirmation_readiness(evidence):
    """Missing, draft or stale evidence must never enable confirmation."""
    required = ("pilot_passed", "head_frozen", "endpoint_frozen", "controls_frozen",
                "membership_frozen", "freshness_verified", "genomic_separation_verified",
                "donor_dependence_addressed", "biological_qc_verified", "power_passed",
                "protocol_hash_verified")
    missing = [k for k in required if evidence.get(k) is not True]
    return dict(ready=not missing, missing=missing)

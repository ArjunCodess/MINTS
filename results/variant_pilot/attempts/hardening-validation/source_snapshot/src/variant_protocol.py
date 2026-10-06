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
        for name in ("seed", "query_radius_bp", "sham_radius_bp", "distance_tolerance_bp", "min_clusters", "bootstrap_samples"):
            if type(getattr(self,name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.seed < 0 or self.bootstrap_samples < 100:
            raise ValueError("Nonnegative seed and at least 100 bootstrap draws required")
        if min(self.query_radius_bp, self.sham_radius_bp, self.bootstrap_samples) < 1:
            raise ValueError("Radii and resampling counts must be positive")
        if self.min_clusters < 3 or self.distance_tolerance_bp < 0:
            raise ValueError("Invalid independence or geometry rule")
        for name in ("gc_tolerance", "sham_pwm_tolerance_bits", "min_effect_nats",
                     "numerical_tolerance", "meaningful_rescue_nats", "min_retention", "min_binding_rho"):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError(f"Invalid {name}")
            if name!="min_binding_rho" and value<=0:
                raise ValueError(f"Invalid {name}")
        if not 0 < self.min_retention <= 1 or not 0 <= self.min_binding_rho <= 1:
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
    def number(value):
        return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
    if summary["controls_passed"] is not True:
        reasons.append("implementation controls not demonstrated" if summary["controls_passed"] is None else "implementation controls failed")
    if not number(summary["retention"]) or not protocol.min_retention <= summary["retention"] <= 1:
        reasons.append("insufficient sequence-only retention")
    if type(summary["clusters"]) is not int or summary["clusters"] < protocol.min_clusters:
        reasons.append("insufficient genomic clusters")
    checks = [("mean", protocol.min_effect_nats), ("ci_low", 0),
              ("binding_rho", protocol.min_binding_rho), ("binding_ci_low", 0)]
    for key, threshold in checks:
        value = summary[key]
        if not number(value) or value <= threshold:
            reasons.append(f"{key} fails frozen sensitivity threshold")
    if number(summary["binding_rho"]) and not -1 <= summary["binding_rho"] <= 1:
        reasons.append("binding correlation outside valid range")
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

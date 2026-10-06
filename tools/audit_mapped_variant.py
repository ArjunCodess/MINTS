"""Cache-independent byte and semantic audit of the mapped-query study."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import math
import numpy as np
import pandas as pd
from src.assay_stats import genomic_clusters
from src.utils import sha256_file
from src.variant_statistics import cluster_summary

ROOT=Path(__file__).resolve().parents[1]


def audit(root=ROOT,raw=False):
    diagnostic=root/"results/control_diagnostics"
    study=root/"results/mapped_variant"
    for directory in (diagnostic,study):
        receipt=json.loads((directory/"execution.json").read_text())
        assert receipt["status"]=="completed" and not receipt["source_changed_during_run"]
        for name,digest in receipt["artifacts"].items():
            assert sha256_file(directory/name)==digest,(directory,name)
        assert sha256_file(directory/"protocol.json")==receipt["protocol_sha256"]
        if directory==study:
            for name,digest in receipt["source_sha256"].items():
                assert sha256_file(study/"source_snapshot"/name)==digest,name
    source=pd.read_csv(root/"results/adastra_exploratory/candidates.csv.gz")
    from src.adastra_candidates import select_candidates
    expected=select_candidates(source,12288)
    membership=pd.read_csv(study/"membership.csv")
    assert len(source)==512556 and len(membership)==12288
    pd.testing.assert_frame_equal(expected,membership.drop(columns="cohort"))
    diag=pd.read_csv(diagnostic/"cases.csv")
    preds=pd.read_csv(diagnostic/"candidate_predicates.csv.gz")
    eligible=pd.read_csv(study/"eligibility.csv")
    assert membership.variant_id.tolist()==diag.variant_id.tolist()==eligible.variant_id.tolist()
    summary=json.loads((study/"summary.json").read_text())
    assert summary["reference_verified"]==int(eligible.reference_verified.sum())==12288
    assert summary["motif_eligible"]==int(eligible.motif_eligible.sum())==433
    assert summary["correspondence_eligible"]==int(eligible.correspondence_eligible.sum())==433
    assert summary["matched_controls"]==int(eligible.matched_control.sum())==61
    assert diag.loc[diag.cohort=="original4096","original"].sum()==0
    assert diag.loc[diag.cohort=="expansion","original"].sum()==1
    for cohort,group in diag.groupby("cohort"):
        saved=json.loads((diagnostic/"summary.json").read_text())["cohorts"][cohort]
        assert saved["records"]==len(group)
        for key,value in saved.items():
            if key!="records":assert int(group[key].sum())==value,(cohort,key)
    for key in ("original","local_only","local_no_width","local_geometry32","local_geometry64"):
        retained=set(preds.loc[preds[key],"variant_id"])
        assert retained==set(diag.loc[diag[key],"variant_id"]),key
    cases=json.loads((study/"cases.json").read_text())
    scores=json.loads((study/"scores.json").read_text())
    assert [c["variant_id"] for c in cases]==[s["variant_id"] for s in scores]
    assert len(cases)==61
    frame=pd.read_csv(study/"case_scores.csv")
    for c,s in zip(cases,scores,strict=True):
        clean,alt,sham=(c[k] for k in ("reference_sequence","alternate_sequence","sham_sequence"))
        i,j=c["variant_index"],c["sham_index"]
        assert i==102 and len(clean)==len(alt)==len(sham)==204
        assert c["position_1based"]-1-c["window_start_0based"]==i
        assert c["window_end_0based"]-c["window_start_0based"]==204
        assert [k for k,(a,b) in enumerate(zip(clean,alt)) if a!=b]==[i]
        assert [k for k,(a,b) in enumerate(zip(clean,sham)) if a!=b]==[j]
        assert clean[i]==clean[j]==c["reference"] and alt[i]==sham[j]==c["alternate"]
        assert clean[i-1:i+2]==clean[j-1:j+2] and abs(i-j)<=16
        frozen=preds[(preds.variant_id==c["variant_id"]) & preds.local_no_width].sort_values(["distance_bp","sham_index"])
        assert int(frozen.iloc[0].sham_index)==j
        assert c["queries"]==[r["query"] for r in s["query_diagnostics"]]
        for q,r in zip(c["queries"],s["query_diagnostics"],strict=True):
            a,b=q["span"]
            assert b-a==q["width"] and b>a
            assert b<=min(i,j) or a>max(i,j)
            assert len(q["indices"])==3 and len(set(x[a:b] for x in (clean,alt,sham)))==1
            assert not(a<c["motif_span"][1] and b>c["motif_span"][0])
            masked=[]
            for ids,k in zip(c["ids"],q["indices"],strict=True):
                assert ids[k]==q["token_id"]
                # Pinned DNABERT-2 [MASK] is verified in the model receipt/tokenizer.
                value=ids.copy();value[k]=-1;masked.append(tuple(value))
            assert len(set(masked))==3
            assert max(r["identity_errors"]+r["final_restoration_errors"]+r["repeat_errors"])<=0.0002
            assert r["normalization_error"]<=1e-12
            assert all(math.isfinite(r[k]) and 0<=r[k]<=math.log(2)+1e-12 for k in ("variant_js","sham_js"))
        weights=[q["width"] for q in c["queries"]]
        v=float(np.average([r["variant_js"] for r in s["query_diagnostics"]],weights=weights))
        h=float(np.average([r["sham_js"] for r in s["query_diagnostics"]],weights=weights))
        assert abs(v-s["variant_divergence"])<1e-12 and abs(h-s["sham_divergence"])<1e-12
        assert abs(v-h-s["contrast"])<1e-12 and s["implementation_valid"]
        row=frame[frame.variant_id==c["variant_id"]].iloc[0]
        assert abs(row.contrast-s["contrast"])<1e-12
        if raw:
            path=root/s["raw_logits_path"]
            assert sha256_file(path)==s["raw_logits_sha256"]
            import torch
            from src.variant_assay import js_divergence
            values=np.load(path)
            assert np.isfinite(values["logits"]).all()
            for logits,r in zip(values["logits"],s["query_diagnostics"],strict=True):
                assert abs(js_divergence(torch.tensor(logits[0]),torch.tensor(logits[1]))-r["variant_js"])<1e-12
                assert abs(js_divergence(torch.tensor(logits[0]),torch.tensor(logits[2]))-r["sham_js"])<1e-12
    names=np.array([[f"{c['chrom']}:{c['window_start_0based']}-{c['window_end_0based']}"]*3 for c in cases])
    sequences=np.array([[c[k] for k in ("reference_sequence","alternate_sequence","sham_sequence")] for c in cases])
    clusters=genomic_clusters(names,block_bp=1000000,sequences=sequences)
    assert clusters.tolist()==[c["cluster"] for c in cases]==frame.cluster.tolist()
    primary=cluster_summary(frame.contrast,frame.cluster,repetitions=10000)
    for k in primary:assert primary[k]==summary["primary"][k] or abs(primary[k]-summary["primary"][k])<1e-12
    assert summary["genomic_proxy_clusters"]==len(set(clusters))==60
    assert summary["implementation_valid"]==61
    assert summary["full_boundary_stable_cases"]==int(frame.full_boundary_stable.sum())==15
    assert summary["full_boundary_stable_clusters"]==frame[frame.full_boundary_stable].cluster.nunique()==15
    assert summary["width_matched_cases"]==int(frame.edit_width_matched.sum())==11
    assert summary["query_index_shift_cases"]==int(frame.query_index_shift.sum())==29
    width=frame[frame.edit_width_matched]
    width_summary=cluster_summary(width.contrast,width.cluster,repetitions=10000)
    for k in width_summary:assert abs(width_summary[k]-summary["width_matched"][k])<1e-12
    assert summary["gates"]["sequence_feasibility"] and summary["gates"]["implementation_valid"]
    assert summary["gates"]["tokenization_isolation"] and not summary["gates"]["biological_qc"]
    assert summary["gates"]["native_sensitivity"]==(primary["ci_low"]>0)
    assert not summary["gates"]["head_search"] and not summary["gates"]["confirmation"]
    assert summary["biological_qc_cases"] is None and summary["independent_donors"] is None
    ledger=json.loads((study/"inspection_ledger.json").read_text())
    assert ledger["native_outcomes_opened"] and not ledger["confirmation_assigned"]
    assert ledger["membership_sha256"]==sha256_file(study/"membership.csv")
    return dict(status="verified",selected=12288,scored=61,clusters=60,raw_logits_checked=raw,
                native_gate="failed",head_search="not run",confirmation="not run")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--raw",action="store_true")
    p.add_argument("--root",type=Path,default=ROOT);a=p.parse_args()
    print(json.dumps(audit(a.root,a.raw),indent=2))

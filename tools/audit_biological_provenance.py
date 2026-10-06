"""Explore live experiment links for frozen sequence cases, without effect selection."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import time
import requests
import pandas as pd
from src.utils import write_json,sha256_file

ROOT=Path(__file__).resolve().parents[1]


def main():
    cache=ROOT/"data/adastra/provenance_v1/snp_details"
    cache.mkdir(parents=True,exist_ok=False)
    output=ROOT/"results/variant_provenance"
    cases=json.loads((ROOT/"results/mapped_variant/cases.json").read_text())
    write_json(output/"inspection_ledger.json",dict(role="post-native exploratory provenance only",
        cases_sha256=sha256_file(ROOT/"results/mapped_variant/cases.json"),live_release="v6.1.1; not substituted for v6.1 denominator",
        outcomes_opened=True,outcome_use="inspect available measurement/provenance fields; no binding correlation or membership selection",
        confirmation_eligible=False))
    def get(c):
        url=f"https://adastra.autosome.org/api/v6/snps/{c['rs_id'].removeprefix('rs')}/{c['alternate']}"
        record=dict(variant_id=c["variant_id"],url=url,access_utc=pd.Timestamp.now(tz="UTC").isoformat())
        try:
            r=requests.get(url,timeout=(15,35));record["http_status"]=r.status_code;r.raise_for_status()
            j=r.json()
            if (str(j["chromosome"]).removeprefix("chr"),j["position"],j["ref"],j["alt"])!=(c["chrom"].removeprefix("chr"),c["position_1based"],c["reference"],c["alternate"]):
                raise ValueError("Live SNP coordinate/allele mismatch")
            path=cache/(c["rs_id"]+"_"+c["alternate"]+".json");path.write_bytes(r.content)
            record.update(status="verified coordinates",bytes=len(r.content),sha256=hashlib.sha256(r.content).hexdigest())
            observations={}
            for aggregate in j.get("tf_aggregated_snps",[])+j.get("cl_aggregated_snps",[]):
                for e in aggregate.get("exp_snps",[]):
                    exp=e.get("experiment",{})
                    if exp.get("tf_name")=="CTCF_HUMAN":
                        observations[e["exp_snp_id"]]=dict(variant_id=c["variant_id"],exp_snp_id=e["exp_snp_id"],
                            experiment=exp.get("exp_id"),alignment=exp.get("align"),cell_type=exp.get("cl_name"),
                            ref_readcount=e.get("ref_readcount"),alt_readcount=e.get("alt_readcount"),bad=e.get("bad"))
            record["available_ctcf_observations"]=len(observations)
            return record,list(observations.values())
        except Exception as exc:
            record.update(status="failed",reason=f"{type(exc).__name__}: {exc}")
            return record,[]
    records,observations=[],[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for number,(receipt,rows) in enumerate(pool.map(get,cases),1):
            records.append(receipt);observations.extend(rows)
            if number%10==0:print(f"provenance {number}/{len(cases)}",flush=True)
    write_json(output/"snp_downloads.json",records)
    table=pd.read_csv(ROOT/"data/adastra/provenance_v1/GTRD_experiments.tsv",sep="\t",dtype=str,keep_default_na=False)
    ctcf=table[table.TF_UNIPROT_ID=="CTCF_HUMAN"]
    ctcf.to_csv(output/"ctcf_experiments.tsv",sep="\t",index=False,lineterminator="\n")
    frame=pd.DataFrame(observations)
    merged=frame.merge(table,left_on="experiment",right_on="EXP",how="left",validate="many_to_one")
    merged.to_csv(output/"case_experiment_counts.csv",index=False,lineterminator="\n")
    # Shared observed experiments connect cases. Counts are distinct coverage
    # observations, not biological replicates or independently measured donors.
    parents={c["variant_id"]:c["variant_id"] for c in cases}
    def find(x):
        while parents[x]!=x:
            parents[x]=parents[parents[x]];x=parents[x]
        return x
    for _,group in merged.groupby("experiment"):
        ids=list(group.variant_id.unique())
        for v in ids[1:]:parents[find(v)]=find(ids[0])
    write_json(output/"provenance_summary.json",dict(queried_cases=len(cases),downloaded=sum(r["status"]=="verified coordinates" for r in records),
        failed=sum(r["status"]=="failed" for r in records),cases_with_available_ctcf_counts=int(frame.variant_id.nunique()),
        ctcf_observations=len(frame),ctcf_experiments=int(frame.experiment.nunique()),
        metadata_linked_observations=int(merged.EXP.notna().sum()),metadata_ctcf_experiments=len(ctcf),
        source_publications_or_series=int(merged.GEO_GSE.replace("",pd.NA).nunique()),
        observed_shared_experiment_components=len({find(v) for v in frame.variant_id.unique()}),
        interpretation="live API exposes some ChIP allele counts and experiment/BAD links; coverage and significance filtering means absent observations are unknown, not zero",
        qc="individual genotype/phase, donor identities, matched per-replicate input counts and mapping correction remain unverified",
        live_version="v6.1.1; post-score exploratory provenance; no outcome correlation or frozen archive replacement"))
    print(json.dumps(json.loads((output/"provenance_summary.json").read_text()),indent=2))


if __name__=="__main__":main()

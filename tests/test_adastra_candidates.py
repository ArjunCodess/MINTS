"""Import coordinates faithfully and select without binding outcome leakage."""
from dataclasses import replace
import hashlib

import pandas as pd
import pytest

from src.adastra_candidates import import_candidates,select_candidates,allele_window,sequence_control
from src.controlled_edits import MotifDefinition
from src.variant_assay import prepare_variant
from src.variant_protocol import VariantProtocol
from tools.download_adastra import verify_archive


def source():
    return pd.DataFrame([dict(chr="chr1",start="100",end="101",ID="rs1",ref="A",alt="G",es_mean_ref="0.5"),
                         dict(chr="2",start="200",end="201",ID="rs2",ref="C",alt="T",es_mean_ref="0.1")])


@pytest.mark.parametrize("key,value",[("start","100.5"),("end","100"),("ref","AA"),
                                     ("alt","A"),("alt","N"),("chr","chr0")])
def test_invalid_source_is_not_silently_repaired(key,value):
    raw=source();raw.loc[0,key]=value
    with pytest.raises(ValueError):import_candidates(raw)


def test_import_preserves_full_denominator_and_bed_conversion():
    rows=import_candidates(source())
    pd.testing.assert_frame_equal(rows,import_candidates(source().rename(columns={"chr":"#chr"})))
    assert list(rows.variant_id)==["chr1:101:A>G","chr2:201:C>T"]
    assert list(rows.source_row)==[1,2]
    with pytest.raises(ValueError):import_candidates(pd.concat([source(),source()]))


def test_membership_ignores_effects_and_source_order():
    a=source();b=a.iloc[::-1].copy();b["es_mean_ref"]=["-99999","99999"]
    assert list(select_candidates(import_candidates(a),1).variant_id)==list(select_candidates(import_candidates(b),1).variant_id)
    for limit in (True,0,-1,1.5):
        with pytest.raises(ValueError):select_candidates(import_candidates(a),limit)


def test_window_verifies_actual_reference_allele():
    row=dict(chrom="chr1",position_1based=103,reference="G",alternate="T")
    genome={"chr1":"A"*102+"G"+"A"*200}
    window,reason=allele_window(genome,row)
    assert reason=="reference verified" and window["variant_index"]==102
    assert window["reference_sequence"][102]=="G" and window["alternate_sequence"][102]=="T"
    assert allele_window(genome,{**row,"reference":"C"})[1]=="hg38 reference allele mismatch"
    assert allele_window(genome,{**row,"position_1based":1})[0] is None


class Tokenizer:
    mask_token_id=9
    def __call__(self,sequence,**kwargs):
        return dict(input_ids=[10]+["ACGT".index(x)+1 for x in sequence]+[11],
                    offset_mapping=[(0,0)]+[(i,i+1) for i in range(len(sequence))]+[(0,0)])


def test_sequence_constraints_match_original_assay_without_invented_biological_qc():
    clean="T"*20+"ACGA"+"TTTCGA"+"T"*50
    window=dict(reference_sequence=clean,alternate_sequence=clean[:22]+"T"+clean[23:],variant_index=22,
                window_start_0based=1000,window_end_0based=1000+len(clean))
    p=replace(VariantProtocol(),gc_tolerance=.2);motif=MotifDefinition("CTCF",pattern="ACGA",scan_reverse=False)
    candidate,reason,trace=sequence_control(Tokenizer(),window,motif,p)
    assert reason=="sequence control feasible" and candidate["sham_index"]==28
    # Read counts belong only to the independently validated synthetic parity fixture.
    original=dict(**window,genome_build="hg19",position_1based=1023,reference="G",alternate="T",reference_verified=True,
                  wgs_wt=20,wgs_mt=20,chip_wt=30,chip_mt=10,zygosity="heterozygous",chrom="chr1",locus_group="locus")
    case,_=prepare_variant(Tokenizer(),original,motif,p,{"locus":1})
    assert candidate["queries"]==case["queries"] and candidate["ids"]==case["ids"]
    assert sum(trace["candidate_rejections"].values())+1==trace["candidate_positions_checked"]


def test_download_requires_publisher_size_and_checksum(tmp_path):
    path=tmp_path/"archive";path.write_bytes(b"release")
    verify_archive(path,7,hashlib.md5(b"release").hexdigest())
    with pytest.raises(ValueError):verify_archive(path,8,hashlib.md5(b"release").hexdigest())
    with pytest.raises(ValueError):verify_archive(path,7,"0"*32)

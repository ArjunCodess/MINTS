"""Mapped-row intervention wiring on a known, pointwise native head."""
from types import SimpleNamespace
import pytest
import torch
from src.mapped_variant import mapped_logits, score_mapped_case
from src.control_diagnostics import token_map


class Tokenizer:
    mask_token_id=9
    def __call__(self,s,**kwargs):
        return dict(input_ids=[0]+["ACGT".index(c)+1 for c in s]+[5],
                    offset_mapping=[(0,0)]+[(i,i+1) for i in range(len(s))]+[(0,0)])


class Context(torch.nn.Module):
    num_attention_heads=2
    attention_head_size=2
    def forward(self,ids):
        values=ids.float()
        return torch.stack([values,values.sum().expand_as(values),values**2,torch.arange(len(ids)).float()],-1)


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.context=Context()
        self.attention=SimpleNamespace(self=self.context)
    def forward(self,ids):
        return self.context(ids)+1


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer=Layer()
        self.bert=SimpleNamespace(encoder=SimpleNamespace(layer=[self.layer]))
    def forward(self,input_ids,**kwargs):
        result=self.layer(input_ids[0])
        return SimpleNamespace(logits=torch.cat([result,-result,result[:,:2]],-1)[None])


def bundle():
    return SimpleNamespace(hf_model=Model(),tokenizer=Tokenizer(),device="cpu")


def test_final_query_restores_logits_across_different_indices_and_lengths():
    b=bundle()
    ref=[0,1,2,9,3,5];alt=[0,4,9,3,5]
    reference,cache=mapped_logits(b,ref,3,cache=True)
    observed,_=mapped_logits(b,alt,2)
    restored,_=mapped_logits(b,alt,2,donor=cache,final=True)
    assert not torch.equal(reference,observed)
    assert torch.equal(reference,restored)
    patched,_=mapped_logits(b,alt,2,donor=cache,head=(0,0))
    assert torch.equal(patched[:2],reference[:2])
    assert torch.equal(patched[2:4],observed[2:4])
    assert not b.hf_model.layer._forward_hooks
    assert not b.hf_model.layer.context._forward_hooks


def test_case_scoring_checks_orientation_and_aggregation(tmp_path):
    b=bundle();sequences=["AAAACCCG","ATAACCCG","AAATCCCG"]
    maps=[token_map(b.tokenizer,s) for s in sequences]
    q=dict(span=[4,5],indices=[5,5,5],token_id=2,width=1)
    case=dict(reference_sequence=sequences[0],alternate_sequence=sequences[1],sham_sequence=sequences[2],
        variant_index=1,sham_index=3,ids=[m[1] for m in maps],queries=[q])
    result=score_mapped_case(b,case,save_logits=tmp_path/"logits.npz")
    assert result["implementation_valid"]
    assert result["contrast"]==0
    assert result["query_diagnostics"][0]["final_restoration_errors"]==[0,0,0,0]
    assert (tmp_path/"logits.npz").is_file()
    case["queries"]=[{**q,"indices":[4,5,5]}]
    with pytest.raises(ValueError,match="Query spans"):
        score_mapped_case(b,case)


def test_invalid_mask_and_head_rejected():
    b=bundle()
    with pytest.raises(ValueError,match="mask"):
        mapped_logits(b,[0,1,2],1)
    with pytest.raises(ValueError,match="out of range"):
        mapped_logits(b,[0,9,2],1,head=(0,5))

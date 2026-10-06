"""Coordinate matching and structural geometry checks without model caches."""
from src.control_diagnostics import common_queries, same_side_queries
from src.native_followup import target_geometry


def test_correspondence_maps_spans_not_indices():
    a=([(0,0),(0,4),(4,8),(8,12)], [0,11,12,13], {(0,4):(1,11),(4,8):(2,12),(8,12):(3,13)})
    b=([(0,0),(0,2),(2,4),(4,8),(8,12)], [0,21,22,12,13], {(0,2):(1,21),(2,4):(2,22),(4,8):(3,12),(8,12):(4,13)})
    q=common_queries([a,b],1,[],32)
    assert q[0]["indices"]==[2,3]
    assert q[0]["span"]==[4,8]
    assert q[0]["width"]==4
    changed=(*b[:2], {**b[2], (4,8):(3,99)})
    assert [x["span"] for x in common_queries([a,changed],1,[],32)]==[[8,12]]


def test_same_side_distance_difference_equals_edit_separation():
    for i in range(10,20):
        for j in range(21,40):
            for span in ([0,4],[45,49]):
                side,gap=target_geometry((i,i+1),span)
                side2,gap2=target_geometry((j,j+1),span)
                assert side==side2
                assert abs(gap-gap2)==abs(j-i)
    queries=[dict(span=[0,4]),dict(span=[15,17]),dict(span=[45,49])]
    assert same_side_queries(queries,12,22)==[queries[0],queries[2]]

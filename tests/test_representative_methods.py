from __future__ import annotations

from wsi_sae.representatives import attach_slide_support_stats, rank_support_rows


def _base_rows():
    rows = [
        {
            "activation": 9.0,
            "source_rank": 1,
            "slide_key": "SLIDE_A",
            "tile_index": 1,
            "coord_x": 0,
            "coord_y": 0,
        },
        {
            "activation": 8.0,
            "source_rank": 2,
            "slide_key": "SLIDE_A",
            "tile_index": 2,
            "coord_x": 128,
            "coord_y": 0,
        },
        {
            "activation": 7.0,
            "source_rank": 3,
            "slide_key": "SLIDE_B",
            "tile_index": 3,
            "coord_x": 0,
            "coord_y": 128,
        },
        {
            "activation": 6.0,
            "source_rank": 4,
            "slide_key": "SLIDE_B",
            "tile_index": 4,
            "coord_x": 128,
            "coord_y": 128,
        },
        {
            "activation": 5.5,
            "source_rank": 5,
            "slide_key": "SLIDE_B",
            "tile_index": 5,
            "coord_x": 256,
            "coord_y": 128,
        },
    ]
    return attach_slide_support_stats(rows)


def test_representative_method_rankings_are_deterministic():
    rows = _base_rows()

    max_rows = rank_support_rows(rows, "max_activation")
    assert [r["tile_index"] for r in max_rows[:3]] == [1, 2, 3]

    median_rows = rank_support_rows(rows, "median_activation")
    assert median_rows[0]["tile_index"] == 3

    diverse_rows = rank_support_rows(rows, "diverse_support")
    assert [r["tile_index"] for r in diverse_rows[:2]] == [1, 3]

    spread_rows = rank_support_rows(rows, "slide_spread")
    assert spread_rows[0]["slide_key"] == "SLIDE_B"
    assert spread_rows[0]["tile_index"] == 3

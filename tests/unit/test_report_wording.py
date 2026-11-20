"""v1.1.4 wording helpers: explanation keys and the drop rationale."""

from holdout.agent.planner import explanation_key
from holdout.plan.schema import DropColumn, drop_rationale


def test_a_flag_without_columns_has_no_trailing_colon() -> None:
    assert explanation_key("A12", []) == "A12"
    assert explanation_key("A1", ["refund_issued"]) == "A1:refund_issued"
    assert explanation_key("A5", ["a", "b"]) == "A5:a,b"


def test_the_drop_rationale_names_the_final_drop_list() -> None:
    assert drop_rationale([]) == "no drops needed"
    one = [DropColumn(column="account", reason="key of the group split")]
    assert drop_rationale(one) == "drops 'account' (key of the group split)"
    many = [DropColumn(column=f"col_{i}", reason="x" * 150) for i in range(3)]
    assert drop_rationale(many) == "drops 'col_0', 'col_1', 'col_2' (reasons per column)"
    lots = [DropColumn(column=f"a_very_long_column_name_{i}", reason="why") for i in range(20)]
    assert drop_rationale(lots) == "drops 20 columns (listed with their reasons in drop_columns)"
    for drops in (one, many, lots):
        assert len(drop_rationale(drops)) <= 200

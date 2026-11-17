"""A9: target encoding is cross-fitted by construction; a guard fails on any new encoder."""

import dataclasses
import typing
from typing import Any

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder

from holdout.audit.checks import a09_encoding as a9
from holdout.audit.types import Severity
from holdout.plan.compiler import compile_plan
from holdout.plan.schema import CategoricalPreprocessing
from tests.audit_harness import trap_context

#: Every categorical encoding the schema offers, classified. Adding an option to the schema
#: without classifying it here fails test_every_schema_encoding_is_classified.
KNOWN_ENCODINGS = {"onehot": "target_free", "ordinal": "target_free", "target_cv": "cross_fitted"}


def test_every_schema_encoding_is_classified() -> None:
    options = set(typing.get_args(CategoricalPreprocessing.model_fields["encode"].annotation))
    assert options == set(KNOWN_ENCODINGS), "classify the new encoding in A9 and here"


@pytest.mark.parametrize("encoding", sorted(KNOWN_ENCODINGS))
def test_every_schema_encoding_compiles_to_a_safe_encoder(encoding: str) -> None:
    ctx = trap_context("group_leakage", 0)  # has categorical columns
    plan = ctx.plan.model_copy(
        update={"categorical": CategoricalPreprocessing(encode=encoding)}  # type: ignore[arg-type]
    )
    found = a9.encoders(compile_plan(plan, ctx.profile, run_seed=0))
    if KNOWN_ENCODINGS[encoding] == "target_free":
        assert found == []
    else:
        assert found == [("TargetEncoder", True)]


def test_target_cv_plan_reports_info_prevented_by_construction() -> None:
    ctx = trap_context("group_leakage", 0)
    assert ctx.plan.categorical.encode == "target_cv"  # high-cardinality 'account'
    flag = a9.run(ctx).flags[0]
    assert flag.severity is Severity.INFO
    assert "prevented by construction" in flag.message


def test_no_target_encoder_no_flag() -> None:
    ctx = trap_context("direct_leak", 0)
    assert ctx.plan.categorical.encode == "onehot"
    assert a9.run(ctx).flags == []


class NaiveTargetMeanEncoder(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Deliberately leaky: encodes each row with the mean label including its own."""

    def fit(self, X: Any, y: Any = None) -> "NaiveTargetMeanEncoder":
        return self

    def transform(self, X: Any) -> Any:
        return np.asarray(X)


def test_a_non_cross_fitted_encoder_is_critical(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = trap_context("group_leakage", 0)
    leaky = Pipeline([("encode", NaiveTargetMeanEncoder()), ("model", TargetEncoder(cv=2))])
    monkeypatch.setattr(a9, "compile_plan", lambda *args, **kwargs: leaky)
    flag = a9.run(dataclasses.replace(ctx)).flags[0]
    assert flag.severity is Severity.CRITICAL
    assert "NaiveTargetMeanEncoder" in flag.message

"""Audit layer: Holdout checks its own pipeline on the development partition.

Every check reads only development rows. Each returns typed flags carrying evidence, a
plain-language message and a typed fix; a critical flag, an incomplete audit or a
contaminated holdout blocks unsealing. See ``docs/audits.md`` for what each check can and
cannot detect.
"""

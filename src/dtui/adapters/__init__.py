"""Per-model trajectory adapters.

Built-in adapters are imported lazily by name to avoid importing heavy optional
dependencies (torch/transformers) on the light install. Third-party adapters can
register under the ``dtui.adapter`` entry-point group.
"""

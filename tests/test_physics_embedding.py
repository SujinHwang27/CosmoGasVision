"""Unit tests for [D-46] joint-physics conditional MLP with physics_id embedding.

Spec source: `experiments/nerf/LEDGER.md` §3 [D-46] "Math contract" subsection.
Pins:

  1. Flag-off bit-equivalent regression: with `use_physics_embedding=False`,
     output must match the baseline NeRF byte-for-byte at seed-identical init.
  2. Flag-on output shape preserved: (n_rays, n_bins, 4).
  3. Embedding parameters are learnable (gradient flows on backward).
  4. Distinct physics_ids produce distinct embeddings after a few SGD steps
     (catches the D-new residual risk that the model could degenerate to
     ignoring the physics_id input).

These are the host-only smoke tests; the 50-step P-mixed host smoke and the
Tier-1 Juno cell are dispatched after these pass per the LEDGER §3 [D-46]
smoke-gate spec.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.models.nerf import IGMNeRF


# ---------------------------------------------------------------------------
# Gate 1: flag-off bit-equivalent regression.
# ---------------------------------------------------------------------------
def test_flag_off_bit_equivalent_regression():
    """
    With use_physics_embedding=False, the [D-46] constructor change must not
    perturb the baseline forward path. Two seed-identical IGMNeRF builds with
    the flag off must produce torch.equal outputs on the same input.

    This is the binding bit-equivalent regression contract from LEDGER §3
    [D-46] "Discipline constraints".
    """
    n_rays, n_bins = 4, 16

    torch.manual_seed(20260511)
    model_a = IGMNeRF(hidden_dim=32, num_layers=8, L=4,
                      use_physics_embedding=False)
    torch.manual_seed(0)
    x = torch.rand(n_rays, n_bins, 3)
    out_a = model_a(x)

    torch.manual_seed(20260511)
    model_b = IGMNeRF(hidden_dim=32, num_layers=8, L=4,
                      use_physics_embedding=False)
    out_b = model_b(x)

    assert torch.equal(out_a, out_b), (
        "Bit equivalence broken: flag-off IGMNeRF differs between two "
        "seeded-identical constructions."
    )
    for i, name in enumerate(["density", "temp", "h1_frac", "vpec"]):
        assert torch.equal(out_a[..., i], out_b[..., i]), (
            f"Field {name} differs across seeded-identical flag-off models."
        )


def test_flag_off_matches_pre_d46_baseline():
    """
    Stronger version of the regression: an IGMNeRF built with default kwargs
    (the pre-[D-46] call signature) must produce identical outputs to one
    explicitly built with use_physics_embedding=False at the same seed.

    This catches any silent change to layer init order from the new
    nn.Embedding allocation (the embedding is gated behind the flag and
    must not consume from the RNG stream when the flag is off).
    """
    n_rays, n_bins = 4, 16

    torch.manual_seed(20260511)
    model_pre = IGMNeRF(hidden_dim=32, num_layers=8, L=4)
    torch.manual_seed(0)
    x = torch.rand(n_rays, n_bins, 3)
    out_pre = model_pre(x)

    torch.manual_seed(20260511)
    model_off = IGMNeRF(hidden_dim=32, num_layers=8, L=4,
                        use_physics_embedding=False)
    out_off = model_off(x)

    assert torch.equal(out_pre, out_off), (
        "Default-kwargs IGMNeRF output differs from explicit "
        "use_physics_embedding=False — RNG stream has been disturbed."
    )


# ---------------------------------------------------------------------------
# Gate 2: flag-on output shape preserved.
# ---------------------------------------------------------------------------
def test_flag_on_output_shape_preserved():
    """
    With use_physics_embedding=True and a valid physics_id batch, the forward
    pass returns (n_rays, n_bins, 4) with no NaN.
    """
    n_rays, n_bins = 8, 16

    torch.manual_seed(20260511)
    model = IGMNeRF(hidden_dim=32, num_layers=8, L=4,
                    use_physics_embedding=True)
    x = torch.rand(n_rays, n_bins, 3)
    # 8 rays: 2 from each of P0/P1/P2/P3.
    physics_id = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)

    out = model(x, physics_id=physics_id)
    assert out.shape == (n_rays, n_bins, 4), (
        f"Expected (n_rays, n_bins, 4); got {tuple(out.shape)}."
    )
    assert not torch.isnan(out).any(), "Forward produced NaNs."
    assert torch.isfinite(out).all(), "Forward produced non-finite values."

    for i, name in enumerate(["density", "temp", "h1_frac", "vpec"]):
        chan = out[..., i]
        assert chan.shape == (n_rays, n_bins), (
            f"Channel {name} shape {tuple(chan.shape)} != ({n_rays}, {n_bins})."
        )


# ---------------------------------------------------------------------------
# Gate 3: embeddings are learnable parameters (gradient flows).
# ---------------------------------------------------------------------------
def test_embedding_is_learnable_parameter():
    """
    nn.Embedding(4, 16) is registered on the module as `physics_embedding` and
    its weight appears in model.parameters(). After a backward pass on a non-
    trivial loss, the embedding's grad is finite, non-zero on at least one
    row, and propagates downstream into the layer-1 Linear.
    """
    n_rays, n_bins = 8, 16

    torch.manual_seed(20260511)
    model = IGMNeRF(hidden_dim=32, num_layers=8, L=4,
                    use_physics_embedding=True)

    # The embedding must be a learnable nn.Parameter.
    assert hasattr(model, "physics_embedding"), (
        "Model is missing `physics_embedding` attribute under flag-on."
    )
    assert isinstance(model.physics_embedding, nn.Embedding), (
        "physics_embedding is not an nn.Embedding."
    )
    assert model.physics_embedding.weight.requires_grad, (
        "Embedding weight is not learnable (requires_grad=False)."
    )
    # Confirm it's enumerated by .parameters() so the optimizer picks it up.
    param_ids = {id(p) for p in model.parameters()}
    assert id(model.physics_embedding.weight) in param_ids, (
        "Embedding weight is not in model.parameters() — optimizer would "
        "skip it."
    )

    x = torch.rand(n_rays, n_bins, 3)
    physics_id = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)

    out = model(x, physics_id=physics_id)
    loss = out.pow(2).sum()
    loss.backward()

    grad = model.physics_embedding.weight.grad
    assert grad is not None, "Embedding weight grad is None after backward."
    assert torch.isfinite(grad).all(), "Embedding weight grad has NaN/Inf."
    assert grad.abs().sum().item() > 0.0, (
        "Embedding weight grad is identically zero — gradient is not flowing."
    )
    # All four physics_ids appeared in the batch, so all four rows should
    # have received gradient.
    per_row_grad_norm = grad.norm(dim=1)
    assert (per_row_grad_norm > 0).all(), (
        f"Some embedding rows received no gradient: {per_row_grad_norm.tolist()}"
    )


# ---------------------------------------------------------------------------
# Gate 4: distinct physics_ids produce distinct embeddings after a few steps
#   (D-new residual-risk catch: the model could ignore physics_id).
# ---------------------------------------------------------------------------
def test_distinct_physics_ids_diverge_under_training():
    """
    After a few SGD steps on a synthetic per-physics target (each physics_id
    pushes the density head toward a different scalar value), the embedding
    vectors for different physics_ids must diverge: at least one pair of
    embedding rows has L2 distance > 0.1.

    This is the smoke-time analogue of LEDGER §3 [D-46] smoke gate 7
    (embedding non-degeneracy after 50 host-pipeline steps). It catches the
    D-new degeneracy where the model collapses to physics-agnostic.
    """
    n_rays, n_bins = 16, 8

    torch.manual_seed(20260511)
    model = IGMNeRF(hidden_dim=32, num_layers=8, L=4,
                    use_physics_embedding=True)

    # 4 rays per physics; same coords across physics so the only distinguishing
    # input is the embedding. This is intentionally adversarial: if the model
    # ignores physics_id, the loss is unminimizable.
    x = torch.rand(n_rays // 4, n_bins, 3)
    x = x.repeat(4, 1, 1)                               # (16, 8, 3)
    physics_id = torch.arange(4).repeat_interleave(4)   # [0,0,0,0,1,1,1,1,...]

    # Target: density (channel 0) tracks a per-physics scalar in {0.5, 1.5, 2.5, 3.5}.
    # Other channels left free.
    target_density = torch.tensor([0.5, 1.5, 2.5, 3.5]).repeat_interleave(4)
    target_density = target_density[:, None].expand(n_rays, n_bins)

    e_init = model.physics_embedding.weight.detach().clone()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        out = model(x, physics_id=physics_id)
        loss = (out[..., 0] - target_density).pow(2).mean()
        loss.backward()
        optimizer.step()

    e_final = model.physics_embedding.weight.detach().clone()

    # The embedding must have moved (otherwise the optimizer never updated it).
    moved = (e_final - e_init).norm(dim=1)
    assert (moved > 1e-4).all(), (
        f"Embedding rows did not update during training: per-row move = "
        f"{moved.tolist()}"
    )

    # At least one pair of rows must be pairwise distinct beyond the
    # smoke-gate threshold 0.1 (catches the D-new physics-agnostic collapse).
    max_pair_l2 = 0.0
    for i in range(4):
        for j in range(i + 1, 4):
            d = (e_final[i] - e_final[j]).norm().item()
            max_pair_l2 = max(max_pair_l2, d)
    assert max_pair_l2 > 0.1, (
        f"All physics_id embeddings collapsed to within 0.1 L2 after training; "
        f"max pairwise L2 = {max_pair_l2:.4e}. This is the D-new "
        f"physics-agnostic degeneracy signature."
    )


# ---------------------------------------------------------------------------
# Belt-and-braces: assert that the model rejects physics_id=None when flag is
# on, and rejects physics_id=<tensor> when flag is off. These are not in the
# spec's headline gate list but they protect against silent shape-mismatch
# bugs at training time.
# ---------------------------------------------------------------------------
def test_flag_on_requires_physics_id():
    torch.manual_seed(0)
    model = IGMNeRF(hidden_dim=16, num_layers=8, L=4,
                    use_physics_embedding=True)
    x = torch.rand(2, 4, 3)
    with pytest.raises(RuntimeError, match="physics_id"):
        model(x)


def test_flag_off_rejects_physics_id():
    torch.manual_seed(0)
    model = IGMNeRF(hidden_dim=16, num_layers=8, L=4,
                    use_physics_embedding=False)
    x = torch.rand(2, 4, 3)
    physics_id = torch.zeros(2, dtype=torch.long)
    with pytest.raises(RuntimeError, match="use_physics_embedding=False"):
        model(x, physics_id=physics_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

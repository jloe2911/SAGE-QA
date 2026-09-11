import inspect

import torch
import torch.nn.functional as F

from training.train_gnn_subgraph_retriever import listwise_soft_target_loss


def _explicit_distribution_kl(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    q = F.softmax(targets / 0.1, dim=0)
    log_p = F.log_softmax(scores, dim=0)
    return torch.sum(q * (torch.log(q) - log_p))


def test_joint_candidate_permutation_is_invariant() -> None:
    scores = torch.tensor([-0.4, 0.7, 0.1, 1.2])
    targets = torch.tensor([0.0, 0.9, 0.3, 1.0])
    permutation = torch.tensor([2, 0, 3, 1])

    original = listwise_soft_target_loss(scores, targets)
    permuted = listwise_soft_target_loss(scores[permutation], targets[permutation])

    torch.testing.assert_close(permuted, original)


def test_unrelated_example_does_not_change_first_individual_loss() -> None:
    first_scores = torch.tensor([0.3, -0.2, 0.8])
    first_targets = torch.tensor([0.3, 0.0, 1.0])
    unrelated_scores = torch.tensor([-1.0, 0.5])
    unrelated_targets = torch.tensor([0.0, 0.9])

    first_loss = listwise_soft_target_loss(first_scores, first_targets)
    batch_once = torch.stack(
        [first_loss, listwise_soft_target_loss(unrelated_scores, unrelated_targets)]
    ).mean()
    batch_duplicated = torch.stack(
        [
            first_loss,
            listwise_soft_target_loss(unrelated_scores, unrelated_targets),
            listwise_soft_target_loss(unrelated_scores, unrelated_targets),
        ]
    ).mean()

    torch.testing.assert_close(
        listwise_soft_target_loss(first_scores, first_targets), first_loss
    )
    assert batch_once.ndim == 0
    assert batch_duplicated.ndim == 0


def test_per_example_loss_is_explicit_kl_sum_not_inverse_list_size() -> None:
    for candidate_count in (2, 10, 100, 320):
        scores = torch.zeros(candidate_count)
        targets = torch.zeros(candidate_count)
        targets[0] = 1.0

        actual = listwise_soft_target_loss(scores, targets)
        expected = _explicit_distribution_kl(scores, targets)

        torch.testing.assert_close(actual, expected)
        if candidate_count > 2:
            assert not torch.isclose(actual, expected / candidate_count)


def test_better_score_target_alignment_decreases_loss() -> None:
    targets = torch.tensor([1.0, 0.3, 0.0])
    misaligned_scores = torch.tensor([-1.0, 0.0, 1.0])
    aligned_scores = torch.tensor([1.0, 0.0, -1.0])

    assert listwise_soft_target_loss(
        aligned_scores, targets
    ) < listwise_soft_target_loss(misaligned_scores, targets)


def test_objective_accepts_only_model_scores_train_targets_and_temperature() -> None:
    signature = inspect.signature(listwise_soft_target_loss)

    assert tuple(signature.parameters) == ("scores", "targets", "temperature")
    assert signature.parameters["temperature"].default == 0.1

    scores = torch.tensor([0.1, 0.2], requires_grad=True)
    train_rank_targets = torch.tensor([0.0, 1.0])
    loss = listwise_soft_target_loss(scores, train_rank_targets)
    loss.backward()

    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()

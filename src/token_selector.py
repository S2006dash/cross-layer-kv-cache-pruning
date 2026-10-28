import torch


def select_token_indices(
    scores,
    keep_count,
    preserve_edges=True,
    random_selection=False,
    seed=42,
):
    sequence_length = scores.numel()

    keep_count = max(
        1,
        min(sequence_length, int(keep_count))
    )

    mandatory = set()

    if preserve_edges and sequence_length > 0:
        mandatory.add(0)
        mandatory.add(sequence_length - 1)

    if keep_count <= len(mandatory):
        selected = sorted(mandatory)[:keep_count]

        return torch.tensor(
            selected,
            device=scores.device,
            dtype=torch.long,
        )

    remaining_count = keep_count - len(mandatory)

    eligible = torch.ones(
        sequence_length,
        device=scores.device,
        dtype=torch.bool,
    )

    for index in mandatory:
        eligible[index] = False

    eligible_indices = torch.nonzero(
        eligible,
        as_tuple=False,
    ).flatten()

    if random_selection:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        permutation = torch.randperm(
            eligible_indices.numel(),
            generator=generator,
        )[:remaining_count]

        extra_indices = (
            eligible_indices.cpu()[permutation]
            .to(scores.device)
        )

    else:
        eligible_scores = scores[eligible_indices]

        top_positions = torch.topk(
            eligible_scores,
            k=min(
                remaining_count,
                eligible_scores.numel(),
            ),
        ).indices

        extra_indices = eligible_indices[top_positions]

    selected = list(mandatory) + extra_indices.tolist()
    selected = sorted(set(int(i) for i in selected))

    if len(selected) > keep_count:
        selected = selected[:keep_count]

    return torch.tensor(
        selected,
        device=scores.device,
        dtype=torch.long,
    )
@torch.no_grad()
def evaluate_induction(model, device, seed=42):
    was_training = model.training
    model.eval()

    g = torch.Generator()
    g.manual_seed(seed)

    tokens = torch.randint(100, 20000, (8, 256), generator=g).to(device)
    for i in range(8):
        tok_A = 45000 + i
        tok_B = 46000 + i
        tokens[i, 100] = tok_A
        tokens[i, 101] = tok_B
        tokens[i, 150] = tok_A

    logits, _ = model(tokens)
    probs = torch.softmax(logits[:, 150, :], dim=-1)

    score = 0.0
    for i in range(8):
        tok_B = 46000 + i
        score += probs[i, tok_B].item()
    score /= 8.0

    model.train(was_training)
    return score

@torch.no_grad()
def measure_attention_entropy(model, device, seed=42):
    was_training = model.training
    model.eval()

    g = torch.Generator()
    g.manual_seed(seed)

    T = min(256, model.config.block_size)
    tokens = torch.randint(0, model.config.vocab_size, (4, T), generator=g).to(device)
    _ = model(tokens)

    entropies = []
    for block in model.transformer.h:
        if block.attn.last_entropy is not None:
            entropies.append(block.attn.last_entropy)

    model.train(was_training)

    if len(entropies) == 0:
        return float("nan")

    return torch.stack(entropies).mean().item()

@torch.no_grad()
def phase_statistics(model):
    all_norms = []

    for block in model.transformer.h:
        if not hasattr(block.attn, "W_phase_q"):
            return None, float("nan")

        Wq = block.attn.W_phase_q
        Wk = block.attn.W_phase_k

        norm_q = torch.linalg.norm(Wq, dim=(1, 2))
        norm_k = torch.linalg.norm(Wk, dim=(1, 2))
        norms = 0.5 * (norm_q + norm_k)
        all_norms.append(norms)

    if len(all_norms) == 0:
        return None, float("nan")

    stacked = torch.stack(all_norms)
    mean_per_head = stacked.mean(dim=0)
    mean_global = stacked.mean()

    return mean_per_head.detach().cpu(), mean_global.item()

@torch.no_grad()
def validation_loss(model, device, val_loader, max_batches=25):
    was_training = model.training
    model.eval()

    total_loss = 0.0
    count = 0

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break

        x = batch["input_ids"][:, :-1].to(device, non_blocking=True)
        y = batch["input_ids"][:, 1:].to(device, non_blocking=True)

        ac_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
        with torch.autocast(device_type=device.type, dtype=ac_dtype):
            _, loss = model(x, y)
            loss = loss.mean()

        total_loss += loss.item()
        count += 1

    model.train(was_training)
    return total_loss / count if count > 0 else float("nan")

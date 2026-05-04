def train_worker(config):
    gc.collect()

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        mixed_precision="fp16",
        kwargs_handlers=[ddp_kwargs],
    )

    seed_everything(config["seed"])

    save_dir = config["save_dir"]
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "train_config.json"), "w") as f:
            json.dump(config, f, indent=2)

    model_config = GPTConfig(**config["model"])
    model = GPT(model_config)

    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=False,
    )

    train_loader_raw = get_dataloader(
        path=config["train_path"],
        seq_len=model_config.block_size,
        batch_size=config["batch_size_per_device"],
        drop_last=True,
    )

    val_loader = None
    if accelerator.is_main_process:
        val_loader = get_dataloader(
            path=config["val_path"],
            seq_len=model_config.block_size,
            batch_size=config["eval_batch_size_per_device"],
            drop_last=False,
        )

    required_global_micro_batches = config["max_steps"] * config["grad_accum"] * accelerator.num_processes
    if len(train_loader_raw) < required_global_micro_batches:
        print(f"[WARNING] Dataset has {len(train_loader_raw)} batches, "
              f"but {required_global_micro_batches} needed.")
        print(f"[WARNING] Training will naturally stop when dataset ends.")

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader_raw)

    tokens_per_update = (
        config["batch_size_per_device"]
        * accelerator.num_processes
        * model_config.block_size
        * config["grad_accum"]
    )

    def get_lr(step):
        if step < config["warmup"]:
            return config["lr"] * (step + 1) / config["warmup"]

        progress = (step - config["warmup"]) / max(1, (config["max_steps"] - config["warmup"]))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return config["min_lr"] + cosine * (config["lr"] - config["min_lr"])

    csv_file = None
    writer = None
    best_val = float("inf")

    if accelerator.is_main_process:
        csv_path = os.path.join(save_dir, "train_log_holo_0.15_run.csv")
        csv_file = open(csv_path, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow([
            "step",
            "train_loss",
            "val_loss",
            "val_ppl",
            "induction",
            "entropy",
            "phase_norm",
            "steps_per_sec",
            "tokens_per_sec",
            "wall_time_sec"
        ])
        csv_file.flush()

        print(f"Using {accelerator.num_processes} GPU(s)")
        print(f"Parameters: {n_params:,}")
        print(f"Tokens/update: {tokens_per_update:,}")
        print(f"Planned total tokens: {tokens_per_update * config['max_steps']:,}")
        print(f"Save dir: {save_dir}")

    completed_step = 0
    micro_in_update = 0
    running_micro_loss = 0.0
    log_loss_accum = 0.0
    log_steps_accum = 0

    train_start_time = time.time()
    last_log_time = train_start_time
    last_log_step = 0

    optimizer.zero_grad(set_to_none=True)
    model.train()

    try:
        for batch in train_loader:
            if completed_step >= config["max_steps"]:
                break

            lr = get_lr(completed_step + 1)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            x = batch["input_ids"][:, :-1]
            y = batch["input_ids"][:, 1:]

            should_sync = (micro_in_update + 1 == config["grad_accum"])
            sync_context = nullcontext() if should_sync else accelerator.no_sync(model)

            with sync_context:
                with accelerator.autocast():
                    _, loss = model(x, y)
                    loss = loss.mean()

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step={completed_step}, micro={micro_in_update}")

                running_micro_loss += loss.detach().float().item()

                accelerator.backward(loss / config["grad_accum"])

            micro_in_update += 1

            if should_sync:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                completed_step += 1
                update_loss = running_micro_loss / config["grad_accum"]
                running_micro_loss = 0.0
                micro_in_update = 0

                log_loss_accum += update_loss
                log_steps_accum += 1

                if completed_step % config["log_every"] == 0:
                    accelerator.wait_for_everyone()

                    smooth_loss = log_loss_accum / max(log_steps_accum, 1)
                    log_loss_accum = 0.0
                    log_steps_accum = 0

                    now = time.time()
                    elapsed = now - last_log_time
                    steps_since_log = completed_step - last_log_step
                    steps_per_sec = steps_since_log / max(elapsed, 1e-8)
                    tokens_per_sec = steps_per_sec * tokens_per_update
                    wall_time_sec = now - train_start_time

                    last_log_time = now
                    last_log_step = completed_step

                    if accelerator.is_main_process:
                        base_model = accelerator.unwrap_model(model)

                        val_loss = validation_loss(
                            base_model,
                            device=accelerator.device,
                            val_loader=val_loader,
                            max_batches=config["val_batches"]
                        )
                        val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
                        induction = evaluate_induction(base_model, device=accelerator.device)
                        entropy = measure_attention_entropy(base_model, device=accelerator.device)
                        _, phase_norm = phase_statistics(base_model)

                        writer.writerow([
                            completed_step,
                            smooth_loss,
                            val_loss,
                            val_ppl,
                            induction,
                            entropy,
                            phase_norm,
                            steps_per_sec,
                            tokens_per_sec,
                            wall_time_sec
                        ])
                        csv_file.flush()

                        print(f"\nStep {completed_step}")
                        print(f"Train Loss:    {smooth_loss:.6f}")
                        print(f"Val Loss:      {val_loss:.6f}")
                        print(f"Val PPL:       {val_ppl:.3f}")
                        print(f"Induction:     {induction:.6f}")
                        print(f"Entropy:       {entropy:.6f}")
                        print(f"Phase Norm:    {phase_norm:.6f}")
                        print(f"Steps/sec:     {steps_per_sec:.4f}")
                        print(f"Tokens/sec:    {tokens_per_sec:.2f}")
                        print(f"Wall time sec: {wall_time_sec:.1f}")

                        if val_loss < best_val:
                            if free_disk_gb(save_dir) < config["min_free_gb_best"]:
                                raise RuntimeError("Not enough free disk space for best checkpoint")

                            best_val = val_loss
                            save_model_weights(
                                base_model,
                                os.path.join(save_dir, "best_val_holo_0.15_run_fp16.pt"),
                                dtype=torch.float16
                            )
                            print(f"Saved best_val_fp16.pt with val_loss={best_val:.6f}")

                        if completed_step % config["save_every"] == 0:
                            if free_disk_gb(save_dir) >= config["min_free_gb_best"]:
                                save_model_weights(
                                    base_model,
                                    os.path.join(save_dir, f"step_{completed_step}_fp16.pt"),
                                    dtype=torch.float16
                                )
                                print(f"Saved step_{completed_step}_fp16.pt")

                    accelerator.wait_for_everyone()

    finally:
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            try:
                base_model = accelerator.unwrap_model(model)
                if free_disk_gb(save_dir) >= config["min_free_gb_final"]:
                    save_model_weights(
                        base_model,
                        os.path.join(save_dir, "step_final_fp32.pt"),
                        dtype=torch.float32
                    )
                    print("Saved final fp32 checkpoint")
                else:
                    print("Not enough disk space to save final fp32 checkpoint")
            except Exception as e:
                print(f"Failed to save final checkpoint: {e}")

            if csv_file is not None:
                csv_file.close()

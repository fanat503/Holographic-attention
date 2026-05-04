def prepare_sterile_datasets(
    dataset_path="HuggingFaceFW/fineweb-edu",
    dataset_config="sample-10BT",  
    train_tokens_count=2_000_000_000,
    val_tokens_count=20_000_000,
    test_tokens_count=20_000_000,   
    save_dir="/kaggle/working",
):
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    eos_id = tokenizer.eos_token_id
    assert eos_id is not None, "gpt2 tokenizer must have eos token id"

    train_tokens = torch.empty(train_tokens_count, dtype=torch.int32)
    val_tokens = torch.empty(val_tokens_count, dtype=torch.int32)
    test_tokens = (
        torch.empty(test_tokens_count, dtype=torch.int32)
        if test_tokens_count > 0 else None
    )

    splits = [
        ("train", train_tokens),
        ("val", val_tokens),
    ]
    if test_tokens is not None:
        splits.append(("test", test_tokens))

    split_idx = 0
    write_ptr = 0

    total_needed = train_tokens_count + val_tokens_count + test_tokens_count
    pbar = tqdm(total=total_needed, desc="Collecting tokens", mininterval=100.0)

    if dataset_config is None:
        dataset = load_dataset(dataset_path, split="train", streaming=True)
    else:
        dataset = load_dataset(dataset_path, dataset_config, split="train", streaming=True)

    for row in dataset:
        if split_idx >= len(splits):
            break

        text = row["text"]
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids.append(eos_id)  

        ids_tensor = torch.tensor(ids, dtype=torch.int32)
        pos = 0

        while pos < ids_tensor.numel() and split_idx < len(splits):
            split_name, split_tensor = splits[split_idx]
            remaining = split_tensor.numel() - write_ptr
            take = min(remaining, ids_tensor.numel() - pos)

            split_tensor[write_ptr:write_ptr + take] = ids_tensor[pos:pos + take]

            write_ptr += take
            pos += take
            pbar.update(take)

            if write_ptr == split_tensor.numel():
                print(f"Finished {split_name}: {write_ptr:,} tokens")
                split_idx += 1
                write_ptr = 0

    pbar.close()

    if split_idx < len(splits):
        raise RuntimeError(
            f"Dataset stream ended too early. "
            f"Filled only {split_idx}/{len(splits)} splits."
        )

    train_path = os.path.join(save_dir, "train_fixed_tokens.pt")
    val_path = os.path.join(save_dir, "val_fixed_tokens.pt")
    torch.save(train_tokens, train_path)
    torch.save(val_tokens, val_path)

    print(f"Saved train -> {train_path}")
    print(f"Saved val   -> {val_path}")

    if test_tokens is not None:
        test_path = os.path.join(save_dir, "test_fixed_tokens.pt")
        torch.save(test_tokens, test_path)
        print(f"Saved test  -> {test_path}")

    meta = {
        "dataset_path": dataset_path,
        "dataset_config": dataset_config,
        "tokenizer": "gpt2",
        "eos_token_id": eos_id,
        "dtype": "int32",
        "train_tokens": train_tokens_count,
        "val_tokens": val_tokens_count,
        "test_tokens": test_tokens_count,
    }
    meta_path = os.path.join(save_dir, "dataset_meta.pt")
    torch.save(meta, meta_path)
    print(f"Saved meta  -> {meta_path}")

    def size_gb(t):
        return t.element_size() * t.numel() / (1024 ** 3)

    print("\nSizes:")
    print(f"train: {size_gb(train_tokens):.2f} GB")
    print(f"val:   {size_gb(val_tokens):.2f} GB")
    if test_tokens is not None:
        print(f"test:  {size_gb(test_tokens):.2f} GB")

prepare_sterile_datasets(
    dataset_path="HuggingFaceFW/fineweb-edu",
    dataset_config="sample-10BT",
    train_tokens_count=2_000_000_000,
    val_tokens_count=20_000_000,
    test_tokens_count=20_000_000,
    save_dir="***", # enter your save_dir
) 

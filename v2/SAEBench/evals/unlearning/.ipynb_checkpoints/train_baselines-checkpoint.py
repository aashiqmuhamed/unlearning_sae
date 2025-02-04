import os
import datetime
import argparse
import numpy as np
import torch
from transformers import AdamW
from tqdm import tqdm
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


def forward_with_cache(model, inputs, module, no_grad=True):
    cache = []

    def hook(module, input, output):
        if isinstance(output, tuple):
            cache.append(output[0])
        else:
            cache.append(output)
        return None

    hook_handle = module.register_forward_hook(hook)
    # import pdb; pdb.set_trace()

    inputs = {**inputs, 'use_cache': False}

    if no_grad:
        with torch.no_grad():
            _ = model(**inputs)
    else:
        _ = model(**inputs)

    hook_handle.remove()
    return cache[0]


def get_params(model, layer_ids, param_ids):
    if layer_ids == ["all"]:
        return list(model.parameters())

    params = []
    for layer_id in layer_ids:
        for i, p in enumerate(model.model.layers[layer_id].parameters()):
            if param_ids == ["all"] or i in param_ids:
                params.append(p)
    return params


def load_model(model_name_or_path):
    torch_dtype = "auto" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True, use_fast=False
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.mask_token_id = tokenizer.eos_token_id
    tokenizer.sep_token_id = tokenizer.eos_token_id
    tokenizer.cls_token_id = tokenizer.eos_token_id

    return model, tokenizer


def get_data(forget_corpora, retain_corpora, min_len=50, max_len=2000, batch_size=4):
    def get_dataset(name):
        data = []
        if name == "wikitext":
            raw_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            for x in raw_data:
                if len(x['text']) > min_len:
                    data.append(str(x['text']))
        else:
            # for line in open(f"data/{name}.jsonl", "r"):
            for line in open(f"/data/aashiq_muhamed/unlearning/SAEBench/evals/unlearning/data/{name}.jsonl", "r"):
                if "bio-forget-corpus" in name:
                    raw_text = json.loads(line)['text']
                else:
                    raw_text = line
                if len(raw_text) > min_len:
                    data.append(str(raw_text))

        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        return data

    return (
        [get_dataset(c) for c in forget_corpora],
        [get_dataset(c) for c in retain_corpora]
    )


def compute_nll_loss(model, inputs):
    """Compute the negative log-likelihood loss."""
    outputs = model(**inputs)
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = inputs['input_ids'][..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
    return loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1))

    print("\n".join(f"{k}={v}" for k, v in vars(args).items()))


def run_unlearning(
        method,
        updated_model,
        tokenizer,
        forget_data_list,
        retain_data_list,
        args,
):
    print(f"==== {method} Config ====")
    print("\n".join(f"{k}={v}" for k, v in vars(args).items()))
    print("=====")

    updated_model = updated_model.train()
    params = get_params(updated_model, args.layer_ids, args.param_ids)
    optimizer = AdamW(params, lr=args.lr)

    # Load frozen model only for RMU method
    if method == "rmu":
        frozen_model, _ = load_model(args.model_name_or_path)
        frozen_module = eval(
            args.module_str.format(model_name="frozen_model", layer_id=args.layer_id)
        )
        updated_module = eval(
            args.module_str.format(model_name="updated_model", layer_id=args.layer_id)
        )

        # Initialize control vectors
        control_vectors_list = []
        for i in range(len(forget_data_list)):
            random_vector = torch.rand(1, 1, updated_model.config.hidden_size,
                                       dtype=updated_model.dtype,
                                       device=updated_model.device)
            control_vec = random_vector / torch.norm(random_vector) * args.steering_coeff_list[i]
            control_vectors_list.append(control_vec)

    num_batches = min(
        args.max_num_batches,
        min([len(f) for f in forget_data_list]),
        min([len(r) for r in retain_data_list]),
    )

    truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = "right"

    for epoch in range(1):
        print(f"======= Epoch {epoch} =======")
        with tqdm(total=num_batches) as pbar:
            for idx in range(num_batches):
                topic_idx = idx % len(forget_data_list)
                batch_idx = idx // len(forget_data_list)

                unlearn_batch = forget_data_list[topic_idx][batch_idx]
                retain_batch = retain_data_list[topic_idx][batch_idx]

                # Process data based on method
                max_length = 512 if topic_idx == 0 else 768
                unlearn_inputs = tokenizer(
                    unlearn_batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_length
                )
                # .to(updated_model.device)

                if method == "rmu":
                    # RMU: Use activation steering
                    updated_forget_activations = forward_with_cache(
                        updated_model, unlearn_inputs, module=updated_module, no_grad=False
                    ).to(updated_model.device)
                    control_vec = control_vectors_list[topic_idx]
                    # import pdb; pdb.set_trace()
                    unlearn_loss = torch.nn.functional.mse_loss(
                        updated_forget_activations, control_vec
                    )
                    
                else:
                    # Gradient ascent: Use NLL loss
                    unlearn_loss = -compute_nll_loss(updated_model, unlearn_inputs)

                # Process retain data
                retain_inputs = tokenizer(
                    retain_batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=512
                ).to(updated_model.device)

                if method == "rmu":
                    # RMU: Use activation matching
                    updated_retain_activations = forward_with_cache(
                        updated_model, retain_inputs, module=updated_module, no_grad=False
                    ).to(updated_model.device)
                    # import pdb; pdb.set_trace()
                    frozen_retain_activations = forward_with_cache(
                        frozen_model, retain_inputs, module=frozen_module, no_grad=True
                    ).to(updated_model.device)
                    retain_loss = torch.nn.functional.mse_loss(
                        updated_retain_activations, frozen_retain_activations
                    )
                else:
                    # Gradient ascent: Use NLL loss
                    retain_loss = compute_nll_loss(updated_model, retain_inputs)

                retain_loss *= args.alpha[topic_idx]

                # Update model
                loss = (unlearn_loss + retain_loss)/ args.gradient_accumulation_steps
                loss.backward()

                # Only update weights after accumulating gradients for specified steps
                if (idx + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                print(f"loss: {loss.item():.4g} | unlearn_loss: {unlearn_loss.item():.4g} | "
                      f"retain_loss: {retain_loss.item():.4g} | ")
                      # f"param_change: {params[0].grad.abs().mean().item():.4g}")

                if args.verbose and method == "rmu":
                    # Only log activation metrics for RMU method
                    # updated_forget_activations = forward_with_cache(
                    #     updated_model, unlearn_inputs, module=updated_module, no_grad=True
                    # )
                    frozen_forget_activations = forward_with_cache(
                        frozen_model, unlearn_inputs, module=frozen_module, no_grad=True
                    ).to(updated_model.device)
                    # updated_retain_activations = forward_with_cache(
                    #     updated_model, retain_inputs, module=updated_module, no_grad=True
                    # )
                    # frozen_retain_activations = forward_with_cache(
                    #     frozen_model, retain_inputs, module=frozen_module, no_grad=True
                    # )

                    unlearn_cosine = torch.nn.functional.cosine_similarity(
                        updated_forget_activations, frozen_forget_activations, dim=-1
                    ).mean()
                    retain_cosine = torch.nn.functional.cosine_similarity(
                        updated_retain_activations, frozen_retain_activations, dim=-1
                    ).mean()

                    print(f"unlearn_cosine_sim={unlearn_cosine.item()}")
                    print(f"retain_cosine_sim={retain_cosine.item()}")
                    # print_norms(topic_idx, updated_forget_activations, frozen_forget_activations,
                    #             updated_retain_activations, frozen_retain_activations)

                pbar.update(1)


    # Handle any remaining gradients at the end of epoch
    if num_batches % args.gradient_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    # Free up memory if frozen model was loaded
    if method == "rmu":
        del frozen_model
        torch.cuda.empty_cache()

    tokenizer.truncation_side = truncation_side

    # Save model
    if args.output_dir:
        path = os.path.join(args.output_dir, method)
    else:
        date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = f"models/{method}_{args.model_name_or_path}_alpha-{args.alpha}_batches-{num_batches}_layer-{args.layer_id}_{date}"

    os.makedirs(path, exist_ok=True)
    updated_model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Saved model to {path}")


def main():
    parser = argparse.ArgumentParser()
    # Method selection
    parser.add_argument("--method", type=str, choices=["rmu", "gradient_ascent"],
                        default="rmu", help="Unlearning method to use")

    # Model arguments
    parser.add_argument("--model_name_or_path", type=str,
                        default="HuggingFaceH4/zephyr-7b-beta")
    parser.add_argument("--module_str", type=str,
                        default="{model_name}.model.layers[{layer_id}]")
    parser.add_argument("--output_dir", type=str, default=None)

    # Data arguments
    parser.add_argument("--retain_corpora", type=str, default="wikitext,wikitext")
    parser.add_argument("--forget_corpora", type=str,
                        default="bio-forget-corpus,cyber-forget-corpus")

    # Hyperparameters
    parser.add_argument("--alpha", type=str, default="100,100",
                        help="retain weight")
    parser.add_argument("--steering_coeffs", type=str, default="20,20",
                        help="RMU only: Steer vector weight in order of topic")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min_len", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_num_batches", type=int, default=80)

    parser.add_argument("--layer_id", type=int, default=7,
                        help="Layer to monitor activations from (for RMU)")
    parser.add_argument("--layer_ids", type=str, default="all",
                        help="Layers to update during training, 'all' for full model")
    parser.add_argument("--param_ids", type=str, default="all",
                        help="Parameter groups to update, 'all' for all parameters")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                    help="Number of steps to accumulate gradients over")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Process arguments
    args.retain_corpora = args.retain_corpora.split(",")
    args.forget_corpora = args.forget_corpora.split(",")
    args.steering_coeff_list = [float(c) for c in args.steering_coeffs.split(",")]
    args.alpha = [float(c) for c in args.alpha.split(",")]
    # args.layer_ids = [int(layer_id) for layer_id in args.layer_ids.split(",")]
    # args.param_ids = [int(param_id) for param_id in args.param_ids.split(",")]


    # Replace these lines in the argument processing section
    args.layer_ids = ["all"] if args.layer_ids == "all" else [int(layer_id) for layer_id in args.layer_ids.split(",")]
    args.param_ids = ["all"] if args.param_ids == "all" else [int(param_id) for param_id in args.param_ids.split(",")]



    # Set random seeds
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load model and data
    updated_model, tokenizer = load_model(args.model_name_or_path)
    forget_data_list, retain_data_list = get_data(
        args.forget_corpora,
        args.retain_corpora,
        args.min_len,
        args.max_len,
        args.batch_size,
    )

    # Run selected unlearning method
    run_unlearning(
        args.method,
        updated_model,
        tokenizer,
        forget_data_list,
        retain_data_list,
        args,
    )


if __name__ == "__main__":
    main()

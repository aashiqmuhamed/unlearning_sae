import argparse
import torch
import os

def create_parser():
    parser = argparse.ArgumentParser(description="Parser for model and SAE parameters")

    # Required arguments
    parser.add_argument("--model_name",default = 'google/gemma-2-2b-it', type=str, help="Name of the model")
    parser.add_argument("--sae_name",default="gemma-scope-2b-pt-res-canonical", type=str, help="Name of the SAE")
    parser.add_argument("--sae_id",default = "layer_3/width_16k/canonical", type=str, help="ID of the SAE")
    parser.add_argument("--retain_dset",default = 'wikitext', type=str, help="Retain dataset")
    parser.add_argument("--fgt_dset",default = 'wmdp_forget_corpora', type=str, help="Forget dataset")

    parser.add_argument("--run_baseline", action='store_true')
    parser.add_argument("--run_unlearn", action='store_true')


    # Optional arguments
    parser.add_argument("--use_error_term", type=bool, default=True, help="Use error term (default: True)")
    parser.add_argument("--device", type=str, default=None, help="Device to use (default: None, will use CUDA if available)")
    parser.add_argument("--num_activations", type=int, nargs='+', default=[100,50,25,10,5], help="Number of activations (default: [100,50,25,10,5])")
    parser.add_argument("--clamp_values", type=int, nargs='+', default=[-10,-25,-50,-100,-200], help="clamp values (default: [-10,-50,-100,-200])")
    parser.add_argument("--th_ratio", type=float, default=1e3, help="Threshold ratio (default: 1e3)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for LLM (default: None)")

    #others
    # ...
    return parser

def get_args():
    parser = create_parser()
    args = parser.parse_args()
    args.root_folder = os.path.dirname(os.path.abspath(__file__)) + "/"
    # If device is not specified, use CUDA if available
    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Print all arguments
    print("Parsed arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    return args

if __name__ == "__main__":
    get_args()
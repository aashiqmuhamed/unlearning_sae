import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

def download_gemma(save_path="./gemma-model"):
    """
    Download the Gemma 2B-IT model and tokenizer.
    Requires HF token with accepted terms of use.
    """
    # Get token from environment variable
    token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    
    if not token:
        raise ValueError(
            "Please set HUGGING_FACE_HUB_TOKEN environment variable. "
            "Make sure you've accepted the model terms on Hugging Face first."
        )
    
    # Login to Hugging Face
    login(token=token)
    
    try:
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "google/gemma-2b-it",
            token=token,
            cache_dir="/data/datasets/wmdp_test/model_dir"
        )
        
        print("Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2b-it",
            token=token,
            device_map="auto",  # Handles multi-GPU or CPU setup automatically
            cache_dir="/data/datasets/wmdp_test/model_dir"
        )
        
        # Save model and tokenizer locally
        print(f"Saving to {save_path}...")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
        print("Download complete!")
        
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        print("Make sure you've accepted the model terms at: https://huggingface.co/google/gemma-2b-it")

if __name__ == "__main__":
    download_gemma()

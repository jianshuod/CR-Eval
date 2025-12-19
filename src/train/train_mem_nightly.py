from src.train.train_nightly import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")

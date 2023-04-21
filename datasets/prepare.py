import argparse
from prophet.prepare import prepare as prophet_maker

# In case you want to prepare from the command line
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', type=str, default='gpt2')
    parser.add_argument('--dataset', type=str, default='prophet')
    args = parser.parse_args()
    if args.dataset == 'prophet':
        prophet_maker(args.tokenizer)
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")
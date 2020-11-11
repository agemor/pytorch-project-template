import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="Type of dataset to process")
    parser.add_argument('--dataset_dir', type=str, help="Directory contains dataset. Dataset name is used if not specified.")

    args = parser.parse_args()

    if args.dataset_dir is None:
        args.dataset_dir = args.dataset

if __name__ == '__main__':
    main()
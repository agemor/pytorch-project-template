import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="Type of dataset to process")
    parser.add_argument('--source', type=str, help="Directory contains dataset.")

    args = parser.parse_args()


if __name__ == '__main__':
    main()

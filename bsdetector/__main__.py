import argparse
from bsdetector.bias import get_text_from_article_file, print_feature_data


DESC = "Biased Statement Detection explains why english prose text is biased."


def main():
    parser = argparse.ArgumentParser(description=DESC)
    parser.add_argument("-i", dest="filename", required=True,
                        help="Input file", metavar="FILE")
    parser.add_argument("-o", dest="output", required=False,
                        help="Output type", metavar="OUTPUT",
                        default='json', choices=['tsv', 'json', 'html'])
    args = parser.parse_args()
    sentence_list = get_text_from_article_file(args.filename).split('\n')
    print_feature_data(sentence_list, output_type=args.output)


if __name__ == '__main__':
    main()

import sys


def add_vocab_vector_size(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    vocab_size = len(lines)
    vector_size = len(lines[0].split()) - 1  # Subtract 1 for the word itself

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"{vocab_size} {vector_size}\n")
        f.writelines(lines)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python add_vocab_vector_size.py <input_file> <output_file>")
        sys.exit(1)
    add_vocab_vector_size(sys.argv[1], sys.argv[2])


# run in command line
# python scr_py/add_vec_size.py data/w2v.txt data/w2v_size.txt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("name", type = str, default = "oguz")
args = parser.parse_args()
name = args.name
print("Hello " + name + "!")

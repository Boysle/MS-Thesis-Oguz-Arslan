import argparse
parser = argparse.ArgumentParser()
parser.add_argument("name", type = str, default = "oguz")
name = args.name
print("Hello " + name + "!")

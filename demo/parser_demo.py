import argparse

class ParseDict(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict()) # set each name of the attribute to hold the created object(s) as dictionary
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value


parser = argparse.ArgumentParser()
parser.add_argument('-k',nargs='*',action=ParseDict)
parser.add_argument("--model", type=str)
args = parser.parse_args()
print(type(args.k['embedding_dim']))

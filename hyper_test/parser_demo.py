import argparse

class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value

parser = argparse.ArgumentParser()
parser.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargs)
args = parser.parse_args()
args.kwargs = {'a':123,'b':456}
print(args.kwargs)


# class FooAction(argparse.Action):
#     def __init__(self, option_strings, dest, nargs=None, **kwargs):
#         if nargs is not None:
#             raise ValueError("nargs not allowed")
#         super().__init__(option_strings, dest, **kwargs)
#     def __call__(self, parser, namespace, values, option_string=None):
#         print('%r %r %r' % (namespace, values, option_string))
#         # print(f'self.dest:{self.dest}')
#         setattr(namespace, self.dest, values)
  

# parser = argparse.ArgumentParser()
# parser.add_argument('--foo', action=FooAction)
# parser.add_argument('bar', action=FooAction)
# args = parser.parse_args('1 --foo 2'.split())




from .version import *

def welcome():
      from importlib.resources import files
      print(f'''
      🌈🧃🧑‍💻 Hooray for you! You've installed `chromatic_extracting {version()}`
      on your computer, probably because you're trying to develop code for it. 
      To do so, please edit or add files in the following directory:
      {files('chromatic_extracting')}
      ''')
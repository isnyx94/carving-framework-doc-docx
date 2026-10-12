import sys, runpy
sys.path.insert(0, '.'); sys.path.insert(0, __import__('os').path.abspath('./reanalysis/common'))
import strict_patch; strict_patch.apply()
sys.argv = ['run_external.py'] + sys.argv[1:]
runpy.run_path(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), 'run_external.py'), run_name='__main__')

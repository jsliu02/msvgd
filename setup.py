from setuptools import setup, find_packages

setup( 
	name='msvgd', 
	version='1.0', 
	description='Mitotic Stein variational gradient descent.', 
	author='Jamie Liu', 
	packages=['msvgd'],
	install_requires=[ 
		'numpy',
		'torch',
		'tqdm'
	], 
) 

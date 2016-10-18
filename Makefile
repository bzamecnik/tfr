install:
	pip install tfr

install_dev:
	pip install -e .

uninstall:
	pip uninstall tfr


# PyPI production
pypi_publish:
	python setup.py register -r pypi

publish:
	python setup.py sdist upload -r pypi


# PyPI test
test_pypi_register:
	python setup.py register -r pypitest

test_publish:
	python setup.py sdist upload -r pypitest

test_install:
	pip install --verbose --index-url https://testpypi.python.org/pypi/ tfr

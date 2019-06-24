oriana:
	python3 setup.py install

test: test/test.py
	pytest test/test.py

.PHONY: oriana test

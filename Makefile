LIB_NAME := cuPSO

.PHONY: clean test stubs

test:
	python -m unittest

clean:
	rm -rf build/* ${LIB_NAME}/${LIB_NAME}$(shell python3-config --extension-suffix)

stubs:
	PYTHONPATH=./ pybind11-stubgen ${LIB_NAME}
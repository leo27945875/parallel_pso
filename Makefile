LIB_NAME      := cuPSO
GITHUB_BRANCH := cuda_version
COMMIT_MSG    := ${m}

.PHONY: clean test stubs cm push

test:
	python -m unittest

clean:
	rm -rf build/* ${LIB_NAME}/${LIB_NAME}$(shell python3-config --extension-suffix)

stubs:
	PYTHONPATH=./ pybind11-stubgen ${LIB_NAME}

cm:
	git add .
	git commit -m '${COMMIT_MSG}'

push:
	git push origin ${GITHUB_BRANCH}
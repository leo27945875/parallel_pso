LIB_NAME      := cuPSO
GITHUB_BRANCH := cuda_version
COMMIT_MSG    := ${m}


.PHONY: clean test stubs cm push cpu gpu perf plot

test:
	python -m unittest

clean:
	rm -rf build/* ${LIB_NAME}/${LIB_NAME}$(shell python3-config --extension-suffix) ${LIB_NAME}/lib${LIB_NAME}.a

stubs:
	PYTHONPATH=./ pybind11-stubgen ${LIB_NAME}

cm:
	git add .
	git commit -m '${COMMIT_MSG}'

push:
	git push origin ${GITHUB_BRANCH}

cpu:
	python -m core.pypso
gpu:
	python -m core.pycupso
perf:
	python -m core ${c}
plot:
	python -m core.plot ${c}
LIB_NAME      := cuPSO
GITHUB_BRANCH := cuda_version
COMMIT_MSG    := ${m}
DEVICE        := ${d}

.PHONY: clean test stubs cm push cpu gpu perf

test:
	python -m unittest

cpu:
	python pypso.py
gpu:
	python pycupso.py
perf:
	python performance.py ${DEVICE}


clean:
	rm -rf build/* ${LIB_NAME}/${LIB_NAME}$(shell python3-config --extension-suffix) ${LIB_NAME}/lib${LIB_NAME}.a

stubs:
	PYTHONPATH=./ pybind11-stubgen ${LIB_NAME}

cm:
	git add .
	git commit -m '${COMMIT_MSG}'

push:
	git push origin ${GITHUB_BRANCH}
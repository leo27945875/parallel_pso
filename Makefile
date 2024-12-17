GITHUB_BRANCH := ${b}
LIB_NAME      := ${l}
COMMIT_MSG    := ${m}


.PHONY: test clean stubs cm push cpu gpu omp pthread perf_number plot_number scale

test:
	python -m unittest

clean:
	rm -rf build/* */*$(shell python3-config --extension-suffix) */*.a

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
omp:
	python -m core.pyomppso
pthread:
	python -m core.pypthreadpso

perf_number:
	python -m core.perf_number ${c}
plot_number:
	python -m core.plot_number ${c}

perf_scale:
	python -m core.perf_scale ${c}
plot_scale:
	python -m core.plot_scale ${c}
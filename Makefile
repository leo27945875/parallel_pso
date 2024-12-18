GITHUB_BRANCH := ${b}
LIB_NAME      := ${l}
COMMIT_MSG    := ${m}

.PHONY: test clean stubs cm push cpu gpu omp pthread perf_number plot_number perf_scale plot_scale perf_cuda

clean:
	rm -rf build/* */*$(shell python3-config --extension-suffix) */*.a
stubs:
	PYTHONPATH=./ pybind11-stubgen ${LIB_NAME}

cm:
	git add .
	git commit -m '${COMMIT_MSG}'
push:
	git push origin ${GITHUB_BRANCH}

test:
	python3 -m unittest

cpu:
	python3 -m core.pypso
gpu:
	python3 -m core.pycupso
omp:
	python3 -m core.pyomppso
pthread:
	python3 -m core.pypthreadpso

perf_number:
	python3 -m core.perf_number ${c}
plot_number:
	python3 -m core.plot_number ${c}

perf_scale:
	python3 -m core.perf_scale ${c}
plot_scale:
	python3 -m core.plot_scale ${c}

perf_cuda:
	python3 -m core.perf_cuda
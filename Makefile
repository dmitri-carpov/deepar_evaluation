IMAGE := deepmodels
ROOT := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

.PHONY: test datasets

STORAGE_PATH=/mnt/storage

RUNTIME_PARAMETERS := -it --rm --user $(shell id -u) \
                      -e PYTHONPATH=/experiment \
                      -e HOME=/tmp \
                      -w /experiment \
                      -v ${ROOT}:/experiment \
                      -v ${ROOT}/storage:${STORAGE_PATH}

build:
	docker build . -t ${IMAGE}

notebook:
	docker run ${RUNTIME_PARAMETERS} \
        -p 8888:8888 -v ${ROOT}:/project --entrypoint jupyter ${IMAGE} lab \
	    --ip=0.0.0.0 --port=8888 --no-browser \
		--LabApp.token='' \
		--LabApp.allow_remote_access=True \
		--LabApp.allow_origin='*' \
		--LabApp.disable_check_xsrf=True

test:
	docker run ${RUNTIME_PARAMETERS} ${IMAGE} python -m unittest

datasets:
	docker run ${RUNTIME_PARAMETERS} ${IMAGE} python main.py datasets

run:
	docker run ${RUNTIME_PARAMETERS} ${IMAGE} \
	         python main.py run --source_dataset=${source_dataset} \
     							--source_subset=${source_subset} \
								--target_dataset=${target_dataset} \
								--target_subset=${target_subset} \
								--frequency=${frequency} \
								--horizon=${horizon} \
								--model_name=${model_name} \
								--experiments=${experiments}
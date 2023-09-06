.PHONY: Dockerfile

Dockerfile:
	docker run --rm repronim/neurodocker:0.9.5 generate docker \
		--pkg-manager apt \
		--base-image ubuntu:22.04 \
		--install gnupg2 curl gcc ca-certificates software-properties-common python3 pip \
		--run "apt-key adv --fetch-keys https://apt.sr-research.com/SRResearch_key && add-apt-repository 'deb [arch=amd64] https://apt.sr-research.com SRResearch main'" \
		--install eyelink-display-software \
		--run "mkdir /eye2bids" \
		--copy "." "/eye2bids" \
		--workdir "/eye2bids" \
		--run "pip install .[dev]" \
		> Dockerfile

docker_build:
	docker build -t eye2bids .

test_data:
	python tools/download_test_data.py

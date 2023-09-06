.PHONY: Dockerfile

Dockerfile:
	docker run --rm repronim/neurodocker:0.9.5 generate docker \
		--pkg-manager apt \
		--base-image ubuntu:22.04 \
		--install gnupg2 curl gcc ca-certificates software-properties-common \
		--run "apt-key adv --fetch-keys https://apt.sr-research.com/SRResearch_key && add-apt-repository 'deb [arch=amd64] https://apt.sr-research.com SRResearch main'" \
		--install eyelink-display-software \
		> Dockerfile

docker_build: Dockerfile
	docker build -t eye2bids .

test_data:
	python tools/download_test_data.py
